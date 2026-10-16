"""
第 2 版评估汇总（按第 1 轮审稿意见）。所有函数只读取 results/predictions/ 的预测文件与数据集，产出表格。

推断口径（事先规定，写入论文 Methods）：
- 主要比较：重复 3 次的 5 折交叉验证，以 GBM-aggregate 为参照，四个模型 × 两个任务（MCI 转化、CN 进展）
  共 8 个比较构成一个假设族，用校正的重复交叉验证 t 检验（15 折）并做 Holm 校正；
  MCI 转化另做等效性判断（差值 95% CI 落在 ±0.02 内）。
- 其余分析（测试集、消融、亚组、敏感性、筛选实验）均为探索性，报告未校正的 P 值与 95% CI。
- 描述性 CI：按参与者整体重抽样的 bootstrap（2,000 次）。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, roc_auc_score

from src.dataset.build import SCALES
from src.evaluation.metrics import (TASKS, calibration_stats, classification_metrics, cluster_bootstrap,
                                    corrected_resampled_t, holm, task_arrays, task_metrics, weighted_auc)

PROB = ["p_CN", "p_MCI", "p_Dementia"]
N_BOOT = 2000
MODELS = ["Index-dx-only", "Logistic", "GBM-snapshot", "GBM-aggregate", "LongForMAD", "GRU", "CTMT"]
SEQ_MODELS = ["GBM-snapshot", "LongForMAD", "GRU", "CTMT"]
REF = "GBM-aggregate"
TASK_NAME = {"mci_conv": "MCI → dementia", "mci_rev": "MCI → CN", "cn_prog": "CN → MCI/dementia"}


# ---------------------------------------------------------------- 读取与拼接

def load_predictions(root: Path) -> pd.DataFrame:
    files = sorted((root / "results/predictions").glob("*.parquet"))
    p = pd.concat([pd.read_parquet(f).assign(source=f.stem) for f in files], ignore_index=True)
    m = p["eval_set"].str.extract(r"cvr(\d+)f(\d+)")
    p["repeat"] = pd.to_numeric(m[0])
    p["fold"] = pd.to_numeric(m[1])
    return p


def join(pred, samples):
    return pred.merge(samples, on="sample_id", how="inner").sort_values(["repeat", "sample_id"], na_position="first").reset_index(drop=True)


def select(preds, model, eval_set=None, repeat=None, source=None):
    q = preds["model"] == model
    if eval_set is not None:
        q &= preds["eval_set"] == eval_set
    if repeat is not None:
        q &= preds["repeat"] == repeat
    if source is not None:
        q &= preds["source"].str.startswith(source)
    return preds[q]


def fmt(e, lo, hi, nd=3):
    f = lambda v: f"{v:.{nd}f}".replace("-", "−")  # noqa: E731
    return f"{f(e)} ({f(lo)} to {f(hi)})" if (lo < 0 or hi < 0) else f"{f(e)} ({f(lo)}–{f(hi)})"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def all_metrics(d):
    out = classification_metrics(d["y"], d[PROB].to_numpy(), d["origin_mci"])
    out.update(task_metrics(d["y"], d[PROB].to_numpy(), d["origin_mci"]))
    return out


def boot_metrics(d, keys, n_boot=N_BOOT, seed=0):
    """点估计 + 参与者 bootstrap 的 95% CI。d 可以包含同一参与者的多份（多次重复），会整体一起抽。"""
    y, p, om = d["y"].to_numpy(), d[PROB].to_numpy(), d["origin_mci"].to_numpy()

    def fn(i):
        m = classification_metrics(y[i], p[i], om[i])
        m.update(task_metrics(y[i], p[i], om[i]))
        return {k: m[k] for k in keys}
    point = fn(np.arange(len(d)))
    boot = cluster_bootstrap(fn, d["subject_id"].to_numpy(), n_boot, seed)
    return point, boot


# ---------------------------------------------------------------- 主要比较（Table 2、Fig. 2）

MAIN_KEYS = ["mci_conv", "mci_rev", "cn_prog", "macro_auroc", "bal_acc", "brier", "mci_conv_auprc"]


def discrimination_table(preds, samples, models=MODELS):
    rows = []
    for setting, kw in [("Repeated five-fold CV (3 × 5)", dict(source=None)), ("Held-out test set", dict(eval_set="test"))]:
        for m in models:
            q = preds["model"] == m
            if setting.startswith("Repeated"):
                q &= preds["repeat"].notna()
            else:
                q &= preds["eval_set"] == "test"
            d = join(preds[q], samples)
            point, boot = boot_metrics(d, MAIN_KEYS)
            row = {"setting": setting, "model": m, "n_points": d["sample_id"].nunique(),
                   "n_participants": d["subject_id"].nunique()}
            for k in MAIN_KEYS:
                row[k] = fmt(point[k], boot[k].quantile(.025), boot[k].quantile(.975))
                row[k + "_est"] = point[k]
            rows.append(row)
    return pd.DataFrame(rows)


def fold_metrics(preds, samples, models, repeats=(0, 1, 2), keys=("mci_conv", "mci_rev", "cn_prog", "macro_auroc", "bal_acc")):
    rows = []
    for m in models:
        d = join(preds[(preds["model"] == m) & preds["repeat"].isin(repeats)], samples)
        for (r, f), g in d.groupby(["repeat", "fold"]):
            met = all_metrics(g)
            rows.append({"model": m, "repeat": r, "fold": f, **{k: met[k] for k in keys}})
    return pd.DataFrame(rows)


def primary_comparisons(fm, ref=REF, others=SEQ_MODELS, tasks=("mci_conv", "cn_prog"), margin=0.02):
    """事先规定的主要假设族：others − ref，两个任务；校正的重复 CV t 检验 + Holm；转化任务另做等效性判断。"""
    rows = []
    for m in others:
        for t in tasks:
            a = fm[fm.model == m].set_index(["repeat", "fold"])[t]
            b = fm[fm.model == ref].set_index(["repeat", "fold"])[t]
            res = corrected_resampled_t((a - b.reindex(a.index)).to_numpy())
            rows.append({"model": m, "vs": ref, "task": t, **res})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    out["equivalent_within_margin"] = np.where(out["task"] == "mci_conv",
                                                (out["ci_low"] > -margin) & (out["ci_high"] < margin), np.nan)
    return out


def all_pairwise(fm, models, tasks=("mci_conv", "mci_rev", "cn_prog", "macro_auroc")):
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            for t in tasks:
                x = fm[fm.model == a].set_index(["repeat", "fold"])[t]
                y = fm[fm.model == b].set_index(["repeat", "fold"])[t]
                rows.append({"model_a": a, "model_b": b, "task": t, **corrected_resampled_t((x - y.reindex(x.index)).to_numpy())})
    return pd.DataFrame(rows)


def seed_variability(preds, samples, models=("LongForMAD", "GRU", "CTMT")):
    rows = []
    for m in models:
        q = (preds["model"] == m) & preds["eval_set"].str.startswith("test_seed")
        for sd, g in preds[q].groupby("eval_set"):
            met = all_metrics(join(g, samples))
            rows.append({"model": m, "seed": sd, **{t: met[t] for t in TASKS}})
    df = pd.DataFrame(rows)
    return df.groupby("model")[list(TASKS)].agg(["mean", "std", "min", "max"])


def calibration_table(preds, samples, models=MODELS):
    rows = []
    for setting in ["CV repeat 1", "Held-out test set"]:
        for m in models:
            q = (preds["model"] == m) & ((preds["repeat"] == 0) if setting.startswith("CV") else (preds["eval_set"] == "test"))
            d = join(preds[q], samples)
            for t in TASKS:
                yb, s, _ = task_arrays(d["y"], d[PROB].to_numpy(), d["origin_mci"], t)
                c = calibration_stats(yb, s)
                rows.append({"setting": setting, "model": m, "task": t, **c})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 历史（Fig. 3）

def history_span(seq):
    return seq.groupby("sample_id")["days_before_index"].max() / 365.25


SPAN_BINS = [(-0.01, 1.0, "<1 year"), (1.0, 3.0, "1–3 years"), (3.0, 99, "≥3 years")]


def history_strata(preds, samples, seq, models=("GBM-aggregate", "GRU", "CTMT", "LongForMAD"), base="GBM-snapshot"):
    """按历史跨度（年）分层；报告每层的样本构成，以及各历史感知模型相对 GBM-snapshot 的配对差值（CV 3 次重复汇总）。"""
    span = history_span(seq)
    s = samples.assign(span=samples["sample_id"].map(span))
    bl_codes = {"bl", "sc", "4_bl", "4_sc", "init", "4_init", "v01"}
    has_mri = seq.groupby("sample_id")["icv_mm3_missing"].min().eq(0)
    rows, mix = [], []
    for lo, hi, lab in SPAN_BINS:
        ss = s[(s["span"] > lo) & (s["span"] <= hi)] if lo >= 0 else s[s["span"] <= hi]
        mix.append({"stratum": lab, "n_points": len(ss), "n_participants": ss["subject_id"].nunique(),
                    "mci_conv_events": int(((ss.origin_mci == 1) & (ss.y == 2)).sum()),
                    "n_mci": int((ss.origin_mci == 1).sum()),
                    "cn_prog_events": int(((ss.origin_mci == 0) & (ss.y != 0)).sum()), "n_cn": int((ss.origin_mci == 0).sum()),
                    "share_enrolment_visit": ss["visit_code"].isin(bl_codes).mean(),
                    "share_mri_in_history": ss["sample_id"].map(has_mri).mean(), "mean_age": ss["approx_age_years"].mean()})
        dbase = join(preds[(preds.model == base) & preds.repeat.notna()], ss)
        for m in (base,) + tuple(models):
            dm = join(preds[(preds.model == m) & preds.repeat.notna()], ss)
            for t in ("mci_conv", "cn_prog"):
                assert (dm["sample_id"].to_numpy() == dbase["sample_id"].to_numpy()).all()
                ya, sa, msk = task_arrays(dm["y"], dm[PROB].to_numpy(), dm["origin_mci"], t)
                yb_, sb, _ = task_arrays(dbase["y"], dbase[PROB].to_numpy(), dbase["origin_mci"], t)
                subj = dm["subject_id"].to_numpy()[msk]

                def fn(i):
                    ok = 0 < ya[i].sum() < len(i)
                    if not ok:
                        return {"auc": np.nan, "diff": np.nan}
                    a = roc_auc_score(ya[i], sa[i])
                    return {"auc": a, "diff": a - roc_auc_score(yb_[i], sb[i])}
                point = fn(np.arange(len(ya)))
                boot = cluster_bootstrap(fn, subj, N_BOOT)
                rows.append({"stratum": lab, "model": m, "task": t, "auc": point["auc"],
                             "auc_lo": boot["auc"].quantile(.025), "auc_hi": boot["auc"].quantile(.975),
                             "diff_vs_snapshot": point["diff"], "diff_lo": boot["diff"].quantile(.025),
                             "diff_hi": boot["diff"].quantile(.975)})
    return pd.DataFrame(rows), pd.DataFrame(mix)


# ---------------------------------------------------------------- 消融（Fig. 4）

def ablation_table(preds, samples):
    """CV 第 1 次重复（5 折）。差值 = 变体 − 参照；校正 t 检验（5 折）与参与者 bootstrap 两种 CI。"""
    ref_of = lambda name: ("CTMT[refit-full]" if "test-time" in name else  # noqa: E731
                           "GBM-aggregate" if name.startswith("GBM-aggregate[") else
                           "GRU" if name.startswith("GRU[") else "CTMT")
    variants = sorted(m for m in preds["model"].unique()
                      if re.match(r"^(CTMT|GRU|GBM-aggregate)\[", m) and "curated" not in m and "refit-full" not in m
                      and "random-removal" not in m
                      and not re.search(r"\[(all_cause|wide|absorbing)\]", m))
    rows = []
    for v in variants:
        ref = ref_of(v)
        fm = fold_metrics(preds, samples, [v, ref], repeats=(0,))
        dv = join(preds[(preds.model == v) & (preds.repeat == 0)], samples)
        dr = join(preds[(preds.model == ref) & (preds.repeat == 0)], samples).set_index("sample_id").loc[dv["sample_id"]].reset_index()
        pv, bv = boot_metrics(dv, list(TASKS))
        pr, br = boot_metrics(dr, list(TASKS))
        for t in TASKS:
            a = fm[fm.model == v].set_index("fold")[t]
            b = fm[fm.model == ref].set_index("fold")[t]
            ct = corrected_resampled_t((a - b.reindex(a.index)).to_numpy())
            db = bv[t] - br[t]
            rows.append({"variant": v, "reference": ref, "task": t, "auc": pv[t], "ref_auc": pr[t],
                         "diff": pv[t] - pr[t], "boot_lo": db.quantile(.025), "boot_hi": db.quantile(.975),
                         "boot_p": min(1, 2 * min((db <= 0).mean(), (db >= 0).mean())),
                         "t_lo": ct["ci_low"], "t_hi": ct["ci_high"], "t_p": ct["p"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 队列筛选（Table 3、Fig. 5）

def curation_breakdown(samples, curated):
    s = samples.assign(curated=curated.to_numpy())
    removed = s[~s["curated"]]
    tab = pd.crosstab(removed["origin_diagnosis"], removed["target_diagnosis"])
    return tab, removed["subject_id"].nunique()


def _metric_on(d, keys):
    return all_metrics(d) if keys is None else {k: all_metrics(d)[k] for k in keys}


def curation_analysis(preds, samples, curated, models=("GBM-aggregate", "LongForMAD", "GRU", "CTMT"), n_null=500, seed=0):
    """
    CV 第 1 次重复的汇总预测。
    (a) 评估集筛选：同一批预测，筛选子集 − 完整集；与两个零分布比较：随机剔除同样多的参与者、
        按结局类别配对随机剔除同样多的预测点。
    (b) 训练集筛选：筛选训练 − 完整训练（都在完整集上评估）；P(CN) 在 MCI 起点中的平均预测值 vs 实际比例。
    (c) 总乐观偏差：筛选训练 + 筛选评估 − 完整训练 + 完整评估。
    """
    rng = np.random.default_rng(seed)
    s = samples.assign(curated=curated.to_numpy())
    keys = ["mci_conv", "cn_prog", "macro_auroc", "bal_acc"]
    n_removed_subj = s.loc[~s["curated"], "subject_id"].nunique()
    removed_counts = s[~s["curated"]].groupby(["origin_diagnosis", "target_diagnosis"]).size()
    subj = s["subject_id"].unique()
    eval_rows, null_rows, train_rows = [], [], []
    for m in models:
        d = join(preds[(preds.model == m) & (preds.repeat == 0)], s)
        full = _metric_on(d, keys)
        sub = _metric_on(d[d["curated"]], keys)
        # 配对 bootstrap 的 CI
        y, p, om, cur = d["y"].to_numpy(), d[PROB].to_numpy(), d["origin_mci"].to_numpy(), d["curated"].to_numpy()

        def fn(i):
            a = classification_metrics(y[i], p[i], om[i])
            j = i[cur[i]]
            b = classification_metrics(y[j], p[j], om[j])
            return {k: b[k] - a[k] for k in keys}
        boot = cluster_bootstrap(fn, d["subject_id"].to_numpy(), N_BOOT)
        for k in keys:
            eval_rows.append({"model": m, "metric": k, "full": full[k], "curated_subset": sub[k],
                              "diff": sub[k] - full[k], "ci_low": boot[k].quantile(.025),
                              "ci_high": boot[k].quantile(.975)})
        # 零分布
        for kind in ["random participants", "class-matched points"]:
            draws = []
            for _ in range(n_null):
                if kind == "random participants":
                    drop = set(rng.choice(subj, n_removed_subj, replace=False))
                    keep = ~d["subject_id"].isin(drop)
                else:
                    keep = pd.Series(True, index=d.index)
                    for (o, t), n in removed_counts.items():
                        idx = d.index[(d.origin_diagnosis == o) & (d.target_diagnosis == t)]
                        keep.loc[rng.choice(idx, min(n, len(idx)), replace=False)] = False
                mm = _metric_on(d[keep.to_numpy()], ["macro_auroc", "bal_acc", "mci_conv"])
                draws.append({k: mm[k] - full[k] for k in mm})
            dr = pd.DataFrame(draws)
            for k in dr.columns:
                obs = sub[k] - full[k]
                null_rows.append({"model": m, "null": kind, "metric": k, "observed": obs, "null_mean": dr[k].mean(),
                                  "null_lo": dr[k].quantile(.025), "null_hi": dr[k].quantile(.975),
                                  "p_null_ge_observed": (dr[k] >= obs).mean() if obs >= 0 else (dr[k] <= obs).mean()})
        # 训练集筛选
        mc = f"{m}[curated-train]"
        dc = join(preds[(preds.model == mc) & (preds.repeat == 0)], s)
        if len(dc):
            dc = dc.set_index("sample_id").loc[d["sample_id"]].reset_index()
            pc = dc[PROB].to_numpy()
            k2 = ["mci_conv", "mci_rev", "cn_prog", "macro_auroc", "bal_acc"]

            def fn2(i):
                a = classification_metrics(y[i], p[i], om[i])
                b = classification_metrics(y[i], pc[i], om[i])
                return {k: b[k] - a[k] for k in k2}
            point = fn2(np.arange(len(d)))
            boot2 = cluster_bootstrap(fn2, d["subject_id"].to_numpy(), N_BOOT)
            mci = om.astype(bool)
            for k in k2:
                bb = boot2[k].dropna()
                train_rows.append({"model": m, "metric": k, "diff": point[k], "ci_low": bb.quantile(.025),
                                   "ci_high": bb.quantile(.975), "p": min(1, 2 * min((bb <= 0).mean(), (bb >= 0).mean())),
                                   "full_train": classification_metrics(y, p, om)[k],
                                   "curated_train": classification_metrics(y, pc, om)[k]})
            train_rows.append({"model": m, "metric": "mean_P(CN)_in_MCI", "full_train": p[mci, 0].mean(),
                               "curated_train": pc[mci, 0].mean(), "diff": pc[mci, 0].mean() - p[mci, 0].mean(),
                               "observed_rate": (y[mci] == 0).mean()})
            cal_f = calibration_stats((y[mci] == 0).astype(int), p[mci, 0])
            cal_c = calibration_stats((y[mci] == 0).astype(int), pc[mci, 0])
            train_rows.append({"model": m, "metric": "cal_intercept_P(CN)_in_MCI", "full_train": cal_f["cal_intercept"],
                               "curated_train": cal_c["cal_intercept"]})
            # 总乐观偏差
            cc = classification_metrics(y[cur], pc[cur], om[cur])
            ff = classification_metrics(y, p, om)
            for k in ["macro_auroc", "bal_acc", "mci_conv", "cn_prog"]:
                train_rows.append({"model": m, "metric": f"total_optimism_{k}", "full_train": ff[k], "curated_train": cc[k],
                                   "diff": cc[k] - ff[k]})
    return pd.DataFrame(eval_rows), pd.DataFrame(null_rows), pd.DataFrame(train_rows)


# ---------------------------------------------------------------- 其他敏感性分析

def subset_table(preds, samples, models, mask_fn, label, weights=None, repeat=0):
    rows = []
    for m in models:
        d = join(preds[(preds.model == m) & (preds.repeat == repeat)], samples)
        d = d[mask_fn(d)] if mask_fn is not None else d
        for t in ("mci_conv", "cn_prog", "mci_rev"):
            yb, sc, msk = task_arrays(d["y"], d[PROB].to_numpy(), d["origin_mci"], t)
            w = None if weights is None else weights(d)[msk]
            subj = d["subject_id"].to_numpy()[msk]

            def fn(i):
                return {"auc": weighted_auc(yb[i], sc[i], None if w is None else w[i])}
            point = fn(np.arange(len(yb)))
            boot = cluster_bootstrap(fn, subj, N_BOOT)
            rows.append({"analysis": label, "model": m, "task": t, "n": len(yb), "events": int(yb.sum()),
                         "auc": point["auc"], "lo": boot["auc"].quantile(.025), "hi": boot["auc"].quantile(.975)})
    return pd.DataFrame(rows)


def confirmed_outcome(samples, dx):
    """CN 起点的 MCI/痴呆结局是否在下一次诊断被确认（下一次仍为 MCI 或痴呆）。没有下一次诊断视为无法确认，保留为事件。"""
    nxt = []
    by = {k: g for k, g in dx.groupby("subject_id")}
    for r in samples.itertuples():
        g = by[r.subject_id]
        later = g[g["date"] > r.target_date]
        nxt.append(later["diagnosis"].iloc[0] if len(later) else None)
    s = samples.assign(next_dx=nxt)
    s["unconfirmed_progression"] = (s.origin_diagnosis == "CN") & (s.target_diagnosis != "CN") & (s.next_dx == "CN")
    return s


def regression_table(preds, samples, models, repeat=0, ref="LOCF"):
    tcols = [f"future_{s}" for s in SCALES]
    pcols = [f"pred_{s}" for s in SCALES]
    base = join(preds[(preds.model == ref) & (preds.repeat == repeat)], samples)
    rows = []
    for m in models:
        d = join(preds[(preds.model == m) & (preds.repeat == repeat)], samples)
        b = base.set_index("sample_id").loc[d["sample_id"]].reset_index()
        for j, sc in enumerate(SCALES):
            y, yh, yb = d[tcols[j]].to_numpy(float), d[pcols[j]].to_numpy(float), b[pcols[j]].to_numpy(float)
            ok = ~np.isnan(y)

            def fn(i, y=y, yh=yh, yb=yb):
                i = i[ok[i]]
                e, eb = yh[i] - y[i], yb[i] - y[i]
                return {"mae": np.abs(e).mean(), "rmse": np.sqrt((e ** 2).mean()),
                        "dmae": np.abs(e).mean() - np.abs(eb).mean()}
            point = fn(np.arange(len(d)))
            boot = cluster_bootstrap(fn, d["subject_id"].to_numpy(), N_BOOT)
            db = boot["dmae"]
            rows.append({"model": m, "scale": sc, "n": int(ok.sum()), "MAE": point["mae"],
                         "MAE_lo": boot["mae"].quantile(.025), "MAE_hi": boot["mae"].quantile(.975),
                         "RMSE": point["rmse"], f"dMAE_vs_{ref}": point["dmae"], "d_lo": db.quantile(.025),
                         "d_hi": db.quantile(.975), "p": min(1, 2 * min((db <= 0).mean(), (db >= 0).mean()))})
    return pd.DataFrame(rows)


def regression_by_transition(preds, samples, models, repeat=0):
    tcols = [f"future_{s}" for s in SCALES]
    rows = []
    for m in models:
        d = join(preds[(preds.model == m) & (preds.repeat == repeat)], samples)
        d["transition"] = d["origin_diagnosis"] + " → " + d["target_diagnosis"]
        for tr, g in d.groupby("transition"):
            for sc in SCALES:
                ok = g[f"future_{sc}"].notna()
                if ok.sum() >= 10:
                    rows.append({"model": m, "transition": tr, "scale": sc, "n": int(ok.sum()),
                                 "MAE": mean_absolute_error(g.loc[ok, f"future_{sc}"], g.loc[ok, f"pred_{sc}"])})
    return pd.DataFrame(rows).pivot_table(index=["scale", "transition", "n"], columns="model", values="MAE").reset_index()


def substitution_analysis(preds, samples, head_models=("GBM-aggregate", "CTMT")):
    """
    未来分数能否替代未来诊断（主划分）：阈值与多项逻辑回归都在验证集的「预测分数」上拟合，
    目标同为平衡准确率；在测试集上与分类头比较，配对 bootstrap CI。另给「真实未来 CDR-SB」的上限。
    """
    grid = np.arange(0, 18.01, 0.25)

    def best_thr(cdr, y):
        best, thr = -1, None
        for i, a in enumerate(grid):
            for b in grid[i + 1:]:
                ba = balanced_accuracy_score(y, np.digitize(cdr, [a, b]))
                if ba > best:
                    best, thr = ba, (a, b)
        return thr

    rows = []
    va_all, te_all = samples[samples.split == "val"], samples[samples.split == "test"]
    for m in head_models:
        v = join(select(preds, m, "val"), va_all)
        t = join(select(preds, m, "test"), te_all)
        v, t = v[v.future_CDRSB.notna()], t[t.future_CDRSB.notna()]
        thr = best_thr(v["pred_CDRSB"].to_numpy(), v["y"].to_numpy())
        pc = [f"pred_{s}" for s in SCALES]
        Xv = np.column_stack([v[pc].to_numpy(), v["origin_mci"]])
        Xt = np.column_stack([t[pc].to_numpy(), t["origin_mci"]])
        mu, sd = Xv.mean(0), Xv.std(0) + 1e-9
        lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit((Xv - mu) / sd, v["y"])
        y = t["y"].to_numpy()
        head = t[PROB].to_numpy().argmax(1)
        # 公平比较：分类头也在验证集上用同样灵活度的决策规则（对 log 概率 + 起点诊断做多项逻辑回归，平衡类别权重）
        Hv = np.column_stack([np.log(np.clip(v[PROB].to_numpy(), 1e-6, 1)), v["origin_mci"]])
        Ht = np.column_stack([np.log(np.clip(t[PROB].to_numpy(), 1e-6, 1)), t["origin_mci"]])
        hmu, hsd = Hv.mean(0), Hv.std(0) + 1e-9
        head_rc = LogisticRegression(max_iter=5000, class_weight="balanced").fit((Hv - hmu) / hsd, v["y"]).predict((Ht - hmu) / hsd)
        thr_pred = np.digitize(t["pred_CDRSB"].to_numpy(), thr)
        lr_pred = lr.predict((Xt - mu) / sd)
        oracle_thr = best_thr(v["future_CDRSB"].to_numpy(), v["y"].to_numpy())
        oracle = np.digitize(t["future_CDRSB"].to_numpy(), oracle_thr)
        preds_ = {"head A (argmax)": head, "head A, validation-fitted decision rule": head_rc,
                  "forecast CDR-SB thresholds": thr_pred,
                  "multinomial on 4 forecast scores + index dx": lr_pred, "oracle: true future CDR-SB thresholds": oracle}

        def fn(i):
            return {k: balanced_accuracy_score(y[i], v_[i]) for k, v_ in preds_.items()}
        point = fn(np.arange(len(y)))
        boot = cluster_bootstrap(fn, t["subject_id"].to_numpy(), N_BOOT)
        ref_rule = "head A, validation-fitted decision rule"
        for k in preds_:
            dd = boot[k] - boot[ref_rule]
            rows.append({"model": m, "method": k, "n": len(y), "bal_acc": point[k], "lo": boot[k].quantile(.025),
                         "hi": boot[k].quantile(.975), "diff_vs_head": point[k] - point[ref_rule],
                         "d_lo": dd.quantile(.025), "d_hi": dd.quantile(.975),
                         "dementia_missed": int(((preds_[k] != 2) & (y == 2)).sum()), "dementia_n": int((y == 2).sum()),
                         "thresholds": str(thr if "forecast CDR" in k else oracle_thr if "oracle" in k else "")})
    return pd.DataFrame(rows)


def operating_points(preds, samples, models=("GBM-aggregate", "CTMT"), thresholds=(0.2, 0.3, 0.4)):
    rows = []
    for m in models:
        d = join(preds[(preds.model == m) & (preds.repeat == 0)], samples)
        yb, s, _ = task_arrays(d["y"], d[PROB].to_numpy(), d["origin_mci"], "mci_conv")
        for th in thresholds:
            pr = s >= th
            tp, fp = (pr & (yb == 1)).sum(), (pr & (yb == 0)).sum()
            fn_, tn = (~pr & (yb == 1)).sum(), (~pr & (yb == 0)).sum()
            n = len(yb)
            rows.append({"model": m, "threshold": th, "sensitivity": tp / (tp + fn_), "specificity": tn / (tn + fp),
                         "PPV": tp / max(tp + fp, 1), "NPV": tn / max(tn + fn_, 1), "flagged_share": pr.mean(),
                         "net_benefit": tp / n - fp / n * th / (1 - th),
                         "net_benefit_treat_all": yb.mean() - (1 - yb.mean()) * th / (1 - th)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 队列描述（Table 1、流程图、纳入 vs 排除）

def _index_scores(cand, cog):
    """每个候选起点在起点前 90 天内（含当天）最近的 MMSE、CDR-SB、ADAS-Cog13，与模型输入同一规则，不用起点之后的测量。"""
    out = {}
    for sc in ["MMSE", "CDRSB", "ADAS13"]:
        c = cog[cog["scale"] == sc].dropna(subset=["score"])[["subject_id", "date", "score"]].sort_values("date")
        m = pd.merge_asof(cand.sort_values("index_date"), c.rename(columns={"date": "sdate"}), left_on="index_date",
                          right_on="sdate", by="subject_id", direction="backward", tolerance=pd.Timedelta(days=90))
        out[sc] = m.set_index(["subject_id", "index_date"])["score"]
    return pd.DataFrame(out)


def included_vs_excluded(cand, cog):
    sc = _index_scores(cand, cog)
    c = cand.join(sc, on=["subject_id", "index_date"])
    c["age"] = c["index_date"].dt.year - c["birth_year"]
    c["group"] = c["status"].map({"judged": "Included", "insufficient_followup": "Excluded: <21 months follow-up",
                                  "no_dx_in_window": "Excluded: no diagnosis in window",
                                  "duplicate_outcome": "Excluded: shares outcome with later point"})
    rows = {}
    for g, x in c.groupby("group"):
        rows[g] = {"Candidate points, n": len(x), "Participants, n": x["subject_id"].nunique(),
                   "Index MCI, %": 100 * (x["origin_diagnosis"] == "MCI").mean(),
                   "Age, mean": x["age"].mean(), "Male, %": 100 * (x["sex"] == "Male").mean(),
                   "Education, mean y": x["education_years"].mean(), "MMSE, mean": x["MMSE"].mean(),
                   "CDR-SB, mean": x["CDRSB"].mean(), "APOE ε4 carrier, %": 100 * (x["apoe4"] > 0).mean(),
                   "ADNI1/GO/2 index visit, %": 100 * x["protocol"].isin(["ADNI1", "ADNIGO", "ADNI2"]).mean(),
                   "MCI points with dementia diagnosis before month 21, n":
                       int((x["dementia_before_window"] & (x["origin_diagnosis"] == "MCI")).sum())}
    return pd.DataFrame(rows).round(1)


def flow_counts(cand, samples, dx_all_participants):
    return {
        "participants_with_any_dx": dx_all_participants,
        "participants_with_cn_mci_dx": cand["subject_id"].nunique(),
        "candidate_points": len(cand),
        **{f"status_{k}": int(v) for k, v in cand["status"].value_counts().items()},
        "included_points": len(samples), "included_participants": samples["subject_id"].nunique(),
        "participants_no_point": cand["subject_id"].nunique() - samples["subject_id"].nunique(),
    }


def cohort_table(samples, seq, cog):
    mri_any = seq.groupby("sample_id")["icv_mm3_missing"].min().eq(0)
    span = history_span(seq)
    prior_mci = seq[(seq["days_before_index"] > 0) & (seq["dx_MCI"] == 1)].groupby("sample_id").size() > 0
    s = samples.assign(mri_any=samples["sample_id"].map(mri_any).fillna(False),
                       span=samples["sample_id"].map(span),
                       prior_mci=samples["sample_id"].map(prior_mci).fillna(False))
    s = s.join(_index_scores(s[["subject_id", "index_date"]].copy(), cog), on=["subject_id", "index_date"])

    def npct(mask, denom):
        n, d = int(np.sum(mask)), int(np.sum(denom))
        return f"{n}/{d} ({100 * n / d:.1f})"

    def miqr(x, nd=0):
        x = pd.Series(x).dropna()
        q = np.percentile(x, [50, 25, 75])
        return f"{q[0]:.{nd}f} ({q[1]:.{nd}f}–{q[2]:.{nd}f})"
    out = {}
    for name, g in [("Overall", s)] + [(k, s[s["split"] == k]) for k in ["train", "val", "test"]]:
        first = g.sort_values("index_date").groupby("subject_id").head(1)
        cn, mci = g[g.origin_diagnosis == "CN"], g[g.origin_diagnosis == "MCI"]
        out[name] = {
            "Participants, n": g["subject_id"].nunique(),
            "Prediction points, n": len(g),
            "Prediction points per participant, median (IQR)": miqr(g.groupby("subject_id").size()),
            "Age at first prediction point, mean (SD), y": f"{first.approx_age_years.mean():.1f} ({first.approx_age_years.std():.1f})",
            "Male, n/N (%)": npct(first.sex_male == 1, first.sex_male.notna()),
            "Education, mean (SD), y": f"{first.education_years.mean():.1f} ({first.education_years.std():.1f})",
            "White, n/N (%)": npct(first.race == "White", first.race.notna()),
            "Black or African American, n/N (%)": npct(first.race == "Black or African American", first.race.notna()),
            "Hispanic or Latino, n/N (%)": npct(first.ethnicity == "Hispanic or Latino", first.ethnicity.notna()),
            "APOE ε4 carrier, n/N (%)": npct(first.apoe4 > 0, first.apoe4.notna()),
            "Index visit in ADNI1/GO/2, n (%)": npct(g.protocol.isin(["ADNI1", "ADNIGO", "ADNI2"]), np.ones(len(g))),
            "CN index, n": len(cn),
            "  CN → CN, n (%)": npct(cn.target_diagnosis == "CN", np.ones(len(cn))),
            "  CN → MCI, n (%)": npct(cn.target_diagnosis == "MCI", np.ones(len(cn))),
            "  CN → dementia, n (%)": npct(cn.target_diagnosis == "Dementia", np.ones(len(cn))),
            "  CN index with a previous MCI diagnosis, n (%)": npct(cn.prior_mci, np.ones(len(cn))),
            "  CN index: MMSE, median (IQR)": miqr(cn.MMSE),
            "  CN index: CDR-SB, median (IQR)": miqr(cn.CDRSB, 1),
            "MCI index, n": len(mci),
            "  MCI → CN (reversion), n (%)": npct(mci.target_diagnosis == "CN", np.ones(len(mci))),
            "  MCI → MCI, n (%)": npct(mci.target_diagnosis == "MCI", np.ones(len(mci))),
            "  MCI → dementia, n (%)": npct(mci.target_diagnosis == "Dementia", np.ones(len(mci))),
            "  MCI index: MMSE, median (IQR)": miqr(mci.MMSE),
            "  MCI index: CDR-SB, median (IQR)": miqr(mci.CDRSB, 1),
            "Assessment days in history, median (IQR)": miqr(g.n_visits),
            "History span, median (IQR), y": miqr(g.span, 1),
            "≥1 MRI in history, n (%)": npct(g.mri_any, np.ones(len(g))),
        }
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 第 1 轮返修补充（R1-M6、R1-m6 / R2-M8）

def reverter_profile(samples, seq, cog, next_dx):
    """MCI 起点按两年结局分组，描述起点时的特征（逆转者离诊断阈值有多近、此前是否被诊断为 CN、之后的诊断）。"""
    mci = samples[samples["origin_diagnosis"] == "MCI"].copy()
    sc = _index_scores(mci[["subject_id", "index_date"]].drop_duplicates(), cog)
    mci = mci.join(sc, on=["subject_id", "index_date"])
    prior_cn = (seq[seq["days_before_index"] > 0].groupby("sample_id")["dx_CN"].max() == 1)
    mci["prior_cn"] = mci["sample_id"].map(prior_cn).fillna(False)
    mci["next_dx"] = mci["sample_id"].map(next_dx)
    rows = {}
    for out, g in mci.groupby("target_diagnosis"):
        n = len(g)
        rows[f"MCI → {out}"] = {
            "Prediction points, n": n,
            "Age, mean": round(g.approx_age_years.mean(), 1),
            "APOE ε4 carrier, %": round(100 * (g.apoe4 > 0).mean(), 1),
            "Index MMSE, median (IQR)": f"{g.MMSE.median():.0f} ({g.MMSE.quantile(.25):.0f}–{g.MMSE.quantile(.75):.0f})",
            "Index CDR-SB, median (IQR)": f"{g.CDRSB.median():.1f} ({g.CDRSB.quantile(.25):.1f}–{g.CDRSB.quantile(.75):.1f})",
            "Index ADAS-Cog13, median (IQR)": f"{g.ADAS13.median():.1f} ({g.ADAS13.quantile(.25):.1f}–{g.ADAS13.quantile(.75):.1f})",
            "MMSE ≥28 and CDR-SB ≤0.5 at index, %": round(100 * ((g.MMSE >= 28) & (g.CDRSB <= 0.5)).mean(), 1),
            "Previously diagnosed CN, %": round(100 * g.prior_cn.mean(), 1),
            "Index visit in ADNI3/4, %": round(100 * g.protocol.isin(["ADNI3", "ADNI4"]).mean(), 1),
            "Next diagnosis after the outcome: MCI or dementia, %": round(100 * g.next_dx.isin(["MCI", "Dementia"]).mean(), 1),
            "Next diagnosis after the outcome: none recorded, %": round(100 * g.next_dx.isna().mean(), 1),
        }
    return pd.DataFrame(rows)


def reversion_vs_stable(preds, samples, models=MODELS[1:], n_boot=N_BOOT):
    """逆转 vs 稳定 MCI（排除两年后痴呆的点）的 AUROC，三次重复交叉验证合并。"""
    rows = []
    for m in models:
        d = join(preds[(preds["model"] == m) & preds["repeat"].notna()], samples)
        d = d[(d["origin_diagnosis"] == "MCI") & d["target_diagnosis"].isin(["CN", "MCI"])]
        y = (d["target_diagnosis"] == "CN").astype(int).to_numpy()
        s = d["p_CN"].to_numpy()
        point = roc_auc_score(y, s)
        boot = cluster_bootstrap(lambda i: {"auc": roc_auc_score(y[i], s[i]) if 0 < y[i].sum() < len(i) else np.nan},
                                 d["subject_id"].to_numpy(), n_boot)
        rows.append({"model": m, "n": d["sample_id"].nunique(), "events": int(y.sum() / 3), "auc": point,
                     "lo": boot["auc"].quantile(.025), "hi": boot["auc"].quantile(.975)})
    return pd.DataFrame(rows)


def calibration_ci(preds, samples, models=MODELS[1:], n_boot=500):
    """校准截距与斜率的参与者 bootstrap 95% CI（交叉验证第 1 次重复与测试集）。"""
    rows = []
    for setting in ["CV repeat 1", "Held-out test set"]:
        for m in models:
            q = (preds["model"] == m) & ((preds["repeat"] == 0) if setting.startswith("CV") else (preds["eval_set"] == "test"))
            d = join(preds[q], samples)
            for t in TASKS:
                yb, s, msk = task_arrays(d["y"], d[PROB].to_numpy(), d["origin_mci"], t)
                subj = d["subject_id"].to_numpy()[msk]
                c = calibration_stats(yb, s)
                boot = cluster_bootstrap(lambda i: {k: calibration_stats(yb[i], s[i])[k] for k in ("cal_intercept", "cal_slope")}
                                         if 0 < yb[i].sum() < len(i) else {"cal_intercept": np.nan, "cal_slope": np.nan},
                                         subj, n_boot)
                rows.append({"setting": setting, "model": m, "task": t, "events": int(yb.sum()), "n": len(yb),
                             "cal_intercept": c["cal_intercept"], "int_lo": boot.cal_intercept.quantile(.025),
                             "int_hi": boot.cal_intercept.quantile(.975), "cal_slope": c["cal_slope"],
                             "slope_lo": boot.cal_slope.quantile(.025), "slope_hi": boot.cal_slope.quantile(.975),
                             "mean_pred": c["mean_pred"], "obs_rate": c["obs_rate"]})
    return pd.DataFrame(rows)


def total_optimism_ci(preds, samples, curated, models=("GBM-aggregate", "LongForMAD", "GRU", "CTMT", "CTMT-classification-only")):
    """总乐观偏差（筛选训练 + 筛选评估 − 完整训练 + 完整评估）的参与者 bootstrap 95% CI，CV 第 1 次重复。"""
    s = samples.assign(curated=curated.to_numpy())
    keys = ["macro_auroc", "bal_acc", "mci_conv", "cn_prog"]
    rows = []
    for m in models:
        d = join(preds[(preds.model == m) & (preds.repeat == 0)], s)
        dc = join(preds[(preds.model == f"{m}[curated-train]") & (preds.repeat == 0)], s)
        if not len(dc):
            continue
        dc = dc.set_index("sample_id").loc[d["sample_id"]].reset_index()
        y, p, pc = d["y"].to_numpy(), d[PROB].to_numpy(), dc[PROB].to_numpy()
        om, cur = d["origin_mci"].to_numpy(), d["curated"].to_numpy()

        def fn(i):
            a = classification_metrics(y[i], p[i], om[i])
            j = i[cur[i]]
            b = classification_metrics(y[j], pc[j], om[j])
            return {k: b[k] - a[k] for k in keys}
        point = fn(np.arange(len(d)))
        boot = cluster_bootstrap(fn, d["subject_id"].to_numpy(), N_BOOT)
        for k in keys:
            b = boot[k].dropna()
            rows.append({"model": m, "metric": k, "diff": point[k], "ci_low": b.quantile(.025), "ci_high": b.quantile(.975),
                         "p": min(1, 2 * min((b <= 0).mean(), (b >= 0).mean()))})
    return pd.DataFrame(rows)
