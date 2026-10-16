"""
ART 第 1 轮模拟审稿后的补充分析（全部基于已保存的预测，不再训练模型）。输出 results/tables/r_*.csv 与 r_summary.json。
  participant      每位参与者一次的估计量（首个 MCI/CN 预测点；按参与者点数倒数加权）及与 GBM-aggregate 的配对差（R1-M1）
  reversion        参与者层面的逆转率、从首次 MCI 诊断起的两年累积发生率（竞争风险）、痴呆后回退、Logical Memory II（R1-M2）
  scores           训练集筛选后逆转的多种打分、逆转 vs 稳定 MCI、P(痴呆) 的变化、随机剔除对照、回归头 15 折检验（R1-M2、R2-M3）
  decomposition    评估集筛选的分解：只去掉 101 个逆转结局 vs 只去掉其余被筛掉的点（R2-M4）
  amyloid          淀粉样蛋白阳性 MCI 中的配对比较（15 折）、按淀粉样蛋白状态的校准、扫描间隔、同期扫描敏感性（R1-M3、R2-M8）
  equivalence      排除调参参与者后的等效性、90% CI、能成立的最小界值（R2-M1）
  temporal         时间外推验证中与 GBM-aggregate 的配对差（R2-M7）
  subgroups        性别、APOE ε4 亚组间 AUROC 差及 CI（R2-M10）
用法：.venv/bin/python scripts/revision_analyses.py [部分名 ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.dataset.cohort_v2 import load_diagnoses  # noqa: E402
from src.evaluation import report_v2 as R  # noqa: E402
from src.evaluation.metrics import (calibration_stats, classification_metrics, cluster_bootstrap,  # noqa: E402
                                    corrected_resampled_t, task_arrays, weighted_auc)
from src.modeling.data import curated_mask, load_dataset, reversion_flags  # noqa: E402

TAB = ROOT / "results/tables"
PROB = ["p_CN", "p_MCI", "p_Dementia"]
LEARNED = ["GBM-snapshot", "GBM-aggregate", "LongForMAD", "GRU", "CTMT"]
NB = 1000
SUMMARY = {}


def save(df, name):
    df.to_csv(TAB / f"{name}.csv", index=False)
    print("wrote", name, df.shape, flush=True)


def auc(y, s, w=None):
    y = np.asarray(y)
    return weighted_auc(y, np.asarray(s), w) if w is not None else (roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan)


def ci(boot, k):
    b = boot[k].dropna()
    return b.quantile(.025), b.quantile(.975)


def cv(preds, samples, model, repeats=(0, 1, 2)):
    return R.join(preds[(preds.model == model) & preds["repeat"].isin(repeats)], samples)


# ---------------------------------------------------------------- 参与者层面的估计量

def participant(preds, samples):
    s = samples.sort_values("index_date")
    first_mci = s[s.origin_diagnosis == "MCI"].groupby("subject_id").head(1)["sample_id"]
    first_cn = s[s.origin_diagnosis == "CN"].groupby("subject_id").head(1)["sample_id"]
    n_mci = samples[samples.origin_diagnosis == "MCI"].groupby("subject_id").size()
    n_cn = samples[samples.origin_diagnosis == "CN"].groupby("subject_id").size()

    def arrays(d, task, how):
        yb, sc, msk = task_arrays(d["y"], d[PROB].to_numpy(), d["origin_mci"], task)
        dd = d[msk]
        if how == "first":
            keep = dd["sample_id"].isin(first_mci if task == "mci_conv" else first_cn).to_numpy()
            return yb[keep], sc[keep], None, dd["subject_id"].to_numpy()[keep]
        cnt = (n_mci if task == "mci_conv" else n_cn)
        return yb, sc, 1.0 / dd["subject_id"].map(cnt).to_numpy(), dd["subject_id"].to_numpy()

    pooled, tests = [], []
    for how in ["first", "weighted"]:
        for task in ["mci_conv", "cn_prog"]:
            fold = {}
            for m in ["Logistic"] + LEARNED:
                d = cv(preds, samples, m)
                yb, sc, w, subj = arrays(d, task, how)

                def fn(i):
                    return {"auc": auc(yb[i], sc[i], None if w is None else w[i])}
                b = cluster_bootstrap(fn, subj, NB)
                lo, hi = ci(b, "auc")
                pooled.append({"estimand": how, "task": task, "model": m, "n": len(yb) // 3, "events": int(yb.sum()) // 3,
                               "auc": fn(np.arange(len(yb)))["auc"], "lo": lo, "hi": hi})
                fold[m] = {}
                for (r, f), g in d.groupby(["repeat", "fold"]):
                    y2, s2, w2, _ = arrays(g, task, how)
                    fold[m][(r, f)] = auc(y2, s2, w2)
            ref = pd.Series(fold["GBM-aggregate"])
            for m in ["GBM-snapshot", "LongForMAD", "GRU", "CTMT"]:
                diff = (pd.Series(fold[m]) - ref.reindex(pd.Series(fold[m]).index)).to_numpy()
                res = corrected_resampled_t(diff)
                tests.append({"estimand": how, "task": task, "model": m, **res,
                              "within_002": (res["ci_low"] > -0.02) and (res["ci_high"] < 0.02)})
    save(pd.DataFrame(pooled), "r_participant_level")
    save(pd.DataFrame(tests), "r_participant_level_tests")


# ---------------------------------------------------------------- 逆转

def aalen_johansen(times, kinds, horizon):
    """kinds: 0 删失, 1 逆转（CN）, 2 痴呆。返回 horizon 时两种事件的累积发生率。"""
    order = np.argsort(times)
    t, k = np.asarray(times)[order], np.asarray(kinds)[order]
    surv, cif1, cif2, n = 1.0, 0.0, 0.0, len(t)
    i = 0
    while i < len(t) and t[i] <= horizon:
        tt = t[i]
        j = i
        while j < len(t) and t[j] == tt:
            j += 1
        at_risk = n - i
        d1, d2 = int((k[i:j] == 1).sum()), int((k[i:j] == 2).sum())
        cif1 += surv * d1 / at_risk
        cif2 += surv * d2 / at_risk
        surv *= 1 - (d1 + d2) / at_risk
        i = j
    return cif1, cif2


def first_mci_episodes(dx):
    """每位参与者第一次 MCI 诊断起，下一个不同的诊断（CN = 逆转、痴呆）及时间；没有则在最后一次诊断删失。"""
    rows = []
    for sid, g in dx.sort_values("date").groupby("subject_id"):
        g = g.reset_index(drop=True)
        idx = g.index[g["diagnosis"] == "MCI"]
        if not len(idx):
            continue
        t0 = g.loc[idx[0], "date"]
        later = g.loc[idx[0] + 1:]
        other = later[later["diagnosis"] != "MCI"]
        if len(other):
            r = other.iloc[0]
            rows.append({"subject_id": sid, "years": (r["date"] - t0).days / 365.25,
                         "kind": 1 if r["diagnosis"] == "CN" else 2})
        else:
            last = g["date"].iloc[-1]
            rows.append({"subject_id": sid, "years": (last - t0).days / 365.25, "kind": 0})
    return pd.DataFrame(rows)


def reversion(samples, seq):
    dx = load_diagnoses(ROOT)
    out = {}
    mci = samples[samples.origin_diagnosis == "MCI"]
    part = mci.groupby("subject_id").agg(any_rev=("target_diagnosis", lambda x: (x == "CN").any()),
                                         first_idx=("index_date", "min"))
    out["mci_participants"] = int(len(part))
    out["mci_participants_with_reversion_outcome"] = int(part.any_rev.sum())
    later_cn = []
    by = {k: g for k, g in dx.groupby("subject_id")}
    for sid, r in part.iterrows():
        g = by[sid]
        later_cn.append(bool(((g["date"] > r.first_idx) & (g["diagnosis"] == "CN")).any()))
    out["mci_participants_ever_later_cn"] = int(sum(later_cn))
    ep = first_mci_episodes(dx)
    for label, e in [("all_adni_mci", ep), ("cohort_mci", ep[ep.subject_id.isin(part.index)])]:
        c1, c2 = aalen_johansen(e["years"].to_numpy(), e["kind"].to_numpy(), 2.0)
        out[f"{label}_n"] = int(len(e))
        out[f"{label}_cif2y_reversion"] = round(c1, 4)
        out[f"{label}_cif2y_dementia"] = round(c2, 4)
    # 按 MCI 持续时间：该预测点是否参与者的第一个 MCI 诊断日
    first_mci_date = dx[dx.diagnosis == "MCI"].groupby("subject_id")["date"].min()
    mm = mci.assign(first=lambda d: pd.to_datetime(d.index_date) == d.subject_id.map(first_mci_date))
    for k, g in mm.groupby("first"):
        out[f"reversion_rate_{'first_mci_visit' if k else 'later_mci_visits'}"] = round(float((g.target_diagnosis == "CN").mean()), 4)
        out[f"n_{'first_mci_visit' if k else 'later_mci_visits'}"] = int(len(g))
    # 痴呆后回退
    rev = reversion_flags(ROOT)
    dem_subj = rev.index[rev["dem_to_lower"]]
    out["participants_dementia_to_lower_any"] = int(len(dem_subj))
    first_dem = dx[dx.diagnosis == "Dementia"].groupby("subject_id")["date"].min()
    s2 = samples.assign(fd=samples.subject_id.map(first_dem))
    after = s2[s2.fd.notna() & (pd.to_datetime(s2.index_date) > s2.fd)]
    out["points_index_after_prior_dementia"] = int(len(after))
    out["participants_index_after_prior_dementia"] = int(after.subject_id.nunique())
    out["points_index_after_prior_dementia_by_outcome"] = after.target_diagnosis.value_counts().to_dict()
    # Logical Memory II（延迟回忆）在指数日前 180 天内最近的一次
    nb = list(pyreadr.read_r(str(ROOT / "data/raw/adni/ADNIMERGE2/data/NEUROBAT.rda")).values())[0]
    nb = nb.dropna(subset=["RID", "LDELTOTAL", "VISDATE"]).copy()
    nb["subject_id"] = nb.RID.astype("int64")
    nb["date"] = pd.to_datetime(nb.VISDATE)
    m = mci[["sample_id", "subject_id", "index_date", "target_diagnosis"]].assign(index_date=lambda d: pd.to_datetime(d.index_date))
    m = m.merge(nb[["subject_id", "date", "LDELTOTAL"]], on="subject_id")
    m = m[(m.date <= m.index_date) & (m.date >= m.index_date - pd.Timedelta(days=180))]
    m = m.sort_values("date").groupby("sample_id").tail(1)
    lm = {}
    for o, g in m.groupby("target_diagnosis"):
        q = g.LDELTOTAL.quantile([.25, .5, .75])
        lm[o] = {"n": int(len(g)), "median": float(q[.5]), "q1": float(q[.25]), "q3": float(q[.75])}
    out["logical_memory_II_by_outcome_mci"] = lm
    SUMMARY["reversion"] = out
    print(json.dumps(out, indent=1, default=str))


# ---------------------------------------------------------------- 逆转的打分方式、P(痴呆)、随机剔除对照、回归头

SCORES = {"P(CN)": lambda p: p[:, 0], "-P(dementia)": lambda p: -p[:, 2],
          "P(CN)/(P(CN)+P(MCI))": lambda p: p[:, 0] / (p[:, 0] + p[:, 1]), "-E[class]": lambda p: -(p[:, 1] + 2 * p[:, 2])}


def _rev_metrics(d, p):
    mci = d["origin_mci"].to_numpy().astype(bool)
    y = d["y"].to_numpy()[mci]
    pm = p[mci]
    rev, stable = y == 0, y == 1
    out = {}
    for name, f in SCORES.items():
        out[f"rev_auc[{name}]"] = auc(rev.astype(int), f(pm))
        sel = rev | stable
        out[f"rev_vs_stable_auc[{name}]"] = auc(rev[sel].astype(int), f(pm[sel]))
    out["mean_P(CN)|reverters"] = pm[rev, 0].mean()
    out["mean_P(CN)|all MCI"] = pm[:, 0].mean()
    out["mean_P(dementia)|reverters"] = pm[rev, 2].mean()
    out["mean_P(dementia)|stable MCI"] = pm[stable, 2].mean()
    out["mean_P(dementia)|all MCI"] = pm[:, 2].mean()
    out["obs_dementia|all MCI"] = (y == 2).mean()
    c = calibration_stats((y == 2).astype(int), pm[:, 2])
    out["cal_intercept_P(dementia)"] = c["cal_intercept"]
    out["cal_slope_P(dementia)"] = c["cal_slope"]
    full = classification_metrics(d["y"].to_numpy(), p, d["origin_mci"].to_numpy())
    for k in ["mci_conv", "cn_prog", "macro_auroc", "bal_acc"]:
        out[k] = full[k]
    return out


def scores(preds, samples):
    rows = []
    pairs = [(m, f"{m}[curated-train]") for m in ["GBM-aggregate", "LongForMAD", "GRU", "CTMT", "CTMT-classification-only"]]
    pairs += [("GBM-aggregate", f"GBM-aggregate[random-removal-{k}]") for k in range(3)] + [("CTMT", "CTMT[random-removal-0]")]
    for base, alt in pairs:
        d = cv(preds, samples, base, (0,))
        da = cv(preds, samples, alt, (0,))
        if not len(da):
            continue
        da = da.set_index("sample_id").loc[d["sample_id"]].reset_index()
        p, pa = d[PROB].to_numpy(), da[PROB].to_numpy()
        a, b = _rev_metrics(d, p), _rev_metrics(d, pa)
        mci = d["origin_mci"].to_numpy().astype(bool)
        y = d["y"].to_numpy()

        def fn(i, f=SCORES["-P(dementia)"]):
            ii = i[mci[i]]
            return {"d_rev_negPdem": auc((y[ii] == 0).astype(int), f(pa[ii])) - auc((y[ii] == 0).astype(int), f(p[ii])),
                    "d_rev_PCN": auc((y[ii] == 0).astype(int), pa[ii, 0]) - auc((y[ii] == 0).astype(int), p[ii, 0])}
        bt = cluster_bootstrap(fn, d["subject_id"].to_numpy(), NB)
        for k in a:
            rows.append({"base": base, "variant": alt, "metric": k, "full_train": a[k], "variant_train": b[k],
                         "diff": b[k] - a[k]})
        for k in ["d_rev_negPdem", "d_rev_PCN"]:
            lo, hi = ci(bt, k)
            rows.append({"base": base, "variant": alt, "metric": k + "_CI", "diff": fn(np.arange(len(d)))[k], "lo": lo, "hi": hi})
    save(pd.DataFrame(rows), "r_reversion_scores")

    # 回归头：15 折上的差中差（逆转 AUROC，P(CN) 打分）
    def fold_rev(model):
        d = cv(preds, samples, model)
        return pd.Series({(r, f): classification_metrics(g["y"], g[PROB].to_numpy(), g["origin_mci"])["mci_rev"]
                          for (r, f), g in d.groupby(["repeat", "fold"])})
    ctmt, ctmt_c = fold_rev("CTMT"), fold_rev("CTMT[curated-train]")
    nob, nob_c = fold_rev("CTMT-classification-only"), fold_rev("CTMT-classification-only[curated-train]")
    gbm, gbm_c = fold_rev("GBM-aggregate"), fold_rev("GBM-aggregate[curated-train]")
    idx = ctmt_c.index.intersection(nob_c.index)
    rows = [{"contrast": "CTMT curated − full", **corrected_resampled_t((ctmt_c - ctmt).reindex(idx).to_numpy())},
            {"contrast": "CTMT without head B curated − full", **corrected_resampled_t((nob_c - nob).reindex(idx).to_numpy())},
            {"contrast": "difference in differences (CTMT − CTMT without head B)",
             **corrected_resampled_t(((ctmt_c - ctmt) - (nob_c - nob)).reindex(idx).to_numpy())},
            {"contrast": "GBM-aggregate curated − full", **corrected_resampled_t((gbm_c - gbm).reindex(gbm_c.index).to_numpy())}]
    save(pd.DataFrame(rows), "r_headB_15folds")


# ---------------------------------------------------------------- 评估集筛选的分解

def decomposition(preds, samples):
    cur = curated_mask(samples, reversion_flags(ROOT)).to_numpy()
    rev_outcome = ((samples.origin_diagnosis == "MCI") & (samples.target_diagnosis == "CN")).to_numpy()
    s = samples.assign(cur=cur, rev_outcome=rev_outcome)
    keys = ["macro_auroc", "bal_acc", "mci_conv", "cn_prog"]
    rows = []
    for m in ["GBM-aggregate", "LongForMAD", "GRU", "CTMT"]:
        d = cv(preds, s, m, (0,))
        y, p, om = d["y"].to_numpy(), d[PROB].to_numpy(), d["origin_mci"].to_numpy()
        masks = {"remove 101 reversion outcomes only": ~d["rev_outcome"].to_numpy(),
                 "remove other curated points only": d["cur"].to_numpy() | d["rev_outcome"].to_numpy(),
                 "remove all curated points": d["cur"].to_numpy()}
        for name, keep in masks.items():
            def fn(i, keep=keep):
                a = classification_metrics(y[i], p[i], om[i])
                j = i[keep[i]]
                b = classification_metrics(y[j], p[j], om[j])
                return {k: b[k] - a[k] for k in keys}
            pt = fn(np.arange(len(d)))
            bt = cluster_bootstrap(fn, d["subject_id"].to_numpy(), NB)
            for k in keys:
                lo, hi = ci(bt, k)
                rows.append({"model": m, "removal": name, "n_removed": int((~keep).sum()), "metric": k, "diff": pt[k],
                             "lo": lo, "hi": hi})
    save(pd.DataFrame(rows), "r_curation_decomposition")


# ---------------------------------------------------------------- 淀粉样蛋白

def amyloid_status(samples, window=None):
    amy = list(pyreadr.read_r(str(ROOT / "data/raw/adni/ADNIMERGE2/data/UCBERKELEY_AMY_6MM.rda")).values())[0]
    amy = amy[amy["qc_flag"].isin([1, 2]) & amy["AMYLOID_STATUS"].notna()].copy()
    amy["subject_id"] = amy["RID"].astype("int64")
    amy["scan_date"] = pd.to_datetime(amy["SCANDATE"])
    m = samples[["sample_id", "subject_id", "index_date"]].assign(index_date=lambda d: pd.to_datetime(d["index_date"]))
    m = m.merge(amy[["subject_id", "scan_date", "AMYLOID_STATUS"]], on="subject_id")
    m["gap"] = (m["scan_date"] - m["index_date"]).dt.days
    if window is None:          # 主分析：指数日当天或之前最近的一次
        m = m[m.gap <= 0].sort_values("gap").groupby("sample_id").tail(1)
    else:                       # 同期扫描：指数日前 window[0] 天至后 window[1] 天内离指数日最近的一次
        m = m[(m.gap >= -window[0]) & (m.gap <= window[1])].assign(a=lambda d: d.gap.abs()).sort_values("a").groupby("sample_id").head(1)
    return m.set_index("sample_id")


def amyloid(preds, samples):
    main = amyloid_status(samples)
    contemp = amyloid_status(samples, (365, 90))
    s = samples.assign(amy=samples.sample_id.map(main["AMYLOID_STATUS"]), amy_c=samples.sample_id.map(contemp["AMYLOID_STATUS"]),
                       lag=-samples.sample_id.map(main["gap"]) / 365.25)
    lag = s.loc[(s.origin_diagnosis == "MCI") & s.amy.notna(), "lag"]
    SUMMARY["amyloid_lag_years_mci"] = {"median": round(lag.median(), 2), "q1": round(lag.quantile(.25), 2),
                                        "q3": round(lag.quantile(.75), 2), "max": round(lag.max(), 1),
                                        "share_over_2y": round(float((lag > 2).mean()), 3)}
    rows, tests, cal = [], [], []
    for col, label in [("amy", "latest scan on or before index"), ("amy_c", "scan within 1 year before to 90 days after index")]:
        for status in [1, 0, "known"]:
            fold = {}
            for m in ["Logistic"] + LEARNED:
                d = cv(preds, s, m)
                d = d[(d[col].notna() if status == "known" else (d[col] == status)) & (d.origin_diagnosis == "MCI")]
                yb, sc = (d["y"] == 2).astype(int).to_numpy(), d["p_Dementia"].to_numpy()
                b = cluster_bootstrap(lambda i: {"auc": auc(yb[i], sc[i])}, d["subject_id"].to_numpy(), NB)
                lo, hi = ci(b, "auc")
                rows.append({"definition": label, "amyloid": {1: "positive", 0: "negative"}.get(status, "known"), "model": m,
                             "n": len(d) // 3, "events": int(yb.sum()) // 3, "auc": auc(yb, sc), "lo": lo, "hi": hi})
                fold[m] = pd.Series({(r, f): auc((g["y"] == 2).astype(int), g["p_Dementia"]) for (r, f), g in d.groupby(["repeat", "fold"])})
                if col == "amy" and status != "known":
                    c = calibration_stats(yb, sc)
                    bc = cluster_bootstrap(lambda i: {k: calibration_stats(yb[i], sc[i])[k] for k in ("cal_intercept", "cal_slope")},
                                           d["subject_id"].to_numpy(), 300)
                    cal.append({"amyloid": "positive" if status else "negative", "model": m, "mean_pred": sc.mean(),
                                "observed": yb.mean(), **{k: c[k] for k in ("cal_intercept", "cal_slope")},
                                "int_lo": ci(bc, "cal_intercept")[0], "int_hi": ci(bc, "cal_intercept")[1],
                                "slope_lo": ci(bc, "cal_slope")[0], "slope_hi": ci(bc, "cal_slope")[1]})
            if status == 1:
                for m in ["GBM-snapshot", "LongForMAD", "GRU", "CTMT"]:
                    res = corrected_resampled_t((fold[m] - fold["GBM-aggregate"].reindex(fold[m].index)).dropna().to_numpy())
                    tests.append({"definition": label, "model": m, **res,
                                  "within_002": (res["ci_low"] > -0.02) and (res["ci_high"] < 0.02)})
    save(pd.DataFrame(rows), "r_amyloid_auc")
    save(pd.DataFrame(tests), "r_amyloid_paired")
    save(pd.DataFrame(cal), "r_amyloid_calibration")


# ---------------------------------------------------------------- 等效性的稳健性

def equivalence(preds, samples):
    val_subj = set(samples.loc[samples.split == "val", "subject_id"])
    s = samples[~samples.subject_id.isin(val_subj)]
    fm = R.fold_metrics(preds, s, LEARNED)
    rows = []
    for m in ["GBM-snapshot", "LongForMAD", "GRU", "CTMT"]:
        for t in ["mci_conv", "cn_prog"]:
            a = fm[fm.model == m].set_index(["repeat", "fold"])[t]
            b = fm[fm.model == "GBM-aggregate"].set_index(["repeat", "fold"])[t]
            rows.append({"analysis": "excluding tuning participants", "model": m, "task": t,
                         **corrected_resampled_t((a - b.reindex(a.index)).to_numpy())})
    pc = pd.read_csv(TAB / "t2_primary_comparisons.csv")
    full = R.fold_metrics(preds, samples, LEARNED)
    from scipy import stats
    for m in ["GBM-snapshot", "LongForMAD", "GRU", "CTMT"]:
        a = full[full.model == m].set_index(["repeat", "fold"])["mci_conv"]
        b = full[full.model == "GBM-aggregate"].set_index(["repeat", "fold"])["mci_conv"]
        dlt = (a - b.reindex(a.index)).to_numpy()
        n = len(dlt)
        se = np.sqrt((1 / n + 0.2 / (0.8 * 6 / 7)) * dlt.var(ddof=1))
        q = stats.t.ppf(0.95, n - 1)
        rows.append({"analysis": "90% CI (TOST at alpha 0.05)", "model": m, "task": "mci_conv", "diff": dlt.mean(),
                     "ci_low": dlt.mean() - q * se, "ci_high": dlt.mean() + q * se})
    out = pd.DataFrame(rows)
    save(out, "r_equivalence_robustness")
    conv = pc[pc.task == "mci_conv"]
    SUMMARY["smallest_margin_95"] = round(float(np.maximum(conv.ci_low.abs(), conv.ci_high.abs()).max()), 4)
    e = out[(out.analysis == "excluding tuning participants") & (out.task == "mci_conv")]
    SUMMARY["smallest_margin_95_excl_tuning"] = round(float(np.maximum(e.ci_low.abs(), e.ci_high.abs()).max()), 4)
    ap = pd.read_csv(TAB / "s_all_pairwise.csv")
    c = ap[ap.task == "mci_conv"]
    SUMMARY["all_pairs_conv_within_002"] = int(((c.ci_low > -0.02) & (c.ci_high < 0.02)).sum())
    SUMMARY["all_pairs_conv_total"] = int(len(c))
    SUMMARY["all_pairs_conv_outside"] = c[~((c.ci_low > -0.02) & (c.ci_high < 0.02))][["model_a", "model_b", "ci_low", "ci_high"]].round(4).to_dict("records")
    fm_sd = full.groupby("model")[["mci_conv", "mci_rev", "cn_prog"]].std().round(3)
    SUMMARY["fold_sd"] = fm_sd.to_dict()


# ---------------------------------------------------------------- 时间外推的配对差

def temporal(preds, samples):
    rows = []
    ref = R.join(preds[(preds.model == "GBM-aggregate") & (preds.eval_set == "temporal")], samples)
    for m in ["Logistic", "GBM-snapshot", "LongForMAD", "GRU", "CTMT"]:
        d = R.join(preds[(preds.model == m) & (preds.eval_set == "temporal")], samples)
        d = d.set_index("sample_id").loc[ref["sample_id"]].reset_index()
        y, om = ref["y"].to_numpy(), ref["origin_mci"].to_numpy()
        pa, pb = d[PROB].to_numpy(), ref[PROB].to_numpy()
        keys = ["mci_conv", "mci_rev", "cn_prog"]

        def fn(i):
            a, b = classification_metrics(y[i], pa[i], om[i]), classification_metrics(y[i], pb[i], om[i])
            return {k: a[k] - b[k] for k in keys}
        pt = fn(np.arange(len(y)))
        bt = cluster_bootstrap(fn, ref["subject_id"].to_numpy(), NB)
        for k in keys:
            lo, hi = ci(bt, k)
            rows.append({"model": m, "task": k, "diff_vs_GBM_aggregate": pt[k], "lo": lo, "hi": hi})
    save(pd.DataFrame(rows), "r_temporal_paired")


# ---------------------------------------------------------------- 亚组差异

def subgroups(preds, samples):
    rows = []
    groups = {"male − female": (lambda d: d.sex_male == 1, lambda d: d.sex_male == 0),
              "APOE ε4 carrier − non-carrier": (lambda d: d.apoe4 > 0, lambda d: d.apoe4 == 0),
              "age ≥75 − <75": (lambda d: d.approx_age_years >= 75, lambda d: d.approx_age_years < 75)}
    for m in ["GBM-aggregate", "CTMT"]:
        d = cv(preds, samples, m, (0,))
        for name, (fa, fb) in groups.items():
            for task in ["mci_conv", "cn_prog"]:
                yb, sc, msk = task_arrays(d["y"], d[PROB].to_numpy(), d["origin_mci"], task)
                dd = d[msk]
                ga, gb = fa(dd).to_numpy(), fb(dd).to_numpy()

                def fn(i):
                    ia, ib = i[ga[i]], i[gb[i]]
                    return {"d": auc(yb[ia], sc[ia]) - auc(yb[ib], sc[ib])}
                bt = cluster_bootstrap(fn, dd["subject_id"].to_numpy(), NB)
                lo, hi = ci(bt, "d")
                rows.append({"model": m, "contrast": name, "task": task, "auc_a": auc(yb[ga], sc[ga]), "auc_b": auc(yb[gb], sc[gb]),
                             "diff": fn(np.arange(len(yb)))["d"], "lo": lo, "hi": hi})
    save(pd.DataFrame(rows), "r_subgroup_differences")


# ---------------------------------------------------------------- 预测变量在指数日的可得性（补充表）

def availability(samples, seq):
    """每个预测变量在指数日前 90 天内（含指数日）至少有一个值的预测点数；静态协变量按是否缺失计。"""
    s = seq[seq["days_before_index"] <= 90]
    rows = []
    for name, col in [("ADAS-Cog13", "ADAS13"), ("CDR-SB", "CDRSB"), ("MMSE", "MMSE"), ("MoCA", "MoCA"),
                      ("MRI (FreeSurfer)", "hippocampus_left_mm3")]:
        n = s.loc[s[col].notna(), "sample_id"].nunique()
        rows.append((name, n))
    for name, col in [("Sex", "sex_male"), ("Age", "approx_age_years"), ("Education", "education_years"),
                      ("Race", "race"), ("Ethnicity", "ethnicity"), ("APOE ε4", "apoe4")]:
        rows.append((name, int(samples[col].notna().sum())))
    n = len(samples)
    df = pd.DataFrame([{"Predictor": a, "available_n": b, "available_pct": round(100 * b / n, 1), "missing_n": n - b,
                        "missing_pct": round(100 * (n - b) / n, 1)} for a, b in rows])
    save(df, "s_predictor_availability")


# ---------------------------------------------------------------- 删失：被排除的 MCI 预测点去向与转化的竞争风险累积发生率（R1 第 2 轮）

def _deaths_withdrawals():
    """TREATDIS（仅 ADNI1/GO/2）：死亡日期与正式退出日期。ADNI1 的死亡编码为 2，ADNI-GO/2 为 1；可多选。"""
    td = list(pyreadr.read_r(str(ROOT / "data/raw/adni/ADNIMERGE2/data/TREATDIS.rda")).values())[0]
    td = td.dropna(subset=["RID", "EXAMDATE"]).copy()
    td["subject_id"] = td["RID"].astype("int64")
    td["date"] = pd.to_datetime(td["EXAMDATE"].astype(str).str.replace("xx", "15"), errors="coerce")  # 部分记录只有年月
    td = td.dropna(subset=["date"])
    codes = [{x.strip().lstrip("0") for x in str(v).split(":")} if v == v else set() for v in td["WDREASON"]]
    death_code = np.where(td["COLPROT"] == "ADNI1", "2", "1")
    td["death"] = [c in s for c, s in zip(death_code, codes)]
    td["full"] = td["WDRAWTYPE"] == 1
    death = td[td.death].groupby("subject_id")["date"].min()
    wd = td[td.full & ~td.death].groupby("subject_id")["date"].min()
    return death, wd


def censoring(samples):
    cand = pd.read_parquet(ROOT / "data/processed/dataset_v2/candidates.parquet")
    dx = load_diagnoses(ROOT)
    death, wd = _deaths_withdrawals()
    m = cand[cand.origin_diagnosis == "MCI"].copy()
    m["index_date"] = pd.to_datetime(m["index_date"])
    horizon = pd.Timedelta(days=int(27 * 30.44))
    m["died"] = m.subject_id.map(death).pipe(lambda s: s.notna() & (s > m.index_date) & (s <= m.index_date + horizon))
    m["withdrew"] = m.subject_id.map(wd).pipe(lambda s: s.notna() & (s > m.index_date) & (s <= m.index_date + horizon))
    exc = m[m.status.isin(["insufficient_followup", "no_dx_in_window"])]
    cat = np.select([exc.dementia_before_window, exc.died, exc.withdrew, exc.protocol.isin(["ADNI3", "ADNI4"])],
                    ["dementia before month 21", "died", "withdrew", "ADNI3/4 (follow-up ongoing or not recorded)"],
                    "other (missed or later visit)")
    out = {"excluded_mci_points": int(len(exc)), "excluded_mci_by_reason": pd.Series(cat).value_counts().to_dict(),
           "included_mci_points_died_or_withdrew_before_outcome": 0}
    # 竞争风险：从每个 MCI 候选预测点起，首个非 MCI 诊断（痴呆 / CN）或死亡，最后一次诊断时删失；两年累积发生率。
    # 对全部候选点与纳入分析的点各算一次，两者用同一个估计量，可直接比较（R2 第 3 轮）
    by = {k: g.sort_values("date") for k, g in dx.groupby("subject_id")}

    def aj(rows):
        times, kinds = [], []
        for r in rows.itertuples():
            g = by.get(r.subject_id)
            later = g[g["date"] > r.index_date] if g is not None else pd.DataFrame()
            other = later[later["diagnosis"] != "MCI"]
            dd = death.get(r.subject_id)
            t_ev, k_ev = None, 0
            if len(other):
                t_ev, k_ev = other["date"].iloc[0], (2 if other["diagnosis"].iloc[0] == "Dementia" else 1)
            if dd is not None and dd > r.index_date and (t_ev is None or dd < t_ev):
                t_ev, k_ev = dd, 3
            if t_ev is None:
                t_ev = later["date"].iloc[-1] if len(later) else r.index_date
            times.append((t_ev - r.index_date).days / 365.25)
            kinds.append(k_ev)
        tt, kk = np.asarray(times), np.asarray(kinds)
        o = np.argsort(tt)
        tt, kk = tt[o], kk[o]
        surv, cif = 1.0, {1: 0.0, 2: 0.0, 3: 0.0}
        n, i = len(tt), 0
        while i < n and tt[i] <= 2.0:
            j = i
            while j < n and tt[j] == tt[i]:
                j += 1
            at = n - i
            d = {k: int((kk[i:j] == k).sum()) for k in (1, 2, 3)}
            for k in cif:
                cif[k] += surv * d[k] / at
            surv *= 1 - sum(d.values()) / at
            i = j
        return n, cif

    for label, rows in [("all_mci_candidates", m[m.status != "duplicate_outcome"]), ("analysed_mci_points", m[m.status == "judged"])]:
        n, cif = aj(rows)
        out[f"aj_n_{label}"] = int(n)
        out[f"cif2y_dementia_{label}"] = round(cif[2], 4)
        out[f"cif2y_reversion_{label}"] = round(cif[1], 4)
        out[f"cif2y_death_{label}"] = round(cif[3], 4)
    SUMMARY["censoring"] = out
    print(json.dumps(out, indent=1, default=str))


def main(parts):
    samples, seq = load_dataset(ROOT)
    preds = R.load_predictions(ROOT)
    want = lambda k: not parts or k in parts  # noqa: E731
    for k, fn in [("participant", lambda: participant(preds, samples)), ("reversion", lambda: reversion(samples, seq)),
                  ("scores", lambda: scores(preds, samples)), ("decomposition", lambda: decomposition(preds, samples)),
                  ("amyloid", lambda: amyloid(preds, samples)), ("equivalence", lambda: equivalence(preds, samples)),
                  ("temporal", lambda: temporal(preds, samples)), ("subgroups", lambda: subgroups(preds, samples)),
                  ("availability", lambda: availability(samples, seq)), ("censoring", lambda: censoring(samples))]:
        if want(k):
            fn()
    path = TAB / "r_summary.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(SUMMARY)
    path.write_text(json.dumps(old, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
