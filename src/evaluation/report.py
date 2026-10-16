"""
阶段 5：把 results/predictions/ 里的全部预测汇总成论文用的表格（results/tables/）。
所有置信区间均为按患者重抽样的 bootstrap（2000 次）；模型间比较为配对 bootstrap。
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset.build import SCALES
from src.evaluation.metrics import (classification_metrics, cluster_bootstrap, evaluate_classifier,
                                    paired_difference, regression_metrics)
from src.modeling.data import curated_mask, load_dataset, reversion_flags

PROB = ["p_CN", "p_MCI", "p_Dementia"]
CLS_METRICS = ["mci_conv", "mci_rev", "cn_prog", "macro_auroc", "mci_conv_auprc", "bal_acc", "macro_f1", "brier"]
N_BOOT = 2000


def load_predictions(root: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((root / "results/predictions").glob("*.parquet"))])


def _join(pred, samples):
    return pred.merge(samples, on="sample_id", how="left").sort_values("sample_id").reset_index(drop=True)


def fmt(est, lo, hi, nd=3):
    return f"{est:.{nd}f} ({lo:.{nd}f}–{hi:.{nd}f})"


def classification_table(preds, samples, models, eval_set="test", ref=None):
    """每个模型一行：点估计 + 95% CI；若给 ref，则另附与 ref 的配对差值和 p 值。"""
    rows, boots, points = [], {}, {}
    for m in models:
        d = _join(preds[(preds["model"] == m) & (preds["eval_set"].str.startswith(eval_set))], samples)
        summ, boot = evaluate_classifier(d["y"], d[PROB].to_numpy(), d["origin_mci"], d["subject_id"], N_BOOT)
        boots[m], points[m] = boot, summ["estimate"].to_dict()
        row = {"model": m, "n_samples": len(d), "n_subjects": d["subject_id"].nunique()}
        for k in CLS_METRICS:
            row[k] = fmt(*summ.loc[k, ["estimate", "ci_low", "ci_high"]])
        rows.append(row)
    table = pd.DataFrame(rows)
    diffs = None
    if ref is not None:
        out = []
        for m in models:
            if m == ref:
                continue
            pdiff = paired_difference(boots[m], boots[ref], points[m], points[ref])
            for k, r in pdiff.iterrows():
                out.append({"model": m, "vs": ref, "metric": k, **r.to_dict()})
        diffs = pd.DataFrame(out)
    return table, diffs, points


def regression_table(preds, samples, models, eval_set="test", ref="LOCF"):
    rows = []
    tcols = [f"future_{s}" for s in SCALES]
    pcols = [f"pred_{s}" for s in SCALES]
    base = _join(preds[(preds["model"] == ref) & (preds["eval_set"].str.startswith(eval_set))], samples)
    for m in models:
        d = _join(preds[(preds["model"] == m) & (preds["eval_set"].str.startswith(eval_set))], samples)
        y, yh, yb = d[tcols].to_numpy(float), d[pcols].to_numpy(float), base[pcols].to_numpy(float)

        def fn(i, y=y, yh=yh, yb=yb):
            a, b = regression_metrics(y[i], yh[i]), regression_metrics(y[i], yb[i])
            return {**{f"mae_{s}": a[f"mae_{j}"] for j, s in enumerate(SCALES)},
                    **{f"dmae_{s}": a[f"mae_{j}"] - b[f"mae_{j}"] for j, s in enumerate(SCALES)}}

        point = fn(np.arange(len(d)))
        boot = cluster_bootstrap(fn, d["subject_id"], N_BOOT)
        for j, s in enumerate(SCALES):
            n = int((~np.isnan(y[:, j])).sum())
            dm = boot[f"dmae_{s}"]
            p = min(1.0, 2 * min((dm <= 0).mean(), (dm >= 0).mean()))
            rows.append({"model": m, "scale": s, "n": n,
                         "MAE": fmt(point[f"mae_{s}"], boot[f"mae_{s}"].quantile(.025), boot[f"mae_{s}"].quantile(.975), 2),
                         f"ΔMAE vs {ref}": fmt(point[f"dmae_{s}"], dm.quantile(.025), dm.quantile(.975), 2),
                         "p": p})
    return pd.DataFrame(rows)


def history_strata(preds, samples, models, eval_set="cv"):
    """按序列长度分层的 MCI 转化与 CN 进展 AUROC（用 5 折 CV 的全部样本，分层后每组样本才够）。"""
    bins = [(1, 1, "1"), (2, 4, "2–4"), (5, 8, "5–8"), (9, 99, "≥9")]
    rows = []
    for m in models:
        d = _join(preds[(preds["model"] == m) & (preds["eval_set"].str.startswith(eval_set))], samples)
        for lo, hi, lab in bins:
            g = d[(d["n_visits"] >= lo) & (d["n_visits"] <= hi)]
            cm = classification_metrics(g["y"], g[PROB].to_numpy(), g["origin_mci"])
            rows.append({"model": m, "history_steps": lab, "n_samples": len(g),
                         "n_mci_conv_events": int(((g["origin_mci"] == 1) & (g["y"] == 2)).sum()),
                         "n_cn_prog_events": int(((g["origin_mci"] == 0) & (g["y"] != 0)).sum()),
                         "mci_conv": cm["mci_conv"], "cn_prog": cm["cn_prog"], "macro_auroc": cm["macro_auroc"]})
    return pd.DataFrame(rows)


def curation_tables(root, preds, samples, models_full, models_cur):
    """
    (a) 训练队列筛选：完整队列训练 vs 筛选队列训练，都在完整测试集上评估
    (b) 测试队列筛选：同一个（完整队列训练的）模型，分别在完整测试集和筛选后测试子集上评估 → 乐观偏差
    """
    cur = curated_mask(samples, reversion_flags(root))
    samples = samples.assign(curated=cur.to_numpy())
    rows = []
    for mf, mc in zip(models_full, models_cur):
        for label, m in [("full-cohort training", mf), ("curated training", mc)]:
            d = _join(preds[(preds["model"] == m) & (preds["eval_set"] == "test")], samples)
            for tset, g in [("full test set", d), ("curated test subset", d[d["curated"]])]:
                summ, _ = evaluate_classifier(g["y"], g[PROB].to_numpy(), g["origin_mci"], g["subject_id"], N_BOOT)
                row = {"architecture": mf, "training cohort": label, "evaluation set": tset, "n_samples": len(g)}
                for k in ["mci_conv", "mci_rev", "cn_prog", "macro_auroc", "bal_acc"]:
                    row[k] = fmt(*summ.loc[k, ["estimate", "ci_low", "ci_high"]])
                rows.append(row)
    return pd.DataFrame(rows)


def cohort_table(samples, seq):
    """Table 1：各数据子集的队列特征。"""
    mri_any = seq.groupby("sample_id")["icv_mm3_missing"].min().eq(0)
    s = samples.assign(mri_any=samples["sample_id"].map(mri_any).fillna(False))
    rows = {}
    for name, g in [("Overall", s)] + [(k, s[s["split"] == k]) for k in ["train", "val", "test"]]:
        first = g.sort_values("index_date").groupby("subject_id").head(1)
        rows[name] = {
            "Participants, n": g["subject_id"].nunique(),
            "Index visits (samples), n": len(g),
            "Samples per participant, median (IQR)": _miqr(g.groupby("subject_id").size()),
            "Age at first index visit, mean (SD), y": f"{first['approx_age_years'].mean():.1f} ({first['approx_age_years'].std():.1f})",
            "Male, n (%)": _np(first["sex_male"] == 1, first["sex_male"].notna()),
            "Education, mean (SD), y": f"{first['education_years'].mean():.1f} ({first['education_years'].std():.1f})",
            "Index diagnosis CN, n (%)": _np(g["origin_diagnosis"] == "CN"),
            "Index diagnosis MCI, n (%)": _np(g["origin_diagnosis"] == "MCI"),
            "CN → CN": _np((g["origin_diagnosis"] == "CN") & (g["target_diagnosis"] == "CN")),
            "CN → MCI": _np((g["origin_diagnosis"] == "CN") & (g["target_diagnosis"] == "MCI")),
            "CN → dementia": _np((g["origin_diagnosis"] == "CN") & (g["target_diagnosis"] == "Dementia")),
            "MCI → CN (reversion)": _np((g["origin_diagnosis"] == "MCI") & (g["target_diagnosis"] == "CN")),
            "MCI → MCI": _np((g["origin_diagnosis"] == "MCI") & (g["target_diagnosis"] == "MCI")),
            "MCI → dementia": _np((g["origin_diagnosis"] == "MCI") & (g["target_diagnosis"] == "Dementia")),
            "History length, visit days, median (IQR)": _miqr(g["n_visits"]),
            "History span, median (IQR), y": _miqr(g["sample_id"].map(seq.groupby("sample_id")["days_before_index"].max() / 365.25), 1),
            "≥1 MRI in history, n (%)": _np(g["mri_any"]),
        }
    return pd.DataFrame(rows)


def _miqr(x, nd=0):
    q = np.nanpercentile(x, [50, 25, 75])
    return f"{q[0]:.{nd}f} ({q[1]:.{nd}f}–{q[2]:.{nd}f})"


def _np(mask, denom=None):
    n = int(np.sum(mask))
    tot = int(np.sum(denom)) if denom is not None else len(mask)
    return f"{n} ({100 * n / tot:.1f})"


def score_threshold_analysis(preds, samples, model="CTMT"):
    """
    头 B 能否替代头 A：
      oracle —— 用真实未来 CDR-SB 穷举最优双阈值反推诊断的准确率上限
      model  —— 用模型预测的未来 CDR-SB + 在训练集真实值上选出的阈值推诊断，与头 A 的 argmax 对比
    """
    tr = samples[(samples["split"] == "train") & samples["future_CDRSB"].notna()]
    grid = np.arange(0, 18.01, 0.25)

    def best_thresholds(cdr, y):
        best = (0, None)
        for i, a in enumerate(grid):
            for b in grid[i + 1:]:
                acc = (np.digitize(cdr, [a, b]) == y).mean()
                if acc > best[0]:
                    best = (acc, (a, b))
        return best

    all_lab = samples[samples["future_CDRSB"].notna()]
    oracle_acc, oracle_thr = best_thresholds(all_lab["future_CDRSB"].to_numpy(), all_lab["y"].to_numpy())
    dem = all_lab["y"] == 2
    miss_dem = ((np.digitize(all_lab["future_CDRSB"], oracle_thr) != 2) & dem).sum()
    _, thr = best_thresholds(tr["future_CDRSB"].to_numpy(), tr["y"].to_numpy())
    d = _join(preds[(preds["model"] == model) & (preds["eval_set"] == "test")], samples)
    d = d[d["future_CDRSB"].notna()]
    acc_b = (np.digitize(d["pred_CDRSB"], thr) == d["y"]).mean()
    acc_a = (d[PROB].to_numpy().argmax(1) == d["y"]).mean()
    from sklearn.metrics import balanced_accuracy_score
    return pd.DataFrame([
        {"analysis": "Oracle: true future CDR-SB, optimal thresholds (all samples)", "n": len(all_lab),
         "accuracy": oracle_acc, "thresholds": oracle_thr, "dementia missed": f"{miss_dem}/{int(dem.sum())}"},
        {"analysis": f"{model} head B: predicted CDR-SB + training thresholds (test)", "n": len(d),
         "accuracy": acc_b, "balanced_accuracy": balanced_accuracy_score(d["y"], np.digitize(d["pred_CDRSB"], thr)),
         "thresholds": thr},
        {"analysis": f"{model} head A: argmax of class probabilities (same test samples)", "n": len(d),
         "accuracy": acc_a, "balanced_accuracy": balanced_accuracy_score(d["y"], d[PROB].to_numpy().argmax(1))},
    ])


def curation_differences(root, preds, samples, models_full, models_cur):
    """
    筛选实验的配对差值：
      training —— 筛选训练 − 完整训练（都在完整测试集上，同一组患者重抽样）
      evaluation —— 同一个完整训练模型：筛选测试子集上的指标 − 完整测试集上的指标
    """
    cur = curated_mask(samples, reversion_flags(root))
    samples = samples.assign(curated=cur.to_numpy())
    rows = []
    for mf, mc in zip(models_full, models_cur):
        a = _join(preds[(preds["model"] == mf) & (preds["eval_set"] == "test")], samples)
        b = _join(preds[(preds["model"] == mc) & (preds["eval_set"] == "test")], samples)
        pa, pb = a[PROB].to_numpy(), b[PROB].to_numpy()
        y, om, cm = a["y"].to_numpy(), a["origin_mci"].to_numpy(), a["curated"].to_numpy()

        def fn(i):
            ma = classification_metrics(y[i], pa[i], om[i])
            mb = classification_metrics(y[i], pb[i], om[i])
            j = i[cm[i]]
            ms = classification_metrics(y[j], pa[j], om[j])
            out = {f"train_{k}": mb[k] - ma[k] for k in ["mci_conv", "mci_rev", "cn_prog", "macro_auroc", "bal_acc"]}
            out.update({f"eval_{k}": ms[k] - ma[k] for k in ["mci_conv", "cn_prog", "macro_auroc", "bal_acc"]})
            return out

        point = fn(np.arange(len(a)))
        boot = cluster_bootstrap(fn, a["subject_id"], N_BOOT)
        for k, v in point.items():
            d = boot[k].dropna()
            rows.append({"architecture": mf, "comparison": k.split("_")[0], "metric": k.split("_", 1)[1],
                         "diff": v, "ci_low": d.quantile(.025), "ci_high": d.quantile(.975),
                         "p": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
    return pd.DataFrame(rows)
