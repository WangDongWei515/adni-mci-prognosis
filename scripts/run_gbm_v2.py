"""
第 2 版（按第 1 轮审稿意见）表格模型与朴素基线。数据集：dataset_v2。与神经网络实验分进程运行（OpenMP 冲突）。

用法：.venv/bin/python scripts/run_gbm_v2.py <stage>
  tune         GBM-snapshot / GBM-aggregate 各 8 组超参数 × 2 个种子，在主划分验证集上按对数损失选择
  primary      主划分：两个 GBM（5 种子集成）、仅起点诊断基线、逻辑回归基线、「分数不变」与漂移校正回归基线
  cv           3 次重复 × 5 折交叉验证，同上全部模型（GBM 3 种子集成，与神经网络 CV 一致）
  ablation     第 0 次重复中的 GBM-aggregate 消融（回看窗口、无既往诊断、无 CDR-SB/MMSE、加 APOE、无 MRI）
  curated      第 0 次重复中，用 LongForMAD 式筛选后的训练集训练 GBM-aggregate
  sensitivity  三个敏感性队列（all_cause / wide / absorbing）的交叉验证：两个 GBM 与基线
  temporal     时间外推验证：ADNI1/GO/2 入组者训练，ADNI3/4 新入组者测试
"""
import lightgbm  # noqa: F401  必须先于任何可能间接加载 torch 的模块导入
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.build import SCALES, STATIC_FEATURES  # noqa: E402
from src.modeling import gbm  # noqa: E402
from src.dataset.cohort_v2 import VISIT_VALUE_FEATURES  # noqa: E402
from src.modeling.cv import CV_PARTITION_SEEDS, fold_samples, make_cv_folds, temporal_split  # noqa: E402
from src.modeling.data import curated_mask, load_dataset, reversion_flags, tabular_features  # noqa: E402
from src.modeling.pipeline import drift_locf_regression, locf_regression, pred_frame, run_gbm  # noqa: E402

OUT = ROOT / "results/predictions"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
CV_SEEDS = [0, 1, 2]
TCOLS = [f"future_{s}" for s in SCALES]


def splits(s):
    return [s[s["split"] == k].reset_index(drop=True) for k in ["train", "val", "test"]]


def params():
    j = json.loads((ROOT / "results/gbm_tuning.json").read_text())
    return {k: v["best"] for k, v in j.items()}


def stage_tune(samples, seq):
    path = ROOT / "results/gbm_tuning.json"
    if path.exists():
        return
    tr, va, _ = splits(samples)
    res = {}
    for mode in ["snapshot", "aggregate"]:
        ftr, fva = tabular_features(tr, seq, mode), tabular_features(va, seq, mode)
        rows = []
        for p in gbm.GRID:
            ll = [gbm.val_logloss(gbm.fit_classifier(ftr, tr["y"], fva, va["y"], sd, p), fva, va["y"]) for sd in (100, 101)]
            rows.append({**p, "val_ce_seeds": ll, "val_ce": float(np.mean(ll))})
            print(mode, p, np.round(ll, 4), flush=True)
        best = min(rows, key=lambda r: r["val_ce"])
        res[f"GBM-{mode}"] = {"best": {k: best[k] for k in ("learning_rate", "num_leaves", "min_child_samples")},
                              "grid": rows}
    path.write_text(json.dumps(res, indent=2))


# ---------------------------------------------------------------- 朴素 / 简单基线

def index_dx_baseline(tr, te):
    """仅用起点诊断：预测概率 = 训练集中同一起点诊断的结局类别比例（「诊断不变」的概率化版本）。"""
    freq = pd.crosstab(tr["origin_diagnosis"], tr["y"], normalize="index").reindex(columns=[0, 1, 2], fill_value=0)
    return freq.loc[te["origin_diagnosis"]].to_numpy(float)


def logistic_baseline(tr, te, seq):
    """多项逻辑回归：起点诊断 + 起点前半年内最近一次的 MMSE、CDR-SB + 年龄（缺失用训练集中位数填补）。"""
    cols = ["MMSE_idx", "CDRSB_idx"]

    def feats(s):
        f = tabular_features(s, seq, "snapshot")[cols].copy()
        f["origin_mci"] = s["origin_mci"].to_numpy()
        f["age"] = s["approx_age_years"].to_numpy()
        return f
    ftr, fte = feats(tr), feats(te)
    med = ftr.median()
    mu, sd = ftr.fillna(med).mean(), ftr.fillna(med).std()
    lr = LogisticRegression(max_iter=2000)
    lr.fit((ftr.fillna(med) - mu) / sd, tr["y"])
    return lr.predict_proba((fte.fillna(med) - mu) / sd)


def all_models(tr, va, te, seq, tag, seeds, p, with_baselines=True, prefix="", **gbm_kw):
    out = []
    for mode in ["snapshot", "aggregate"]:
        pr, rg = run_gbm(tr, va, [te], seq, mode, seeds, p[f"GBM-{mode}"], **gbm_kw)[0][0]
        out.append(pred_frame(f"{prefix}GBM-{mode}", te, pr, rg, tag))
    if with_baselines:
        out.append(pred_frame(f"{prefix}Index-dx-only", te, index_dx_baseline(tr, te), None, tag))
        out.append(pred_frame(f"{prefix}Logistic", te, logistic_baseline(tr, te, seq), None, tag))
        dummy = np.full((len(te), 3), 1 / 3)
        out.append(pred_frame(f"{prefix}LOCF", te, dummy, locf_regression(te, seq, tr[TCOLS].mean().to_numpy()), tag))
        out.append(pred_frame(f"{prefix}Drift-LOCF", te, dummy, drift_locf_regression(te, seq, tr), tag))
    return out


def stage_primary(samples, seq):
    path = OUT / "gbm_primary.parquet"
    if path.exists():
        return
    tr, va, te = splits(samples)
    p = params()
    frames = all_models(tr, va, te, seq, "test", SEEDS, p)
    # 验证集预测（供「用预测分数反推诊断」的阈值在验证集上选择）
    pv, rv = run_gbm(tr, va, [va], seq, "aggregate", SEEDS, p["GBM-aggregate"])[0][0]
    frames.append(pred_frame("GBM-aggregate", va, pv, rv, "val"))
    pd.concat(frames).to_parquet(path)
    print("primary done", flush=True)


def cv_loop(samples, seq, r, fn):
    folds = make_cv_folds(samples, seed=CV_PARTITION_SEEDS[r])
    frames = []
    for f in range(5):
        s = fold_samples(samples, folds, f, seed=CV_PARTITION_SEEDS[r])
        frames += fn(s, f"cvr{r}f{f}")
        print("cv", r, f, flush=True)
    return frames


def stage_cv(samples, seq):
    path = OUT / "gbm_cv.parquet"
    if path.exists():
        return
    p = params()
    frames = []
    for r in range(len(CV_PARTITION_SEEDS)):
        frames += cv_loop(samples, seq, r, lambda s, tag: all_models(*splits(s), seq, tag, CV_SEEDS, p))
    pd.concat(frames).to_parquet(path)


def stage_ablation(samples, seq):
    path = OUT / "gbm_ablation.parquet"
    if path.exists():
        return
    p = params()["GBM-aggregate"]
    dx_cols = ("dx_CN", "dx_MCI", "dx_DEM")
    variants = {"window-1y": dict(max_years=1.0), "window-2y": dict(max_years=2.0), "window-3y": dict(max_years=3.0),
                "no-prior-dx": dict(drop_features=dx_cols), "no-CDRSB-MMSE": dict(drop_features=("CDRSB", "MMSE")),
                "plus-APOE": dict(static_cols=tuple(STATIC_FEATURES) + ("apoe4",)),
                "no-MRI": dict(drop_features=tuple(c for c in VISIT_VALUE_FEATURES if c not in SCALES and c not in dx_cols)),
                # 返修（R1-M4）：诊断信息与 CDR-SB/MMSE 全部去掉（含指数诊断），只剩 ADAS-Cog13、MoCA、MRI 与人口学
                "no-dx-CDRSB-MMSE": dict(drop_features=dx_cols + ("CDRSB", "MMSE"), index_dx=False)}

    def fn(s, tag):
        tr, va, te = splits(s)
        out = []
        for name, kw in variants.items():
            pr, _ = run_gbm(tr, va, [te], seq, "aggregate", CV_SEEDS, p, regression=False, **kw)[0][0]
            out.append(pred_frame(f"GBM-aggregate[{name}]", te, pr, None, tag))
        return out
    pd.concat(cv_loop(samples, seq, 0, fn)).to_parquet(path)


def stage_curated(samples, seq):
    """
    LongForMAD 式筛选后训练 GBM-aggregate，三次重复都做；测试折保持完整。
    返修（R2-M3）加对照：在训练/验证集中随机剔除参与者（整人剔除），直到剔除的预测点数与筛选相同，
    以区分"训练数据变少"与"去掉逆转结局"两种效应（第 0 次重复，3 次随机抽取）。
    """
    path = OUT / "gbm_curated.parquet"
    if path.exists():
        return
    p = params()["GBM-aggregate"]
    cur = pd.Series(curated_mask(samples, reversion_flags(ROOT)).to_numpy(), index=samples["sample_id"])

    def fit(tr, va, te, name, tag):
        pr, _ = run_gbm(tr.reset_index(drop=True), va.reset_index(drop=True), [te], seq, "aggregate", CV_SEEDS, p,
                        regression=False)[0][0]
        return pred_frame(name, te, pr, None, tag)

    def fn_factory(r):
        def fn(s, tag):
            keep = s["sample_id"].map(cur).to_numpy()
            dev = (s["split"] != "test").to_numpy()
            te = s[~dev].reset_index(drop=True)
            out = [fit(s[(s["split"] == "train").to_numpy() & keep], s[(s["split"] == "val").to_numpy() & keep], te,
                       "GBM-aggregate[curated-train]", tag)]
            if r == 0:
                n_drop = int((dev & ~keep).sum())
                subj = s.loc[dev, "subject_id"].drop_duplicates().to_numpy()
                counts = s.loc[dev].groupby("subject_id").size()
                for k in range(3):
                    order = np.random.default_rng(1000 + k).permutation(subj)
                    cum = counts.loc[order].cumsum().to_numpy()
                    gone = set(order[:int(np.searchsorted(cum, n_drop)) + 1])
                    kp = ~s["subject_id"].isin(gone).to_numpy()
                    out.append(fit(s[(s["split"] == "train").to_numpy() & kp], s[(s["split"] == "val").to_numpy() & kp], te,
                                   f"GBM-aggregate[random-removal-{k}]", tag))
            return out
        return fn
    frames = []
    for r in range(len(CV_PARTITION_SEEDS)):
        frames += cv_loop(samples, seq, r, fn_factory(r))
    pd.concat(frames).to_parquet(path)


def stage_sensitivity():
    path = OUT / "gbm_sensitivity.parquet"
    if path.exists():
        return
    p = params()
    frames = []
    for variant in ["all_cause", "wide", "absorbing"]:
        samples, seq = load_dataset(ROOT, variant)
        frames += cv_loop(samples, seq, 0, lambda s, tag: all_models(*splits(s), seq, tag, CV_SEEDS, p,
                                                                     prefix=f"[{variant}]"))
    pd.concat(frames).to_parquet(path)


def stage_temporal(samples, seq):
    path = OUT / "gbm_temporal.parquet"
    if path.exists():
        return
    s = temporal_split(samples)
    pd.concat(all_models(*splits(s), seq, "temporal", SEEDS, params())).to_parquet(path)


def main():
    stage = sys.argv[1]
    if stage == "sensitivity":
        return stage_sensitivity()
    samples, seq = load_dataset(ROOT)
    {"tune": stage_tune, "primary": stage_primary, "cv": stage_cv, "ablation": stage_ablation,
     "curated": stage_curated, "temporal": stage_temporal}[stage](samples, seq)


if __name__ == "__main__":
    main()
