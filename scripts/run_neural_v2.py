"""
第 2 版（按第 1 轮审稿意见）神经网络实验。数据集：dataset_v2。

用法：.venv/bin/python scripts/run_neural_v2.py <stage> [参数]
  tune                      三个模型 × 8 组超参数 × 2 个种子，在主划分验证集上选择（R2-M3：两个种子降低选择噪声）
  primary                   主划分：三个模型 5 种子集成；保存测试集集成预测、逐种子预测、验证集预测
  cv <r>                    第 r 次重复的 5 折交叉验证（r = 0,1,2），三个模型 × 3 种子
  ablation <组名>           第 0 次重复的交叉验证中做 CTMT/GRU 消融（组名见 ABLATION_GROUPS）
  curated                   第 0 次重复的交叉验证中，用 LongForMAD 式筛选后的训练集训练
  sensitivity <队列>        敏感性分析队列（all_cause / wide / absorbing）的交叉验证，CTMT 与 GRU
  temporal                  时间外推验证：ADNI1/GO/2 入组者训练，ADNI3/4 入组者测试
产出：results/predictions/nn_*.parquet；已存在则跳过（断点续跑）
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.build import STATIC_FEATURES  # noqa: E402
from src.modeling.cv import CV_PARTITION_SEEDS, fold_samples, make_cv_folds, temporal_split  # noqa: E402
from src.modeling.data import curated_mask, fit_normalizer, load_dataset, reversion_flags  # noqa: E402
from src.modeling.pipeline import (GROUPS, build_view, per_seed_predictions, pred_frame, reg_to_raw,  # noqa: E402
                                   run_neural)
from src.modeling.train import TrainConfig  # noqa: E402

torch.set_num_threads(2)
OUT = ROOT / "results/predictions"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
CV_SEEDS = [0, 1, 2]
TUNE_SEEDS = [100, 101]
GRID = [dict(d_model=d, dropout=p, lr=lr) for d in (32, 64) for p in (0.1, 0.3) for lr in (1e-3, 3e-4)]
KIND = {"CTMT": "sequence", "GRU": "sequence", "LongForMAD": "annual"}
MODELS = ["CTMT", "LongForMAD", "GRU"]
APOE_STATIC = tuple(STATIC_FEATURES) + ("apoe4",)

# 消融：名称 → (模型, 视图参数, 训练参数, 静态变量)
ABLATIONS = {
    "no-time-encoding": ("CTMT", {}, {"use_time": False}, None),
    "window-1y": ("CTMT", {"window": 1.0}, {}, None),
    "window-2y": ("CTMT", {"window": 2.0}, {}, None),
    "window-3y": ("CTMT", {"window": 3.0}, {}, None),
    "no-MRI": ("CTMT", {"drop": GROUPS["MRI"]}, {}, None),
    "MRI-only": ("CTMT", {"drop": GROUPS["COG"]}, {}, None),
    "no-prior-dx": ("CTMT", {"drop": GROUPS["DX"]}, {}, None),
    "no-CDRSB-MMSE": ("CTMT", {"drop": GROUPS["CDR_MMSE"]}, {}, None),
    "no-step-dropout": ("CTMT", {}, {"step_dropout": 0.0}, None),
    "classification-only": ("CTMT", {}, {"reg_weight": 0.0}, None),
    "snapshot-retrained": ("CTMT", {"kind": "snapshot"}, {}, None),
    "plus-APOE": ("CTMT", {}, {}, APOE_STATIC),
    "GRU-window-1y": ("GRU", {"window": 1.0}, {}, None),
    "GRU-window-2y": ("GRU", {"window": 2.0}, {}, None),
    "GRU-window-3y": ("GRU", {"window": 3.0}, {}, None),
}
ABLATION_GROUPS = {
    "A": ["no-time-encoding", "window-1y", "window-2y", "window-3y", "snapshot-retrained"],
    "B": ["no-MRI", "MRI-only", "no-prior-dx", "no-CDRSB-MMSE", "plus-APOE"],
    "C": ["no-step-dropout", "classification-only", "GRU-window-1y", "GRU-window-2y", "GRU-window-3y"],
}


def splits(samples):
    return {k: samples[samples["split"] == k].reset_index(drop=True) for k in ["train", "val", "test"]}


def make_cfg(model, hp, **over):
    cfg = TrainConfig(lr=hp["lr"], model_kwargs={"d_model": hp["d_model"], "dropout": hp["dropout"]},
                      reg_weight=0.0 if model == "LongForMAD" else 0.5)
    for k, v in over.items():
        if k == "use_time":
            cfg.model_kwargs[k] = v
        else:
            setattr(cfg, k, v)
    return cfg


def hps():
    return {m: v["best"] for m, v in json.loads((ROOT / "results/tuning.json").read_text()).items()}


def fit_predict(model, hp, sp, seq, norm, tests, seeds, view_kw=None, cfg_kw=None):
    view_kw = dict(view_kw or {})
    kind = view_kw.pop("kind", KIND[model])
    views = {k: build_view(sp[k], seq, norm, kind, **view_kw) for k in ["train", "val"]}
    test_views = {name: build_view(s, seq, norm, kind, **view_kw) for name, s in tests.items()}
    preds, vl, models = run_neural(model, make_cfg(model, hp, **(cfg_kw or {})), views["train"], views["val"],
                                   test_views, seeds, norm)
    return preds, vl, models, test_views


def frame(model, s, pr, norm, tag):
    p, r = pr
    return pred_frame(model, s, p, None if r is None else reg_to_raw(r, norm), tag)


def stage_tune(samples, seq):
    path = ROOT / "results/tuning.json"
    if path.exists():
        return
    sp, norm = splits(samples), fit_normalizer(samples, seq)
    res = {}
    for model in MODELS:
        rows = []
        for hp in GRID:
            t = time.time()
            vls = [fit_predict(model, hp, sp, seq, norm, {}, [sd])[1] for sd in TUNE_SEEDS]
            rows.append({**hp, "val_ce_seeds": vls, "val_ce": float(np.mean(vls))})
            print(model, hp, np.round(vls, 4), f"{time.time() - t:.0f}s", flush=True)
        best = min(rows, key=lambda r: r["val_ce"])
        res[model] = {"best": {k: best[k] for k in ("d_model", "dropout", "lr")}, "grid": rows}
    path.write_text(json.dumps(res, indent=2))


def stage_primary(samples, seq):
    path = OUT / "nn_primary.parquet"
    if path.exists():
        return
    sp, norm, hp = splits(samples), fit_normalizer(samples, seq), hps()
    frames, seed_frames = [], []
    for model in MODELS:
        preds, _, models, tv = fit_predict(model, hp[model], sp, seq, norm, {"test": sp["test"], "val": sp["val"]}, SEEDS)
        frames += [frame(model, sp["test"], preds["test"], norm, "test"), frame(model, sp["val"], preds["val"], norm, "val")]
        for sd, p in zip(SEEDS, per_seed_predictions(models, tv["test"])):
            seed_frames.append(pred_frame(model, sp["test"], p, None, f"test_seed{sd}"))
        print("primary", model, "done", flush=True)
    pd.concat(frames).to_parquet(path)
    pd.concat(seed_frames).to_parquet(OUT / "nn_primary_seeds.parquet")


def cv_loop(samples, seq, r, fn, tag_prefix):
    folds = make_cv_folds(samples, seed=CV_PARTITION_SEEDS[r])
    frames = []
    for f in range(5):
        s = fold_samples(samples, folds, f, seed=CV_PARTITION_SEEDS[r])
        frames += fn(s, f, f"{tag_prefix}r{r}f{f}")
    return pd.concat(frames)


def stage_cv(samples, seq, r):
    path = OUT / f"nn_cv_r{r}.parquet"
    if path.exists():
        return
    hp = hps()

    def fn(s, f, tag):
        sp, norm = splits(s), fit_normalizer(s, seq)
        out = []
        for model in MODELS:
            preds, _, _, _ = fit_predict(model, hp[model], sp, seq, norm, {"test": sp["test"]}, CV_SEEDS)
            out.append(frame(model, sp["test"], preds["test"], norm, tag))
            print("cv", r, f, model, flush=True)
        return out
    cv_loop(samples, seq, r, fn, "cv").to_parquet(path)


def stage_ablation(samples, seq, group):
    path = OUT / f"nn_ablation_{group}.parquet"
    if path.exists():
        return
    hp = hps()

    def fn(s, f, tag):
        out = []
        for name in ABLATION_GROUPS[group]:
            model, vkw, ckw, static = ABLATIONS[name]
            norm = fit_normalizer(s, seq, static_cols=static or tuple(STATIC_FEATURES))
            sp = splits(s)
            preds, _, _, _ = fit_predict(model, hp[model], sp, seq, norm, {"test": sp["test"]}, CV_SEEDS, vkw, ckw)
            out.append(frame(f"{model}[{name}]", sp["test"], preds["test"], norm, tag))
            print("ablation", group, f, name, flush=True)
        if group == "A":
            # 测试期鲁棒性：完整 CTMT 在测试时拿不到 MRI
            norm = fit_normalizer(s, seq)
            sp = splits(s)
            tests = {"test": sp["test"]}
            views = {k: build_view(sp[k], seq, norm, "sequence") for k in ["train", "val"]}
            tv = {"full": build_view(sp["test"], seq, norm, "sequence"),
                  "noMRI": build_view(sp["test"], seq, norm, "sequence", drop=GROUPS["MRI"])}
            preds, _, _ = run_neural("CTMT", make_cfg("CTMT", hp["CTMT"]), views["train"], views["val"], tv, CV_SEEDS, norm)
            out.append(frame("CTMT[refit-full]", sp["test"], preds["full"], norm, tag))
            out.append(frame("CTMT[test-time-no-MRI]", sp["test"], preds["noMRI"], norm, tag))
        return out
    cv_loop(samples, seq, 0, fn, "cv").to_parquet(path)


def stage_curated(samples, seq, r=0):
    """
    LongForMAD 式筛选：训练/验证集剔除曾出现 MCI→CN 的参与者和 CN→痴呆预测点，测试集保持完整。
    第 0 次重复：四个模型 + 完整队列训练的 CTMT 仅分类头，外加随机剔除对照（CTMT，剔除点数与筛选相同，R2-M3）；
    第 1、2 次重复（返修加做，R2-M3）：只做 CTMT 与 CTMT 仅分类头，用 15 折检验回归头的作用。
    """
    path = OUT / ("nn_curated.parquet" if r == 0 else f"nn_curated_r{r}.parquet")
    if path.exists():
        return
    hp = hps()
    cur = pd.Series(curated_mask(samples, reversion_flags(ROOT)).to_numpy(), index=samples["sample_id"])
    models = ([("CTMT", "CTMT", {}), ("CTMT-classification-only", "CTMT", {"reg_weight": 0.0}),
               ("LongForMAD", "LongForMAD", {}), ("GRU", "GRU", {})] if r == 0 else
              [("CTMT", "CTMT", {}), ("CTMT-classification-only", "CTMT", {"reg_weight": 0.0})])

    def train_on(sub, sp_test, name, model, ckw, tag):
        spc = splits(sub)
        norm = fit_normalizer(sub, seq)
        views = {k: build_view(spc[k], seq, norm, KIND[model]) for k in ["train", "val"]}
        tv = {"test": build_view(sp_test, seq, norm, KIND[model])}
        preds, _, _ = run_neural(model, make_cfg(model, hp[model], **ckw), views["train"], views["val"], tv, CV_SEEDS, norm)
        return frame(name, sp_test, preds["test"], norm, tag)

    def fn(s, f, tag):
        keep = s["sample_id"].map(cur).to_numpy() | (s["split"] == "test").to_numpy()
        sp = splits(s)
        out = []
        for name, model, ckw in models:
            out.append(train_on(s[keep].reset_index(drop=True), sp["test"], f"{name}[curated-train]", model, ckw, tag))
            print("curated", r, f, name, flush=True)
        # 对照：完整队列训练的 CTMT 仅分类头（检验「回归头保留了逆转识别能力」的解释）
        out.append(train_on(s, sp["test"], "CTMT-classification-only", "CTMT", {"reg_weight": 0.0}, tag))
        if r == 0:
            dev = (s["split"] != "test").to_numpy()
            n_drop = int((~keep).sum())
            subj = s.loc[dev, "subject_id"].drop_duplicates().to_numpy()
            counts = s.loc[dev].groupby("subject_id").size()
            order = np.random.default_rng(1000).permutation(subj)
            cum = counts.loc[order].cumsum().to_numpy()
            gone = set(order[:int(np.searchsorted(cum, n_drop)) + 1])
            sub = s[~s["subject_id"].isin(gone)].reset_index(drop=True)
            out.append(train_on(sub, sp["test"], "CTMT[random-removal-0]", "CTMT", {}, tag))
            print("curated", r, f, "random-removal", flush=True)
        return out
    cv_loop(samples, seq, r, fn, "cv").to_parquet(path)


def stage_sensitivity(variant):
    path = OUT / f"nn_sens_{variant}.parquet"
    if path.exists():
        return
    samples, seq = load_dataset(ROOT, variant)
    hp = hps()

    def fn(s, f, tag):
        sp, norm = splits(s), fit_normalizer(s, seq)
        out = []
        for model in ["CTMT", "GRU"]:
            preds, _, _, _ = fit_predict(model, hp[model], sp, seq, norm, {"test": sp["test"]}, CV_SEEDS)
            out.append(frame(f"{model}[{variant}]", sp["test"], preds["test"], norm, tag))
            print("sens", variant, f, model, flush=True)
        return out
    cv_loop(samples, seq, 0, fn, "cv").to_parquet(path)


def stage_temporal(samples, seq):
    path = OUT / "nn_temporal.parquet"
    if path.exists():
        return
    hp = hps()
    s = temporal_split(samples)
    sp, norm = splits(s), fit_normalizer(s, seq)
    out = []
    for model in MODELS:
        preds, _, _, _ = fit_predict(model, hp[model], sp, seq, norm, {"test": sp["test"]}, SEEDS)
        out.append(frame(model, sp["test"], preds["test"], norm, "temporal"))
        print("temporal", model, flush=True)
    pd.concat(out).to_parquet(path)


def main():
    stage, args = sys.argv[1], sys.argv[2:]
    if stage == "sensitivity":
        return stage_sensitivity(args[0])
    samples, seq = load_dataset(ROOT)
    if stage == "tune":
        stage_tune(samples, seq)
    elif stage == "primary":
        stage_primary(samples, seq)
    elif stage == "cv":
        stage_cv(samples, seq, int(args[0]))
    elif stage == "ablation":
        stage_ablation(samples, seq, args[0])
    elif stage == "curated":
        stage_curated(samples, seq, int(args[0]) if args else 0)
    elif stage == "temporal":
        stage_temporal(samples, seq)


if __name__ == "__main__":
    main()
