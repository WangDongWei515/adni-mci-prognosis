"""
阶段 3–5 的神经网络实验：超参数选择、主比较、消融、队列筛选实验、5 折交叉验证。

用法：.venv/bin/python scripts/run_neural.py [stage ...]
  stage ∈ tune main ablation curated cv（默认全部）；已存在的产出文件会跳过（断点续跑）
产出：results/tuning.json、results/predictions/nn_*.parquet
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

from src.modeling.cv import fold_samples  # noqa: E402
from src.modeling.data import curated_mask, fit_normalizer, load_dataset, reversion_flags  # noqa: E402
from src.modeling.pipeline import COG_IDX, MRI_IDX, build_view, pred_frame, reg_to_raw, run_neural  # noqa: E402
from src.modeling.train import TrainConfig  # noqa: E402

torch.set_num_threads(4)
OUT = ROOT / "results/predictions"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
CV_SEEDS = [0, 1, 2]
# 三个神经网络模型用同样大小的超参数网格，避免「主模型调得更细」带来的不公平
GRID = [dict(d_model=d, dropout=p, lr=lr) for d in (32, 64) for p in (0.1, 0.3) for lr in (1e-3, 3e-4)]
KIND = {"CTMT": "sequence", "GRU": "sequence", "LongForMAD": "annual"}


def splits(samples):
    return {k: samples[samples["split"] == k].reset_index(drop=True) for k in ["train", "val", "test"]}


def make_cfg(model, hp, **over):
    kw = {"d_model": hp["d_model"], "dropout": hp["dropout"]}
    cfg = TrainConfig(lr=hp["lr"], model_kwargs=kw, reg_weight=0.0 if model == "LongForMAD" else 0.5)
    for k, v in over.items():
        if k in ("use_time",):
            cfg.model_kwargs[k] = v
        else:
            setattr(cfg, k, v)
    return cfg


def frames_from(model_name, sp, preds, norm, tag="test", key="test"):
    p, r = preds[key]
    return pred_frame(model_name, sp[key if key in sp else "test"], p,
                      None if r is None else reg_to_raw(r, norm), tag)


def stage_tune(samples, seq):
    path = ROOT / "results/tuning.json"
    if path.exists():
        return json.loads(path.read_text())
    sp, norm = splits(samples), fit_normalizer(samples, seq)
    res = {}
    for model in ["CTMT", "LongForMAD", "GRU"]:
        views = {k: build_view(v, seq, norm, KIND[model]) for k, v in sp.items() if k != "test"}
        rows = []
        for hp in GRID:
            t = time.time()
            _, vl, _ = run_neural(model, make_cfg(model, hp), views["train"], views["val"], {}, seeds=[0])
            rows.append({**hp, "val_ce": vl})
            print(model, hp, round(vl, 4), f"{time.time() - t:.0f}s", flush=True)
        best = min(rows, key=lambda r: r["val_ce"])
        res[model] = {"best": {k: best[k] for k in ("d_model", "dropout", "lr")}, "grid": rows}
    path.write_text(json.dumps(res, indent=2))
    return res


def stage_main(samples, seq, hps):
    path = OUT / "nn_primary.parquet"
    if path.exists():
        return
    sp, norm = splits(samples), fit_normalizer(samples, seq)
    frames = []
    for model in ["CTMT", "LongForMAD", "GRU"]:
        views = {k: build_view(v, seq, norm, KIND[model]) for k, v in sp.items()}
        preds, _, _ = run_neural(model, make_cfg(model, hps[model]["best"]), views["train"], views["val"],
                                 {"test": views["test"]}, SEEDS)
        frames.append(frames_from(model, sp, preds, norm))
        print("main", model, "done", flush=True)
    pd.concat(frames).to_parquet(path)


def stage_ablation(samples, seq, hps):
    path = OUT / "nn_ablation.parquet"
    if path.exists():
        return
    sp, norm = splits(samples), fit_normalizer(samples, seq)
    hp = hps["CTMT"]["best"]
    frames = []
    # 训练期消融：每个变体都重新训练，训练与测试使用同一种输入设定
    variants = {
        "no-time-encoding": (dict(), dict(use_time=False)),
        "window-1y": (dict(window=1.0), dict()),
        "window-3y": (dict(window=3.0), dict()),
        "cognition-only": (dict(drop=MRI_IDX), dict()),
        "MRI-only": (dict(drop=COG_IDX), dict()),
        "no-step-dropout": (dict(), dict(step_dropout=0.0)),
        "classification-only": (dict(), dict(reg_weight=0.0)),
    }
    for name, (vkw, ckw) in variants.items():
        views = {k: build_view(v, seq, norm, "sequence", **vkw) for k, v in sp.items()}
        preds, _, _ = run_neural("CTMT", make_cfg("CTMT", hp, **ckw), views["train"], views["val"],
                                 {"test": views["test"]}, SEEDS)
        frames.append(frames_from(f"CTMT[{name}]", sp, preds, norm))
        print("ablation", name, "done", flush=True)
    # 测试期鲁棒性：用完整模型，只在测试时拿掉部分输入（模拟真实场景中检查没做）
    views = {k: build_view(v, seq, norm, "sequence") for k, v in sp.items()}
    tests = {
        "test-time-no-MRI": build_view(sp["test"], seq, norm, "sequence", drop=MRI_IDX),
        "test-time-last-step-only": build_view(sp["test"], seq, norm, "sequence", keep_last_only=True),
    }
    preds, _, _ = run_neural("CTMT", make_cfg("CTMT", hp), views["train"], views["val"], tests, SEEDS)
    for name in tests:
        p, r = preds[name]
        frames.append(pred_frame(f"CTMT[{name}]", sp["test"], p, reg_to_raw(r, norm)))
    pd.concat(frames).to_parquet(path)


def stage_curated(samples, seq, hps):
    """在 LongForMAD 式筛选后的队列上训练，在完整测试集上评估（筛选对真实世界表现的影响）。"""
    path = OUT / "nn_curated.parquet"
    if path.exists():
        return
    cur = curated_mask(samples, reversion_flags(ROOT)).to_numpy()
    sc = samples[cur | (samples["split"] == "test").to_numpy()].reset_index(drop=True)
    sp_c = splits(sc)
    sp = splits(samples)
    norm = fit_normalizer(sc, seq)
    frames = []
    for model in ["CTMT", "LongForMAD"]:
        v_tr = build_view(sp_c["train"], seq, norm, KIND[model])
        v_va = build_view(sp_c["val"], seq, norm, KIND[model])
        v_te = build_view(sp["test"], seq, norm, KIND[model])
        preds, _, _ = run_neural(model, make_cfg(model, hps[model]["best"]), v_tr, v_va, {"test": v_te}, SEEDS)
        frames.append(frames_from(f"{model}[curated-train]", sp, preds, norm))
        print("curated", model, "done", flush=True)
    pd.concat(frames).to_parquet(path)


def stage_cv(samples, seq, hps):
    path = OUT / "nn_cv.parquet"
    if path.exists():
        return
    folds = pd.read_parquet(ROOT / "results/cv_folds.parquet")  # 由 run_gbm.py 生成，保证两边划分一致
    frames = []
    for f in range(5):
        s = fold_samples(samples, folds, f)
        sp, norm = splits(s), fit_normalizer(s, seq)
        for model in ["CTMT", "LongForMAD", "GRU"]:
            views = {k: build_view(v, seq, norm, KIND[model]) for k, v in sp.items()}
            preds, _, _ = run_neural(model, make_cfg(model, hps[model]["best"]), views["train"], views["val"],
                                     {"test": views["test"]}, CV_SEEDS)
            frames.append(frames_from(model, sp, preds, norm, tag=f"cv{f}"))
            print("cv", f, model, "done", flush=True)
    pd.concat(frames).to_parquet(path)


ABLATIONS = {
    "no-time-encoding": (dict(), dict(use_time=False)),
    "window-1y": (dict(window=1.0), dict()),
    "window-3y": (dict(window=3.0), dict()),
    "cognition-only": (dict(drop=MRI_IDX), dict()),
    "MRI-only": (dict(drop=COG_IDX), dict()),
    "no-step-dropout": (dict(), dict(step_dropout=0.0)),
    "classification-only": (dict(), dict(reg_weight=0.0)),
}


def stage_cv_ablation(samples, seq, hps):
    """消融的 5 折交叉验证版本：每个变体在每一折都重新训练，用全部 4393 条样本评估，比只用测试集更有统计效力。"""
    path = OUT / "nn_cv_ablation.parquet"
    if path.exists():
        return
    folds = pd.read_parquet(ROOT / "results/cv_folds.parquet")
    hp = hps["CTMT"]["best"]
    frames = []
    for f in range(5):
        s = fold_samples(samples, folds, f)
        sp, norm = splits(s), fit_normalizer(s, seq)
        for name, (vkw, ckw) in ABLATIONS.items():
            views = {k: build_view(v, seq, norm, "sequence", **vkw) for k, v in sp.items()}
            preds, _, _ = run_neural("CTMT", make_cfg("CTMT", hp, **ckw), views["train"], views["val"],
                                     {"test": views["test"]}, CV_SEEDS)
            frames.append(frames_from(f"CTMT[{name}]", sp, preds, norm, tag=f"cv{f}"))
            print("cv_ablation", f, name, "done", flush=True)
        views = {k: build_view(v, seq, norm, "sequence") for k, v in sp.items()}
        tests = {"test-time-no-MRI": build_view(sp["test"], seq, norm, "sequence", drop=MRI_IDX),
                 "test-time-last-step-only": build_view(sp["test"], seq, norm, "sequence", keep_last_only=True)}
        preds, _, _ = run_neural("CTMT", make_cfg("CTMT", hp), views["train"], views["val"], tests, CV_SEEDS)
        for name in tests:
            p, r = preds[name]
            frames.append(pred_frame(f"CTMT[{name}]", sp["test"], p, reg_to_raw(r, norm), tag=f"cv{f}"))
        print("cv_ablation", f, "test-time done", flush=True)
        pd.concat(frames).to_parquet(OUT / "nn_cv_ablation.partial.parquet")
    pd.concat(frames).to_parquet(path)
    (OUT / "nn_cv_ablation.partial.parquet").unlink(missing_ok=True)


def stage_cv_curated(samples, seq, hps):
    """队列筛选实验的 5 折交叉验证版本：每折在筛选后的训练/验证患者上训练，在该折完整的测试集上评估。"""
    path = OUT / "nn_cv_curated.parquet"
    if path.exists():
        return
    folds = pd.read_parquet(ROOT / "results/cv_folds.parquet")
    cur = pd.Series(curated_mask(samples, reversion_flags(ROOT)).to_numpy(), index=samples["sample_id"])
    frames = []
    for f in range(5):
        s = fold_samples(samples, folds, f)
        keep = s["sample_id"].map(cur).to_numpy() | (s["split"] == "test").to_numpy()
        sc = s[keep].reset_index(drop=True)
        sp_c, sp = splits(sc), splits(s)
        norm = fit_normalizer(sc, seq)
        for model in ["CTMT", "LongForMAD"]:
            v_tr = build_view(sp_c["train"], seq, norm, KIND[model])
            v_va = build_view(sp_c["val"], seq, norm, KIND[model])
            v_te = build_view(sp["test"], seq, norm, KIND[model])
            preds, _, _ = run_neural(model, make_cfg(model, hps[model]["best"]), v_tr, v_va, {"test": v_te}, CV_SEEDS)
            frames.append(frames_from(f"{model}[curated-train]", sp, preds, norm, tag=f"cv{f}"))
            print("cv_curated", f, model, "done", flush=True)
    pd.concat(frames).to_parquet(path)


def main():
    stages = sys.argv[1:] or ["tune", "main", "ablation", "curated", "cv"]
    samples, seq = load_dataset(ROOT)
    hps = stage_tune(samples, seq)
    print("best:", {k: v["best"] for k, v in hps.items()}, flush=True)
    for st in stages:
        if st != "tune":
            globals()[f"stage_{st}"](samples, seq, hps)


if __name__ == "__main__":
    main()
