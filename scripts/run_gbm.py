"""
阶段 3：表格基线（起点快照 GBM / 历史聚合 GBM）+ 朴素回归基线「分数不变」。
与神经网络实验分进程运行（lightgbm 与 torch 的 OpenMP 冲突，见 src/modeling/pipeline.py）。

用法：.venv/bin/python scripts/run_gbm.py
产出：results/predictions/gbm_*.parquet
"""
import lightgbm  # noqa: F401  必须先于任何可能间接加载 torch 的模块导入
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.build import SCALES  # noqa: E402
from src.modeling.cv import fold_samples, make_cv_folds  # noqa: E402
from src.modeling.data import curated_mask, load_dataset, reversion_flags  # noqa: E402
from src.modeling.pipeline import locf_regression, pred_frame, run_gbm  # noqa: E402

OUT = ROOT / "results/predictions"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    samples, seq = load_dataset(ROOT)
    tr, va, te = (samples[samples["split"] == s].reset_index(drop=True) for s in ["train", "val", "test"])
    cur = curated_mask(samples, reversion_flags(ROOT))
    samples["curated"] = cur.to_numpy()
    te_cur = samples[(samples["split"] == "test") & samples["curated"]].reset_index(drop=True)

    frames = []
    for mode, name in [("snapshot", "GBM-snapshot"), ("aggregate", "GBM-aggregate")]:
        outs, _ = run_gbm(tr, va, [te], seq, mode)
        p, r = outs[0]
        frames.append(pred_frame(name, te, p, r))
        print(name, "done")

    # 队列筛选实验：在 LongForMAD 式筛选后的训练集上训练，分别在完整测试集与筛选后测试集上评估
    tr_c = samples[(samples["split"] == "train") & samples["curated"]].reset_index(drop=True)
    va_c = samples[(samples["split"] == "val") & samples["curated"]].reset_index(drop=True)
    outs, _ = run_gbm(tr_c, va_c, [te], seq, "aggregate")
    frames.append(pred_frame("GBM-aggregate[curated-train]", te, outs[0][0], outs[0][1]))

    train_mean = tr[[f"future_{s}" for s in SCALES]].mean().to_numpy()
    locf = locf_regression(te, seq, train_mean)
    frames.append(pred_frame("LOCF", te, pd.DataFrame([[1 / 3] * 3] * len(te)).to_numpy(), locf))
    pd.concat(frames).to_parquet(OUT / "gbm_primary.parquet")

    # 5 折交叉验证（稳健性分析）
    folds = make_cv_folds(samples.drop(columns="curated"))
    folds.to_parquet(ROOT / "results/cv_folds.parquet")
    cv_frames = []
    for f in range(5):
        s = fold_samples(samples.drop(columns="curated"), folds, f)
        s_tr, s_va, s_te = (s[s["split"] == k].reset_index(drop=True) for k in ["train", "val", "test"])
        outs, _ = run_gbm(s_tr, s_va, [s_te], seq, "aggregate")
        df = pred_frame("GBM-aggregate", s_te, outs[0][0], outs[0][1], tag=f"cv{f}")
        cv_frames.append(df)
        outs, _ = run_gbm(s_tr, s_va, [s_te], seq, "snapshot")
        cv_frames.append(pred_frame("GBM-snapshot", s_te, outs[0][0], outs[0][1], tag=f"cv{f}"))
        tm = s_tr[[f"future_{x}" for x in SCALES]].mean().to_numpy()
        cv_frames.append(pred_frame("LOCF", s_te, pd.DataFrame([[1 / 3] * 3] * len(s_te)).to_numpy(),
                                    locf_regression(s_te, seq, tm), tag=f"cv{f}"))
        print("cv fold", f, "done")
    pd.concat(cv_frames).to_parquet(OUT / "gbm_cv.parquet")


def cv_curated():
    """队列筛选实验的 5 折交叉验证版本（GBM-aggregate）：用筛选后的训练/验证患者训练，在完整的该折测试集上评估。"""
    samples, seq = load_dataset(ROOT)
    cur = pd.Series(curated_mask(samples, reversion_flags(ROOT)).to_numpy(), index=samples["sample_id"])
    folds = pd.read_parquet(ROOT / "results/cv_folds.parquet")
    frames = []
    for f in range(5):
        s = fold_samples(samples, folds, f)
        keep = s["sample_id"].map(cur).to_numpy()
        s_tr = s[(s["split"] == "train").to_numpy() & keep].reset_index(drop=True)
        s_va = s[(s["split"] == "val").to_numpy() & keep].reset_index(drop=True)
        s_te = s[s["split"] == "test"].reset_index(drop=True)
        outs, _ = run_gbm(s_tr, s_va, [s_te], seq, "aggregate")
        frames.append(pred_frame("GBM-aggregate[curated-train]", s_te, outs[0][0], outs[0][1], tag=f"cv{f}"))
        print("cv_curated fold", f, "done", flush=True)
    pd.concat(frames).to_parquet(OUT / "gbm_cv_curated.parquet")


if __name__ == "__main__":
    if sys.argv[1:] == ["cv_curated"]:
        cv_curated()
    else:
        main()
