"""
按第 1 轮审稿意见重建数据集（dataset_v2）：主队列 + 三个敏感性分析队列变体。
用法：.venv/bin/python scripts/build_dataset_v2.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset.cohort_v2 import build_dataset_v2  # noqa: E402

OUT = ROOT / "data/processed/dataset_v2"
for variant in ["main", "all_cause", "wide", "absorbing"]:
    samples, seq, cand = build_dataset_v2(ROOT, variant)
    d = OUT if variant == "main" else OUT / "variants" / variant
    d.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(d / "samples.parquet")
    seq.to_parquet(d / "sequences.parquet")
    cand.to_parquet(d / "candidates.parquet")
    print(variant, "samples", len(samples), "participants", samples.subject_id.nunique(), "steps", len(seq),
          "| candidates by status:", cand.status.value_counts().to_dict())
