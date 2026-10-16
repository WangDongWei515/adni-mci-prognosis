"""
淀粉样蛋白亚组（返修补充，回应 Alzheimer's Research & Therapy 对 MCI 需有淀粉样蛋白状态的要求）。
每个预测点取指数日当天或之前最近一次质控通过（qc_flag 1/2）的 UC Berkeley 淀粉样蛋白 PET 状态，不用指数日之后的信息；
不重新训练模型，只在第一次交叉验证重复的预测上按淀粉样蛋白状态分层评估区分度。
用法：.venv/bin/python scripts/amyloid_subgroups.py
产出：results/tables/s_amyloid.csv、results/tables/s_amyloid_counts.json
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pyreadr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import report_v2 as R  # noqa: E402
from src.modeling.data import load_dataset  # noqa: E402

TAB = ROOT / "results/tables"
MODELS = ["GBM-snapshot", "GBM-aggregate", "LongForMAD", "GRU", "CTMT"]


def amyloid_at_index(samples):
    amy = list(pyreadr.read_r(str(ROOT / "data/raw/adni/ADNIMERGE2/data/UCBERKELEY_AMY_6MM.rda")).values())[0]
    amy = amy[amy["qc_flag"].isin([1, 2]) & amy["AMYLOID_STATUS"].notna()].copy()
    amy["subject_id"] = amy["RID"].astype("int64")
    amy["scan_date"] = pd.to_datetime(amy["SCANDATE"])
    m = samples[["sample_id", "subject_id", "index_date"]].assign(index_date=lambda d: pd.to_datetime(d["index_date"]))
    m = m.merge(amy[["subject_id", "scan_date", "AMYLOID_STATUS", "CENTILOIDS", "TRACER"]], on="subject_id")
    m = m[m["scan_date"] <= m["index_date"]].sort_values("scan_date").groupby("sample_id").tail(1)
    m["lag_years"] = (m["index_date"] - m["scan_date"]).dt.days / 365.25
    return m.set_index("sample_id")


def main():
    samples, _ = load_dataset(ROOT)
    preds = R.load_predictions(ROOT)
    amy = amyloid_at_index(samples)
    samples["amyloid"] = samples["sample_id"].map(amy["AMYLOID_STATUS"])
    groups = {
        "amyloid known": lambda d: d.amyloid.notna(),
        "amyloid positive": lambda d: d.amyloid == 1,
        "amyloid negative": lambda d: d.amyloid == 0,
    }
    tab = pd.concat([R.subset_table(preds, samples, MODELS, fn, name) for name, fn in groups.items()])
    tab.to_csv(TAB / "s_amyloid.csv", index=False)

    counts = {}
    for dx in ["MCI", "CN"]:
        s = samples[samples.origin_diagnosis == dx]
        known = s[s.amyloid.notna()]
        counts[dx] = {
            "points": int(len(s)), "with_status": int(len(known)),
            "positive": int((known.amyloid == 1).sum()),
            "median_lag_years": round(float(amy.loc[known.sample_id, "lag_years"].median()), 2),
            "outcome_by_status": {str(int(k)): v.value_counts().to_dict() for k, v in known.groupby("amyloid")["target_diagnosis"]},
        }
    (TAB / "s_amyloid_counts.json").write_text(json.dumps(counts, indent=2))
    # Table 1 用：按数据划分与指数诊断统计淀粉样蛋白阳性比例（预测点层面）
    rows = {}
    for dx in ["CN", "MCI"]:
        cells = {}
        for col, sel in [("Overall", samples.split.notna()), ("Training", samples.split == "train"),
                         ("Validation", samples.split == "val"), ("Test", samples.split == "test")]:
            s = samples[sel & (samples.origin_diagnosis == dx) & samples.amyloid.notna()]
            cells[col] = f"{int((s.amyloid == 1).sum())}/{len(s)} ({100 * (s.amyloid == 1).mean():.1f})"
        rows[f"  {dx} index: amyloid PET positive, n/N (%)"] = cells
    pd.DataFrame(rows).T.to_csv(TAB / "t1_amyloid_rows.csv")
    print(json.dumps(counts, indent=2))
    print(tab[tab.model.isin(["GBM-aggregate", "CTMT"])].round(3).to_string())


if __name__ == "__main__":
    main()
