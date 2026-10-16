"""
阶段 5：汇总全部预测 → 论文表格（results/tables/*.csv），最后调用 make_figures.py 生成配图。

用法：.venv/bin/python scripts/make_report.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import report as R  # noqa: E402
from src.modeling.data import load_dataset  # noqa: E402

TAB = ROOT / "results/tables"
TAB.mkdir(parents=True, exist_ok=True)

MAIN = ["GBM-snapshot", "GBM-aggregate", "LongForMAD", "GRU", "CTMT"]


def main():
    samples, seq = load_dataset(ROOT)
    preds = R.load_predictions(ROOT)

    R.cohort_table(samples, seq).to_csv(TAB / "table1_cohort.csv")

    main_t, main_d, points = R.classification_table(preds, samples, MAIN, "test", ref="CTMT")
    main_t.to_csv(TAB / "table2_main_test.csv", index=False)
    main_d.to_csv(TAB / "table2_pairwise_vs_CTMT.csv", index=False)

    cv_t, cv_d, cv_points = R.classification_table(preds, samples, MAIN, "cv", ref="CTMT")
    cv_t.to_csv(TAB / "table3_cv.csv", index=False)
    cv_d.to_csv(TAB / "table3_pairwise_vs_CTMT.csv", index=False)

    strata = R.history_strata(preds, samples, MAIN, "cv")
    strata.to_csv(TAB / "table_history_strata_cv.csv", index=False)

    abl_models = ["CTMT"] + sorted(m for m in preds["model"].unique() if m.startswith("CTMT[") and "curated" not in m)
    abl_t, abl_d, _ = R.classification_table(preds, samples, abl_models, "test", ref="CTMT")
    abl_t.to_csv(TAB / "table4_ablation.csv", index=False)
    abl_d.to_csv(TAB / "table4_ablation_diff.csv", index=False)

    cur = R.curation_tables(ROOT, preds, samples, ["CTMT", "LongForMAD", "GBM-aggregate"],
                            ["CTMT[curated-train]", "LongForMAD[curated-train]", "GBM-aggregate[curated-train]"])
    cur.to_csv(TAB / "table5_curation.csv", index=False)
    R.curation_differences(ROOT, preds, samples, ["CTMT", "LongForMAD", "GBM-aggregate"],
                           ["CTMT[curated-train]", "LongForMAD[curated-train]", "GBM-aggregate[curated-train]"]
                           ).to_csv(TAB / "table5_curation_diff.csv", index=False)

    reg_models = ["GBM-snapshot", "GBM-aggregate", "GRU", "CTMT"]
    R.regression_table(preds, samples, reg_models, "test").to_csv(TAB / "table6_regression_test.csv", index=False)
    R.regression_table(preds, samples, reg_models, "cv").to_csv(TAB / "table6_regression_cv.csv", index=False)
    R.score_threshold_analysis(preds, samples).to_csv(TAB / "table7_score_threshold.csv", index=False)

    # 配图统一由 make_figures.py 生成（尺寸按投稿模板版心设定）
    import make_figures
    make_figures.main()
    print("tables:", sorted(p.name for p in TAB.glob("*.csv")))


if __name__ == "__main__":
    main()
