"""
把正文引用的关键数字从 results/tables 汇总打印出来，供改稿时逐条核对（只读）。
用法：.venv/bin/python scripts/numbers_digest.py > results/tables/numbers_digest.txt
"""
import json
from pathlib import Path

import pandas as pd

TAB = Path(__file__).resolve().parents[1] / "results/tables"
L = ["GBM-snapshot", "GBM-aggregate", "LongForMAD", "GRU", "CTMT"]
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 400)


def r(df):
    return df.round(3).to_string(index=False)


def rng(s):
    return f"{s.min():.3f}–{s.max():.3f}"


def sec(title):
    print(f"\n==== {title}")


t2 = pd.read_csv(TAB / "t2_discrimination.csv")
for setting in t2.setting.unique():
    sec(f"t2 {setting}")
    g = t2[t2.setting == setting]
    print(r(g[["model"] + [c for c in g.columns if c.endswith("_est")]]))
    gl = g[g.model.isin(L)]
    print("learned ranges:", {k: rng(gl[k + "_est"]) for k in ["mci_conv", "mci_rev", "cn_prog", "macro_auroc", "bal_acc", "brier"]})
for name in ["t2_primary_comparisons", "s_all_pairwise", "s_per_repeat", "s_seed_variability", "s_calibration_ci",
             "s_operating_points", "f3_history_strata", "s_history_strata_casemix", "f4_ablation", "t3_curation_eval",
             "t3_curation_null", "t3_curation_train", "s_curation_headB_test", "s_total_optimism",
             "s_sensitivity_cohorts", "s_participant_weighting_and_confirmation", "s_subgroups", "s_temporal",
             "s_regression_vs_locf", "s_regression_vs_drift", "s_regression_by_transition", "s_substitution",
             "s_reversion_vs_stable", "s_reverter_profile", "s_amyloid",
             "r_participant_level", "r_participant_level_tests", "r_reversion_scores", "r_headB_15folds",
             "r_curation_decomposition", "r_amyloid_auc", "r_amyloid_paired", "r_amyloid_calibration",
             "r_equivalence_robustness", "r_temporal_paired", "r_subgroup_differences"]:
    p = TAB / f"{name}.csv"
    if p.exists():
        sec(name)
        df = pd.read_csv(p)
        if name == "f4_ablation":
            df = df[["variant", "task", "auc", "diff", "t_lo", "t_hi", "t_p"]]
        if name == "s_calibration_ci":
            df = df.drop(columns=[c for c in df.columns if c in ("n",)])
        print(r(df))
for name in ["flow_counts", "curation_counts", "confirmation_counts", "s_amyloid_counts", "r_summary"]:
    p = TAB / f"{name}.json"
    if p.exists():
        sec(name)
        print(json.dumps(json.loads(p.read_text()), indent=1, default=str)[:6000])
