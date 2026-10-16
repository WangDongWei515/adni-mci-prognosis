"""
第 2 版：汇总全部预测 → results/tables/*.csv（论文与回复信用）。
用法：.venv/bin/python scripts/make_report_v2.py [部分名 ...]   不给参数则全部生成
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.cohort_v2 import load_diagnoses  # noqa: E402
from src.evaluation import report_v2 as R  # noqa: E402
from src.modeling.data import curated_mask, load_dataset, reversion_flags  # noqa: E402

TAB = ROOT / "results/tables"
TAB.mkdir(parents=True, exist_ok=True)


def save(df, name, index=False):
    df.to_csv(TAB / f"{name}.csv", index=index)
    print("wrote", name, df.shape, flush=True)


def main(parts):
    samples, seq = load_dataset(ROOT)
    preds = R.load_predictions(ROOT)
    cog = pd.read_parquet(ROOT / "data/processed/adni_v0/cognition_long.parquet")
    cog["subject_id"] = cog["subject_id"].astype("int64")
    cog["date"] = pd.to_datetime(cog["date"])
    want = lambda k: not parts or k in parts  # noqa: E731

    if want("cohort"):
        cand = pd.read_parquet(ROOT / "data/processed/dataset_v2/candidates.parquet")
        save(R.cohort_table(samples, seq, cog), "t1_cohort", index=True)
        save(R.included_vs_excluded(cand, cog), "s_included_vs_excluded", index=True)
        dx_all = pd.read_parquet(ROOT / "data/processed/adni_v0/diagnoses_audit.parquet")
        n_all = dx_all.loc[dx_all["date_eligible"] & dx_all["diagnosis"].notna(), "subject_id"].nunique()
        (TAB / "flow_counts.json").write_text(json.dumps(R.flow_counts(cand, samples, int(n_all)), indent=2))
        rev = reversion_flags(ROOT)
        cur = curated_mask(samples, rev)
        tab, n_subj = R.curation_breakdown(samples, cur)
        save(tab, "s_curation_removed_by_transition", index=True)
        (TAB / "curation_counts.json").write_text(json.dumps({
            "removed_points": int((~cur).sum()), "removed_participants": int(n_subj),
            "participants_mci_to_cn_any": int(rev["mci_to_cn"].sum()),
            "participants_dem_to_lower_any": int(rev["dem_to_lower"].sum())}, indent=2))

    if want("main"):
        save(R.discrimination_table(preds, samples), "t2_discrimination")
        fm = R.fold_metrics(preds, samples, R.MODELS)
        save(fm, "s_fold_metrics")
        save(R.primary_comparisons(fm), "t2_primary_comparisons")
        save(R.all_pairwise(fm, ["GBM-snapshot", "GBM-aggregate", "LongForMAD", "GRU", "CTMT"]), "s_all_pairwise")
        save(fm.groupby(["model", "repeat"])[["mci_conv", "mci_rev", "cn_prog"]].mean().reset_index(), "s_per_repeat")
        save(R.seed_variability(preds, samples).reset_index(), "s_seed_variability")
        save(R.calibration_table(preds, samples), "s_calibration")

    if want("history"):
        st, mix = R.history_strata(preds, samples, seq)
        save(st, "f3_history_strata")
        save(mix, "s_history_strata_casemix")

    if want("ablation"):
        save(R.ablation_table(preds, samples), "f4_ablation")

    if want("curation"):
        cur = curated_mask(samples, reversion_flags(ROOT))
        ev, null, tr = R.curation_analysis(preds, samples, cur)
        save(ev, "t3_curation_eval")
        save(null, "t3_curation_null")
        save(tr, "t3_curation_train")
        # 检验「回归头保留了逆转识别能力」：仅分类头的 CTMT，完整 vs 筛选训练
        ev2, _, tr2 = R.curation_analysis(preds, samples, cur, models=("CTMT-classification-only",), n_null=1)
        save(tr2, "s_curation_headB_test")

    if want("sensitivity"):
        rows = []
        for variant in ["all_cause", "wide", "absorbing"]:
            vs, _ = load_dataset(ROOT, variant)
            ms = {f"[{variant}]GBM-snapshot": "GBM-snapshot", f"[{variant}]GBM-aggregate": "GBM-aggregate",
                  f"GRU[{variant}]": "GRU", f"CTMT[{variant}]": "CTMT"}
            t = R.subset_table(preds, vs, list(ms), None, variant)
            t["model"] = t["model"].map(ms)
            t["n_points_cohort"] = len(vs)
            rows.append(t)
        rows.append(R.subset_table(preds, samples, ["GBM-snapshot", "GBM-aggregate", "GRU", "CTMT"], None, "main"))
        save(pd.concat(rows), "s_sensitivity_cohorts")

    if want("temporal"):
        rows = []
        for m in R.MODELS:
            d = R.join(preds[(preds.model == m) & (preds.eval_set == "temporal")], samples)
            point, boot = R.boot_metrics(d, R.MAIN_KEYS)
            rows.append({"model": m, "n_points": len(d), "n_participants": d.subject_id.nunique(),
                         **{k: R.fmt(point[k], boot[k].quantile(.025), boot[k].quantile(.975)) for k in R.MAIN_KEYS},
                         **{k + "_est": point[k] for k in R.MAIN_KEYS}})
        save(pd.DataFrame(rows), "s_temporal")

    if want("subsets"):
        first_idx = samples.sort_values("index_date").groupby("subject_id").head(1)["sample_id"]
        models = ["GBM-snapshot", "GBM-aggregate", "LongForMAD", "GRU", "CTMT"]
        a = R.subset_table(preds, samples, models, lambda d: d["sample_id"].isin(first_idx), "first point per participant")
        n_i = samples.groupby("subject_id").size()
        b = R.subset_table(preds, samples, models, None, "inverse-count weighted",
                           weights=lambda d: 1.0 / d["subject_id"].map(n_i).to_numpy())
        dx = load_diagnoses(ROOT)
        conf = R.confirmed_outcome(samples, dx)
        c_samples = conf.copy()
        c_samples.loc[c_samples["unconfirmed_progression"], "y"] = 0
        c = R.subset_table(preds, c_samples, models, None, "confirmed CN progression (unconfirmed = no event)")
        save(pd.concat([a, b, c]), "s_participant_weighting_and_confirmation")
        (TAB / "confirmation_counts.json").write_text(json.dumps({
            "cn_progression_events": int(((conf.origin_diagnosis == "CN") & (conf.target_diagnosis != "CN")).sum()),
            "unconfirmed_next_visit_cn": int(conf["unconfirmed_progression"].sum()),
            "no_next_diagnosis": int(((conf.origin_diagnosis == "CN") & (conf.target_diagnosis != "CN") & conf.next_dx.isna()).sum()),
            "mci_reversions": int(((conf.origin_diagnosis == "MCI") & (conf.target_diagnosis == "CN")).sum()),
            "reversions_mci_again_next": int(((conf.origin_diagnosis == "MCI") & (conf.target_diagnosis == "CN") & conf.next_dx.isin(["MCI", "Dementia"])).sum()),
            "reversions_no_next": int(((conf.origin_diagnosis == "MCI") & (conf.target_diagnosis == "CN") & conf.next_dx.isna()).sum()),
        }, indent=2))
        subgroups = {
            "female": lambda d: d.sex_male == 0, "male": lambda d: d.sex_male == 1,
            "age <75": lambda d: d.approx_age_years < 75, "age ≥75": lambda d: d.approx_age_years >= 75,
            "education ≤15 y": lambda d: d.education_years <= 15, "education ≥16 y": lambda d: d.education_years >= 16,
            "APOE ε4 non-carrier": lambda d: d.apoe4 == 0, "APOE ε4 carrier": lambda d: d.apoe4 > 0,
            "ADNI1/GO/2 index visit": lambda d: d.protocol.isin(["ADNI1", "ADNIGO", "ADNI2"]),
            "ADNI3/4 index visit": lambda d: d.protocol.isin(["ADNI3", "ADNI4"]),
        }
        save(pd.concat([R.subset_table(preds, samples, ["GBM-aggregate", "CTMT"], fn, name) for name, fn in subgroups.items()]),
             "s_subgroups")

    if want("regression"):
        models = ["GBM-aggregate", "GBM-snapshot", "GRU", "CTMT", "Drift-LOCF", "LOCF"]
        save(R.regression_table(preds, samples, models, ref="LOCF"), "s_regression_vs_locf")
        save(R.regression_table(preds, samples, models, ref="Drift-LOCF"), "s_regression_vs_drift")
        save(R.regression_by_transition(preds, samples, models), "s_regression_by_transition")
        save(R.substitution_analysis(preds, samples), "s_substitution")

    if want("operating"):
        save(R.operating_points(preds, samples), "s_operating_points")

    if want("extras"):
        conf = R.confirmed_outcome(samples, load_diagnoses(ROOT))
        save(R.reverter_profile(samples, seq, cog, conf.set_index("sample_id")["next_dx"]), "s_reverter_profile", index=True)
        save(R.reversion_vs_stable(preds, samples), "s_reversion_vs_stable")
        save(R.calibration_ci(preds, samples), "s_calibration_ci")
        save(R.total_optimism_ci(preds, samples, curated_mask(samples, reversion_flags(ROOT))), "s_total_optimism")


if __name__ == "__main__":
    main(sys.argv[1:])
