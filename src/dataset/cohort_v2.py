"""
数据集第二版（dataset_v2）：按第 1 轮审稿意见重建。与 v1 的差异：

1. 去除重复的预测起点：同一患者的两个起点若对应同一次结局诊断（多为筛查访视 + 基线访视，相隔数周），
   只保留较晚的一个（信息最多）。v1 中有 571 组、1,142 个这样的起点。
2. 输入序列加入既往诊断：每个就诊日若有临床诊断，记为 dx_CN / dx_MCI / dx_DEM（0/1），无诊断则为缺失。
   只使用起点当天及以前的诊断。
3. 记录每个候选起点被排除的原因（随访不足 / 结局窗内无诊断 / 与较晚起点共用结局），用于流程图和
   纳入 vs 排除比较。
4. 支持敏感性分析的队列变体：
   - all_cause：非 AD 病因的痴呆也计为痴呆（v1 删除了这 50 条记录）
   - wide：结局窗放宽到 18–30 个月
   - absorbing：结局窗内无诊断、但结局窗之前已诊断为痴呆且此后没有非痴呆诊断的，结局记为痴呆
5. 补充人口学与临床描述变量：种族、民族、APOE ε4 等位基因数、ADNI 阶段。

时间步仍按自然日定义（一个就诊日 = 有任何认知量表、MRI 或诊断记录的一天）。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

from src.data_engineering.diagnosis_labels import NON_AD_DEMENTIA, _load_dxdue
from src.dataset.build import (DAYS_PER_MONTH, IMAGE_META, SCALES, _build_regression_labels, _build_visit_table,
                               _load_inputs)
from src.dataset.build import VISIT_VALUE_FEATURES as V1_FEATURES

DX_FEATURES = ["dx_CN", "dx_MCI", "dx_DEM"]
VISIT_VALUE_FEATURES = V1_FEATURES + DX_FEATURES
VISIT_MISSING_FLAGS = [f"{c}_missing" for c in VISIT_VALUE_FEATURES]
SEVERITY = {"CN": 0, "MCI": 1, "Dementia": 2}


def load_diagnoses(root: Path, include_non_ad: bool = False) -> pd.DataFrame:
    """合格的诊断记录（日期有效、诊断非空），每人每天一条；同日冲突时取最严重的诊断（全库仅 2 例）。"""
    dx = pd.read_parquet(root / "data/processed/adni_v0/diagnoses_audit.parquet")
    dx = dx[dx["date_eligible"] & ~dx["date_missing"]].dropna(subset=["diagnosis", "date"]).copy()
    dx["non_ad"] = (dx["diagnosis"] == "DEM") & (dx["source_row"].map(_load_dxdue(root)) == NON_AD_DEMENTIA)
    if not include_non_ad:
        dx = dx[~dx["non_ad"]]
    dx["diagnosis"] = dx["diagnosis"].map({"CN": "CN", "MCI": "MCI", "DEM": "Dementia"})
    dx["subject_id"] = dx["subject_id"].astype("int64")
    dx["date"] = pd.to_datetime(dx["date"])
    dx["sev"] = dx["diagnosis"].map(SEVERITY)
    dx = dx.sort_values(["subject_id", "date", "sev"]).drop_duplicates(["subject_id", "date"], keep="last")
    return dx[["subject_id", "date", "diagnosis", "protocol", "visit_code", "screening", "non_ad"]].reset_index(drop=True)


def build_candidates(dx: pd.DataFrame, lo_months=21, hi_months=27, target_months=24, absorbing=False) -> pd.DataFrame:
    """
    所有 CN/MCI 诊断记录都是候选起点。返回每个候选的去向：
      judged —— 结局窗内有诊断（取最接近 target 的一次）
      absorbing —— （仅 absorbing=True）窗内无诊断，但窗前已有痴呆诊断且之后无非痴呆诊断
      insufficient_followup —— 此后的随访不足 lo 个月
      no_dx_in_window —— 随访够长，但结局窗内恰好没有诊断
    并标记：窗前是否已诊断痴呆（用于纳入 vs 排除比较）。
    """
    lo, hi = int(lo_months * DAYS_PER_MONTH), int(hi_months * DAYS_PER_MONTH)
    target = np.timedelta64(round(target_months * DAYS_PER_MONTH), "D")
    rows = []
    for sid, g in dx.groupby("subject_id", sort=False):
        dates, diags = g["date"].to_numpy(), g["diagnosis"].to_numpy()
        protos, vcodes, scr = g["protocol"].to_numpy(), g["visit_code"].to_numpy(), g["screening"].to_numpy()
        for i in range(len(g)):
            if diags[i] not in ("CN", "MCI"):
                continue
            d0 = dates[i]
            gap = (dates[i + 1:] - d0) / np.timedelta64(1, "D")
            fut_d, fut_dx = dates[i + 1:], diags[i + 1:]
            in_win = (gap >= lo) & (gap <= hi)
            before_win = gap < lo
            dem_before = bool((fut_dx[before_win] == "Dementia").any())
            rec = dict(subject_id=sid, index_date=d0, origin_diagnosis=diags[i], protocol=protos[i],
                       visit_code=vcodes[i], screening=bool(scr[i]), dementia_before_window=dem_before,
                       target_date=pd.NaT, target_diagnosis=None, actual_gap_days=np.nan)
            if in_win.any():
                k = np.argmin(np.abs((fut_d[in_win] - (d0 + target)) / np.timedelta64(1, "D")))
                rec.update(status="judged", target_date=fut_d[in_win][k], target_diagnosis=fut_dx[in_win][k],
                           actual_gap_days=gap[in_win][k])
            else:
                first_dem = np.flatnonzero(fut_dx == "Dementia")
                stays_dem = len(first_dem) and (fut_dx[first_dem[0]:] == "Dementia").all()
                if absorbing and dem_before and stays_dem:
                    rec.update(status="absorbing", target_date=fut_d[first_dem[0]], target_diagnosis="Dementia",
                               actual_gap_days=gap[first_dem[0]])
                elif len(gap) == 0 or gap.max() < lo:
                    rec.update(status="insufficient_followup")
                else:
                    rec.update(status="no_dx_in_window")
            rows.append(rec)
    cand = pd.DataFrame(rows)
    # 同一患者多个起点共用同一次结局诊断时，只保留最晚的起点
    lab = cand["status"].isin(["judged", "absorbing"])
    later = cand[lab].sort_values("index_date").duplicated(["subject_id", "target_date"], keep="last")
    cand.loc[later[later].index, "status"] = "duplicate_outcome"
    return cand


def _demographics(root: Path) -> pd.DataFrame:
    demo = pd.read_parquet(root / "data/processed/adni_v0/demographics_audit.parquet")
    demo = demo[demo["date_eligible"] & ~demo["date_missing"]].sort_values("date")
    demo["subject_id"] = demo["subject_id"].astype("int64")
    # 静态变量取每人最后一个非空值：后期访视的人口学记录常常只有日期、性别等字段为空，
    # 直接取最后一条记录会让这些人的性别、出生年和教育年限变成缺失（返修时发现，影响 34 人）
    demo = demo.groupby("subject_id")[["sex", "birth_year", "education_years"]].last().reset_index()
    raw = list(pyreadr.read_r(str(root / "data/raw/adni/ADNIMERGE2/data/PTDEMOG.rda")).values())[0]
    raw = raw.dropna(subset=["RID"]).copy()
    raw["subject_id"] = raw["RID"].astype("int64")
    race = (raw.dropna(subset=["PTRACCAT"]).drop_duplicates("subject_id", keep="last")
            .set_index("subject_id")[["PTRACCAT", "PTETHCAT"]])
    apoe = list(pyreadr.read_r(str(root / "data/raw/adni/ADNIMERGE2/data/APOERES.rda")).values())[0]
    apoe = apoe.dropna(subset=["RID", "GENOTYPE"]).copy()
    apoe["subject_id"] = apoe["RID"].astype("int64")
    apoe["apoe4"] = apoe["GENOTYPE"].astype(str).str.count("4").astype(float)
    apoe = apoe.drop_duplicates("subject_id", keep="last").set_index("subject_id")[["apoe4"]]
    out = demo.set_index("subject_id")[["sex", "birth_year", "education_years"]].join(race).join(apoe)
    return out.rename(columns={"PTRACCAT": "race", "PTETHCAT": "ethnicity"})


def build_dataset_v2(root, variant="main", split_from="dataset_v1"):
    """
    variant: main / all_cause / wide / absorbing
    返回 samples, sequences, candidates。main 变体沿用 v1 的患者级 train/val/test 划分（测试集患者不变）。
    """
    root = Path(root)
    dx = load_diagnoses(root, include_non_ad=(variant == "all_cause"))
    lo, hi = (18, 30) if variant == "wide" else (21, 27)
    cand = build_candidates(dx, lo, hi, 24, absorbing=(variant == "absorbing"))
    labels = cand[cand["status"].isin(["judged", "absorbing"])].copy()

    _, cog, mri = _load_inputs(root)
    reg = _build_regression_labels(labels.reset_index(drop=True), cog, 24, 3)
    labels = pd.concat([labels.reset_index(drop=True), reg.reset_index(drop=True)], axis=1)
    demo = _demographics(root)
    labels = labels.join(demo, on="subject_id")
    labels["approx_age_years"] = labels["index_date"].dt.year - labels["birth_year"]
    labels["sex_male"] = np.where(labels["sex"].isna(), np.nan, (labels["sex"] == "Male").astype(float))

    # 就诊日表：认知量表 + MRI + 当日诊断
    visits = _build_visit_table(cog, mri)
    dxd = dx[["subject_id", "date", "diagnosis"]].copy()
    for c, name in zip(DX_FEATURES, ["CN", "MCI", "Dementia"]):
        dxd[c] = (dxd["diagnosis"] == name).astype(float)
    visits = visits.merge(dxd.drop(columns="diagnosis"), on=["subject_id", "date"], how="outer")
    for c in VISIT_VALUE_FEATURES:
        visits[f"{c}_missing"] = visits[c].isna().astype("int8")
    visits = visits.sort_values(["subject_id", "date"]).reset_index(drop=True)

    meta = [c for c in IMAGE_META if c in visits.columns]
    cols = ["date", *VISIT_VALUE_FEATURES, *VISIT_MISSING_FLAGS, *meta]
    by_sub = {sid: g[cols] for sid, g in visits.groupby("subject_id")}
    frames, n_steps = [], []
    labels["sample_id"] = np.arange(len(labels))
    for r in labels.itertuples():
        g = by_sub.get(r.subject_id)
        hist = g[g["date"] <= r.index_date] if g is not None else None
        n_steps.append(0 if hist is None else len(hist))
        if hist is not None and len(hist):
            h = hist.copy()
            h.insert(0, "sample_id", r.sample_id)
            h.insert(1, "time_step", np.arange(len(h)))
            frames.append(h)
    labels["n_visits"] = n_steps
    seq = pd.concat(frames, ignore_index=True)
    seq["days_before_index"] = (seq["sample_id"].map(labels.set_index("sample_id")["index_date"]) - seq["date"]).dt.days.astype(float)

    samples = labels[labels["n_visits"] > 0].reset_index(drop=True)
    if variant == "main":
        v1 = pd.read_parquet(root / f"data/processed/{split_from}/samples.parquet")
        subj_split = v1.drop_duplicates("subject_id").set_index("subject_id")["split"]
        samples["split"] = samples["subject_id"].map(subj_split)
        assert samples["split"].notna().all(), "v2 主队列应是 v1 患者的子集"
    keep = ["sample_id", "subject_id", "index_date", "target_date", "actual_gap_days", "origin_diagnosis",
            "target_diagnosis", "status", "protocol", "visit_code", "screening",
            *[f"future_{s}" for s in SCALES], *[f"future_{s}_missing" for s in SCALES],
            "sex_male", "approx_age_years", "education_years", "race", "ethnicity", "apoe4", "n_visits"]
    if "split" in samples:
        keep.append("split")
    samples = samples[keep]
    seq = seq[seq["sample_id"].isin(samples["sample_id"])].reset_index(drop=True)
    return samples, seq, cand.join(demo, on="subject_id")
