"""
构建头 A（未来诊断状态三分类）的标签表。

核心规则（对应架构页与项目记忆里的定稿设计）：
- 以每条 CN/MCI 诊断记录为预测起点（index_date）
- 在 [horizon-tolerance, horizon+tolerance] 月的窗口内找最靠近 horizon 的未来诊断，
  作为 target_diagnosis（CN/MCI/Dementia 三选一）
- 窗口内无记录、且随访不足 (horizon-tolerance) 个月的样本标记为删失（censored），
  删失样本从最终标签表剔除，但计入流失漏斗统计
- 剔除病因非 AD 的痴呆记录（DXDDUE == "Dementia due to other etiology"）——
  这类记录既不能作为可靠的"起点"也不能作为可靠的"标签"
"""
from pathlib import Path

import pandas as pd
import pyreadr

NON_AD_DEMENTIA = "Dementia due to other etiology"
AD_DEMENTIA = "Dementia due to Alzheimer's Disease"


def _load_dxdue(root: Path) -> pd.Series:
    """从原始 DXSUM.rda 读取 DXDDUE（痴呆病因）字段，索引为源表行号（0-based）。"""
    dxsum_path = root / "data/raw/adni/ADNIMERGE2/data/DXSUM.rda"
    res = pyreadr.read_r(str(dxsum_path))
    dxsum = list(res.values())[0]
    return dxsum["DXDDUE"]


def build_diagnosis_transitions(
    root: str | Path,
    horizon_months: int = 24,
    tolerance_months: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """构建诊断状态转移标签表 + 流失漏斗统计。

    Returns
    -------
    labels : 最终标签表（已剔除删失和非AD痴呆），列见模块末尾说明
    funnel : dict，记录每一步筛选后剩余的样本数/患者数，MCI/CN 起点分别统计
    """
    root = Path(root)
    audit_path = root / "data/processed/adni_v0/diagnoses_audit.parquet"
    dx_audit = pd.read_parquet(audit_path)

    # 第 1 步：只保留日期合法、诊断非空的记录
    dx_ok = dx_audit[dx_audit["date_eligible"] & ~dx_audit["date_missing"]].dropna(
        subset=["diagnosis", "date"]
    ).copy()
    n_total_records = len(dx_ok)
    n_total_patients = dx_ok["subject_id"].nunique()

    # 第 2 步：用 source_row 直接对齐原始 DXSUM 取 DXDDUE，剔除非 AD 病因的痴呆记录
    dxdue = _load_dxdue(root)
    dx_ok["dxdue"] = dx_ok["source_row"].map(dxdue)
    non_ad_mask = (dx_ok["diagnosis"] == "DEM") & (dx_ok["dxdue"] == NON_AD_DEMENTIA)
    n_excluded_non_ad = int(non_ad_mask.sum())
    dx_ok = dx_ok.loc[~non_ad_mask].copy()

    # 统一诊断标签文案：DEM -> Dementia（论文表述用 progression to dementia）
    diag_map = {"CN": "CN", "MCI": "MCI", "DEM": "Dementia"}
    dx_ok["diagnosis"] = dx_ok["diagnosis"].map(diag_map)

    dx_seq = (
        dx_ok[["subject_id", "date", "diagnosis"]]
        .drop_duplicates()
        .sort_values(["subject_id", "date"])
        .reset_index(drop=True)
    )

    lo_days = int((horizon_months - tolerance_months) * 30.4375)
    hi_days = int((horizon_months + tolerance_months) * 30.4375)
    target_delta = pd.Timedelta(days=round(horizon_months * 30.4375))

    # 人口学信息不在 diagnoses_audit 里，需要单独从 demographics_audit 关联
    demo_path = root / "data/processed/adni_v0/demographics_audit.parquet"
    demo_audit = pd.read_parquet(demo_path)
    demo_ok = demo_audit[demo_audit["date_eligible"] & ~demo_audit["date_missing"]]
    demo_lookup = (
        demo_ok.sort_values("date")
        .drop_duplicates(subset="subject_id", keep="last")
        .set_index("subject_id")[["sex", "birth_year", "education_years"]]
    )
    have_demo = ["sex", "approx_age_years", "education_years"]

    rows = []
    funnel_rows = []
    for sid, g in dx_seq.groupby("subject_id"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            d0, dg0 = g.loc[i, "date"], g.loc[i, "diagnosis"]
            if dg0 not in ("CN", "MCI"):
                continue
            fut = g[g["date"] > d0]
            gap_days = (fut["date"] - d0).dt.days
            win = fut[gap_days.between(lo_days, hi_days)]
            if len(win):
                target_date_ideal = d0 + target_delta
                pick = win.iloc[(win["date"] - target_date_ideal).abs().argsort().iloc[0]]
                rows.append(
                    dict(
                        subject_id=sid,
                        index_date=d0,
                        target_date=pick["date"],
                        origin_diagnosis=dg0,
                        target_diagnosis=pick["diagnosis"],
                        actual_gap_days=int((pick["date"] - d0).days),
                        censored=False,
                    )
                )
                funnel_rows.append((sid, dg0, "judged"))
            else:
                is_censored = len(fut) == 0 or (fut["date"].max() - d0).days < lo_days
                if is_censored:
                    funnel_rows.append((sid, dg0, "censored"))
                else:
                    funnel_rows.append((sid, dg0, "gap_not_covered"))

    labels = pd.DataFrame(rows)
    labels["horizon_months"] = horizon_months
    labels["tolerance_months"] = tolerance_months
    if len(labels):
        labels = labels.join(demo_lookup, on="subject_id")
        labels["approx_age_years"] = labels["index_date"].dt.year - labels["birth_year"]
        labels = labels.drop(columns=["birth_year"])

    col_order = [
        "subject_id",
        "index_date",
        "target_date",
        "horizon_months",
        "tolerance_months",
        "actual_gap_days",
        "origin_diagnosis",
        "target_diagnosis",
        "censored",
        *have_demo,
    ]
    labels = labels[[c for c in col_order if c in labels.columns]]

    # 流失漏斗统计
    funnel_df = pd.DataFrame(funnel_rows, columns=["subject_id", "origin", "outcome"])
    funnel = {
        "诊断人群总记录数": n_total_records,
        "诊断人群总患者数": n_total_patients,
        "剔除的非AD病因痴呆记录数": n_excluded_non_ad,
    }
    for origin in ("MCI", "CN"):
        sub = funnel_df[funnel_df["origin"] == origin]
        n_candidate = len(sub)
        n_judged = int((sub["outcome"] == "judged").sum())
        n_censored = int((sub["outcome"] == "censored").sum())
        n_gap_not_covered = int((sub["outcome"] == "gap_not_covered").sum())
        n_patients = sub["subject_id"].nunique()
        lab_sub = labels[labels["origin_diagnosis"] == origin] if len(labels) else labels
        if origin == "MCI":
            n_pos = int((lab_sub["target_diagnosis"] == "Dementia").sum()) if len(lab_sub) else 0
        else:
            n_pos = int(lab_sub["target_diagnosis"].isin(["MCI", "Dementia"]).sum()) if len(lab_sub) else 0
        funnel[origin] = dict(
            候选样本=n_candidate,
            可判定=n_judged,
            删失=n_censored,
            窗口外未覆盖=n_gap_not_covered,
            阳性事件=n_pos,
            阳性率=round(n_pos / n_judged * 100, 1) if n_judged else None,
            删失率=round(n_censored / n_candidate * 100, 1) if n_candidate else None,
            独立患者数=int(n_patients),
        )

    return labels, funnel
