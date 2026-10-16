"""
把「标签表 + 认知量表 + MRI 体积」拼成模型可用的序列数据集。

产出两张表：
  samples   —— 一行 = 一条训练样本（患者 × index_date），含双头标签、静态特征、划分归属
  sequences —— 一行 = 一条样本的一个时间步（一次就诊），含该次就诊的全部模态特征

设计要点（均为前几轮讨论确定）：
- 时间步按「自然日」定义，临近日期的检查不强行合并为同一次访视
- 只使用 date <= index_date 的记录作为输入（严禁未来信息回流）
- 缺失模态保留占位并置 *_missing 标志位，由模型侧的 [MISS] 机制处理
- 头 A 标签（未来诊断状态）来自标签表；头 B 标签（未来量表分数）在此按同一时间窗规则构建，
  确保两个头对齐在同一批样本、同一个 index_date 上
- MRI 体积同时给出原始 mm³ 与 ICV 归一化比值，由模型自行学习哪个有用
"""
from pathlib import Path

import numpy as np
import pandas as pd

SCALES = ["ADAS13", "CDRSB", "MMSE", "MoCA"]

MRI_RAW = [
    "hippocampus_left_mm3",
    "hippocampus_right_mm3",
    "entorhinal_left_mm3",
    "entorhinal_right_mm3",
    "icv_mm3",
]
MRI_NORM = [
    "hippocampus_left_icv",
    "hippocampus_right_icv",
    "entorhinal_left_icv",
    "entorhinal_right_icv",
]

# 每个时间步的数值特征（缺失时置 NaN，并由同名 *_missing 列标记）
VISIT_VALUE_FEATURES = SCALES + MRI_RAW + MRI_NORM
VISIT_MISSING_FLAGS = [f"{c}_missing" for c in VISIT_VALUE_FEATURES]

STATIC_FEATURES = ["sex_male", "approx_age_years", "education_years"]

# 影像溯源字段：不参与建模，是留给论文 2（原始影像）的接口。
# image_uid 是 ADNI 扫描编号，日后从 LONI IDA 下载原始影像后按它把每个时间步关联到具体扫描文件；
# field_strength（1.5T/3T）在原始影像建模时用于协议调和。
IMAGE_META = ["image_uid", "field_strength"]

DAYS_PER_MONTH = 30.4375


def _load_inputs(root: Path):
    d = root / "data/processed/adni_v0"
    labels = pd.read_parquet(d / "diagnosis_transitions.parquet")
    cog = pd.read_parquet(d / "cognition_long.parquet")
    mri = pd.read_parquet(d / "mri_volumes.parquet")

    # 统一类型：三张表的 subject_id / date 精度不一致，拼接前必须对齐
    for df in (labels, cog, mri):
        df["subject_id"] = df["subject_id"].astype("int64")
        df["date"] = pd.to_datetime(df["date"]) if "date" in df.columns else None
    labels["index_date"] = pd.to_datetime(labels["index_date"])
    labels["target_date"] = pd.to_datetime(labels["target_date"])
    return labels, cog, mri


def _build_visit_table(cog: pd.DataFrame, mri: pd.DataFrame) -> pd.DataFrame:
    """把认知量表长表和 MRI 表合成「每人每天一行」的宽表，即所有可能的时间步。"""
    cog_wide = (
        cog.pivot_table(index=["subject_id", "date"], columns="scale", values="score", aggfunc="first")
        .reindex(columns=SCALES)
        .reset_index()
    )
    mri_meta = [c for c in IMAGE_META if c in mri.columns]
    mri_use = mri[["subject_id", "date", *MRI_RAW, *mri_meta]].copy()

    visits = cog_wide.merge(mri_use, on=["subject_id", "date"], how="outer")

    # ICV 归一化比值（ICV 缺失或为 0 时保持 NaN）
    icv = visits["icv_mm3"].replace(0, np.nan)
    for raw, norm in zip(MRI_RAW[:4], MRI_NORM):
        visits[norm] = visits[raw] / icv

    for col in VISIT_VALUE_FEATURES:
        visits[f"{col}_missing"] = visits[col].isna().astype("int8")

    return visits.sort_values(["subject_id", "date"]).reset_index(drop=True)


def _build_regression_labels(
    labels: pd.DataFrame,
    cog: pd.DataFrame,
    horizon_months: int,
    tolerance_months: int,
) -> pd.DataFrame:
    """为每条样本构建头 B 标签：未来同一时间窗内各量表的分数（没测到的留空，训练时 mask 掉）。"""
    lo = int((horizon_months - tolerance_months) * DAYS_PER_MONTH)
    hi = int((horizon_months + tolerance_months) * DAYS_PER_MONTH)
    target_delta = pd.Timedelta(days=round(horizon_months * DAYS_PER_MONTH))

    # 日期与分数分开存为同构数组，避免 object 数组导致日期运算失败
    cog_by_sub_scale = {}
    for key, g in cog.dropna(subset=["score"]).groupby(["subject_id", "scale"]):
        g = g.sort_values("date")
        cog_by_sub_scale[key] = (
            g["date"].to_numpy(dtype="datetime64[ns]"),
            g["score"].to_numpy(dtype="float64"),
        )

    out = {f"future_{s}": [] for s in SCALES}
    for row in labels.itertuples():
        ideal = np.datetime64(row.index_date + target_delta)
        idx = np.datetime64(row.index_date)
        for scale in SCALES:
            entry = cog_by_sub_scale.get((row.subject_id, scale))
            value = np.nan
            if entry is not None:
                dates, scores = entry
                gap = (dates - idx) / np.timedelta64(1, "D")
                sel = (gap >= lo) & (gap <= hi)
                if sel.any():
                    cand_dates, cand_scores = dates[sel], scores[sel]
                    closest = np.abs((cand_dates - ideal) / np.timedelta64(1, "D")).argmin()
                    value = float(cand_scores[closest])
            out[f"future_{scale}"].append(value)

    reg = pd.DataFrame(out, index=labels.index)
    for scale in SCALES:
        reg[f"future_{scale}_missing"] = reg[f"future_{scale}"].isna().astype("int8")
    return reg


def build_dataset(
    root: str | Path,
    horizon_months: int = 24,
    tolerance_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """构建序列数据集。

    Returns
    -------
    samples   : 每条训练样本一行（标签、静态特征、序列长度）
    sequences : 每条样本的每个时间步一行（就诊特征 + 距 index_date 的天数）
    stats     : 构建过程的统计信息（供 notebook 展示与论文报告）
    """
    root = Path(root)
    labels, cog, mri = _load_inputs(root)
    visits = _build_visit_table(cog, mri)

    # 头 B 标签：与头 A 对齐在同一批样本上
    reg = _build_regression_labels(labels, cog, horizon_months, tolerance_months)
    labels = pd.concat([labels.reset_index(drop=True), reg.reset_index(drop=True)], axis=1)

    # 静态特征
    labels["sex_male"] = (labels["sex"] == "Male").astype("float64")
    labels.loc[labels["sex"].isna(), "sex_male"] = np.nan

    present_meta = [c for c in IMAGE_META if c in visits.columns]
    visit_cols = ["date", *VISIT_VALUE_FEATURES, *VISIT_MISSING_FLAGS, *present_meta]
    visits_by_sub = {sid: g[visit_cols].to_numpy() for sid, g in visits.groupby("subject_id")}
    date_pos = 0

    seq_frames = []
    n_visits = []
    for i, row in enumerate(labels.itertuples()):
        arr = visits_by_sub.get(row.subject_id)
        if arr is None:
            n_visits.append(0)
            continue
        hist = arr[arr[:, date_pos] <= row.index_date]
        n_visits.append(len(hist))
        if len(hist) == 0:
            continue
        frame = pd.DataFrame(hist, columns=visit_cols)
        frame.insert(0, "sample_id", i)
        frame.insert(1, "time_step", np.arange(len(hist)))
        seq_frames.append(frame)

    labels["n_visits"] = n_visits

    # 无任何历史记录的样本无法作为输入，剔除并记录
    n_empty = int((labels["n_visits"] == 0).sum())
    labels["sample_id"] = np.arange(len(labels))
    samples = labels[labels["n_visits"] > 0].reset_index(drop=True)

    sequences = pd.concat(seq_frames, ignore_index=True)
    sequences["date"] = pd.to_datetime(sequences["date"])
    idx_lookup = samples.set_index("sample_id")["index_date"]
    sequences["days_before_index"] = (
        idx_lookup.reindex(sequences["sample_id"]).to_numpy() - sequences["date"].to_numpy()
    ) / np.timedelta64(1, "D")

    for col in VISIT_VALUE_FEATURES:
        sequences[col] = pd.to_numeric(sequences[col], errors="coerce")
    for col in VISIT_MISSING_FLAGS:
        sequences[col] = sequences[col].astype("int8")

    sample_cols = [
        "sample_id", "subject_id", "index_date", "target_date",
        "horizon_months", "tolerance_months", "actual_gap_days",
        "origin_diagnosis", "target_diagnosis",
        *[f"future_{s}" for s in SCALES], *[f"future_{s}_missing" for s in SCALES],
        *STATIC_FEATURES, "n_visits",
    ]
    samples = samples[[c for c in sample_cols if c in samples.columns]]

    seq_cols = ["sample_id", "time_step", "date", "days_before_index",
                *VISIT_VALUE_FEATURES, *VISIT_MISSING_FLAGS, *present_meta]
    sequences = sequences[seq_cols]

    stats = {
        "原始标签数": int(len(labels)),
        "因无历史记录剔除": n_empty,
        "最终样本数": int(len(samples)),
        "覆盖患者数": int(samples["subject_id"].nunique()),
        "序列总行数": int(len(sequences)),
        "序列长度_中位": float(samples["n_visits"].median()),
        "序列长度_最大": int(samples["n_visits"].max()),
        "长度为1的样本数": int((samples["n_visits"] == 1).sum()),
        "头B各量表有标签的样本数": {
            s: int((samples[f"future_{s}_missing"] == 0).sum()) for s in SCALES
        },
        "带影像溯源编号的时间步数": (
            int(sequences["image_uid"].notna().sum()) if "image_uid" in sequences.columns else 0
        ),
    }
    return samples, sequences, stats
