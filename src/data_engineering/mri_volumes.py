"""
从 ADNIMERGE2 原始 UCSFFSX7（FreeSurfer 7.x 全协议重处理）表提取 MRI 体积特征，
产出风格对齐 cognition_long.parquet 的长表——项目的第二模态。

字段来源（已用 DATADIC 核实）：
  ST29SV = 左海马体积 (aseg, mm3)      ST88SV = 右海马体积 (aseg, mm3)
  ST24CV = 左内嗅皮层体积 (aparc, mm3) ST83CV = 右内嗅皮层体积 (aparc, mm3)
  ST10CV = 颅内总容积 ICV (mm3)，常用于体积归一化

另外保留两个**影像溯源字段**（不作为模型特征，是给论文 2 留的接口）：
  IMAGEUID       = 该次扫描在 ADNI 的唯一编号，日后从 LONI IDA 下载原始影像后按它关联
  FIELD_STRENGTH = 场强 1.5T / 3T，原始影像建模时需要做协议调和
"""
from pathlib import Path

import pandas as pd
import pyreadr

VOLUME_FIELDS = {
    "ST29SV": "hippocampus_left_mm3",
    "ST88SV": "hippocampus_right_mm3",
    "ST24CV": "entorhinal_left_mm3",
    "ST83CV": "entorhinal_right_mm3",
    "ST10CV": "icv_mm3",
}


def build_mri_volumes(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    res = pyreadr.read_r(str(root / "data/raw/adni/ADNIMERGE2/data/UCSFFSX7.rda"))
    df = list(res.values())[0]

    # 剔除明确质量不合格的行；OVERALLQC 大量缺失属正常（未人工复核），不视为问题
    df = df[df["OVERALLQC"] != "Fail"].copy()

    # 同一 (RID, EXAMDATE) 有多个 FreeSurfer 版本重跑时：优先取版本更新的一次
    df["_fsver_rank"] = df["FSVER"].rank(method="dense")
    df = df.sort_values(["RID", "EXAMDATE", "_fsver_rank"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["RID", "EXAMDATE"], keep="first")

    keep_cols = ["RID", "EXAMDATE", "COLPROT", "OVERALLQC", "FSVER",
                 "IMAGEUID", "FIELD_STRENGTH"] + list(VOLUME_FIELDS)
    df = df[keep_cols].rename(columns={
        "RID": "subject_id", "EXAMDATE": "date",
        "IMAGEUID": "image_uid", "FIELD_STRENGTH": "field_strength",
        **VOLUME_FIELDS,
    })
    df["date"] = pd.to_datetime(df["date"])
    df["subject_id"] = df["subject_id"].astype("Int64")

    # 至少有一个体积字段非空才保留
    vol_cols = list(VOLUME_FIELDS.values())
    df = df[df[vol_cols].notna().any(axis=1)].reset_index(drop=True)

    return df.sort_values(["subject_id", "date"]).reset_index(drop=True)
