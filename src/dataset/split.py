"""
按患者切分 train / val / test。

两条铁律：
1. 同一 subject_id 的全部样本必须整体归入同一个集合——同一患者的后期样本，其输入历史
   包含了早期样本的信息，拆开会造成数据泄漏，评估结果虚高。
2. 分层：保证起点诊断构成与转化结局在三个集合里比例一致，避免某个集合里阳性事件过少。
"""
import numpy as np
import pandas as pd

SEVERITY = {"CN": 0, "MCI": 1, "Dementia": 2}


def _patient_stratum(g: pd.DataFrame) -> str:
    """给每位患者一个分层标签：起点构成 × 最严重结局。"""
    origins = set(g["origin_diagnosis"])
    origin_mix = "both" if len(origins) > 1 else next(iter(origins))
    worst = max(g["target_diagnosis"], key=lambda d: SEVERITY.get(d, -1))
    return f"{origin_mix}|{worst}"


def assign_splits(
    samples: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """为每条样本分配 split 归属。

    Returns
    -------
    samples : 原表增加 split 列（train / val / test）
    report  : 各集合的样本数、患者数、起点与结局分布，供核对分层效果
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, "三个比例之和必须为 1"

    strata = (
        samples.groupby("subject_id")[["origin_diagnosis", "target_diagnosis"]]
        .apply(_patient_stratum)
        .rename("stratum")
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    assignment = {}
    for stratum, g in strata.groupby("stratum"):
        pids = g["subject_id"].to_numpy().copy()
        rng.shuffle(pids)
        n = len(pids)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        # 小分层保证至少不把样本全丢给 train：按顺序切，余数归 test
        for pid in pids[:n_train]:
            assignment[pid] = "train"
        for pid in pids[n_train : n_train + n_val]:
            assignment[pid] = "val"
        for pid in pids[n_train + n_val :]:
            assignment[pid] = "test"

    samples = samples.copy()
    samples["split"] = samples["subject_id"].map(assignment)

    rows = []
    for split in ("train", "val", "test"):
        sub = samples[samples["split"] == split]
        row = {
            "split": split,
            "样本数": len(sub),
            "患者数": sub["subject_id"].nunique(),
            "MCI起点": int((sub["origin_diagnosis"] == "MCI").sum()),
            "CN起点": int((sub["origin_diagnosis"] == "CN").sum()),
        }
        for outcome in ("CN", "MCI", "Dementia"):
            row[f"结局_{outcome}"] = int((sub["target_diagnosis"] == outcome).sum())
        mci = sub[sub["origin_diagnosis"] == "MCI"]
        row["MCI→痴呆阳性"] = int((mci["target_diagnosis"] == "Dementia").sum())
        row["MCI阳性率%"] = round(row["MCI→痴呆阳性"] / len(mci) * 100, 1) if len(mci) else None
        rows.append(row)
    report = pd.DataFrame(rows).set_index("split")

    return samples, report


def verify_no_leakage(samples: pd.DataFrame) -> dict:
    """核验：没有任何患者跨越多个 split。"""
    per_patient = samples.groupby("subject_id")["split"].nunique()
    offenders = per_patient[per_patient > 1]
    return {
        "跨集合的患者数": int(len(offenders)),
        "通过": len(offenders) == 0,
        "示例": offenders.index[:5].tolist(),
    }
