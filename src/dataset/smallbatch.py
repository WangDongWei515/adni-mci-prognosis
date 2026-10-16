"""
分层小批量采样：为本地冒烟测试挑选样本。

随机抽样抽到的大多是「规矩样本」，测不到真正容易出错的代码分支。这里按类型定额挑选，
确保 padding / attention mask / 缺失编码这些高风险逻辑都被真正走到。
"""
import numpy as np
import pandas as pd


def sample_smallbatch(
    samples: pd.DataFrame,
    sequences: pd.DataFrame,
    per_group: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按边界情况定额挑选样本。

    Returns
    -------
    sb_samples   : 挑选出的样本
    sb_sequences : 对应的序列行
    coverage     : 每类边界情况实际挑到几条，供核对是否真的覆盖到
    """
    rng = np.random.default_rng(seed)
    train = samples[samples["split"] == "train"] if "split" in samples.columns else samples

    # 每条样本的序列里，是否存在「某次就诊缺了某个模态」
    miss_cols = [c for c in sequences.columns if c.endswith("_missing")]
    per_sample_miss = sequences.groupby("sample_id")[miss_cols].max().sum(axis=1)
    has_missing = set(per_sample_miss[per_sample_miss > 0].index)

    # 贡献多个起点的患者
    multi_origin_pids = set(
        train.groupby("subject_id").size()[lambda s: s >= 3].index
    )

    groups = {
        "序列长度=1（padding 极端情况）": train["n_visits"] == 1,
        "历史很长（≥10 次，padding 另一端）": train["n_visits"] >= 10,
        "含缺失模态（[MISS] 编码）": train["sample_id"].isin(has_missing),
        "同一患者贡献多个起点（切分逻辑）": train["subject_id"].isin(multi_origin_pids),
        "起点=CN": train["origin_diagnosis"] == "CN",
        "起点=MCI": train["origin_diagnosis"] == "MCI",
        "结局=CN": train["target_diagnosis"] == "CN",
        "结局=MCI": train["target_diagnosis"] == "MCI",
        "结局=痴呆（阳性事件）": train["target_diagnosis"] == "Dementia",
        "头B有MoCA标签（稀疏输出头）": train.get(
            "future_MoCA_missing", pd.Series(1, index=train.index)
        ) == 0,
    }

    picked, rows = set(), []
    for name, mask in groups.items():
        pool = train.loc[mask, "sample_id"].to_numpy()
        n_avail = len(pool)
        take = rng.choice(pool, size=min(per_group, n_avail), replace=False) if n_avail else []
        picked.update(int(x) for x in take)
        rows.append({"边界情况": name, "可选样本数": n_avail, "本次挑选": len(take)})

    coverage = pd.DataFrame(rows)
    sb_samples = samples[samples["sample_id"].isin(picked)].reset_index(drop=True)
    sb_sequences = sequences[sequences["sample_id"].isin(picked)].reset_index(drop=True)
    return sb_samples, sb_sequences, coverage
