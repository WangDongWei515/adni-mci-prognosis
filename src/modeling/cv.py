"""
稳健性分析用的按患者 5 折交叉验证，可重复多次（不同随机种子 → 不同的患者分折）。

每个患者在每一次重复中恰好在一折里当一次测试集，5 折的测试预测拼起来覆盖全部样本。
每一折内部再按患者、按「起点诊断 × 结局」分层留出约 1/7 作为早停用的验证集。
超参数沿用主划分上选出的（非嵌套交叉验证，文中已说明）。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CV_PARTITION_SEEDS = (2026, 2027, 2028)


def _strata(samples):
    return samples["origin_diagnosis"] + "_" + samples["target_diagnosis"]


def make_cv_folds(samples: pd.DataFrame, k: int = 5, seed: int = 2026) -> pd.DataFrame:
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fold = np.full(len(samples), -1)
    for f, (_, te) in enumerate(sgkf.split(samples, _strata(samples), samples["subject_id"])):
        fold[te] = f
    return pd.DataFrame({"sample_id": samples["sample_id"].to_numpy(), "fold": fold})


def fold_samples(samples: pd.DataFrame, folds: pd.DataFrame, f: int, seed: int = 2026) -> pd.DataFrame:
    """返回第 f 折重新标注 split 列后的样本表：第 f 折 = test，其余按患者分层再切出约 1/7 作 val。"""
    s = samples.drop(columns="split", errors="ignore").merge(folds, on="sample_id")
    rest = s[s["fold"] != f]
    inner = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed + f)
    tr_idx, va_idx = next(inner.split(rest, _strata(rest), rest["subject_id"]))
    split = pd.Series("test", index=s.index)
    split.loc[rest.index[tr_idx]] = "train"
    split.loc[rest.index[va_idx]] = "val"
    s["split"] = split
    return s.drop(columns="fold").sort_values("sample_id").reset_index(drop=True)


def temporal_split(samples: pd.DataFrame, seed: int = 2026) -> pd.DataFrame:
    """
    时间外推验证：按入组阶段划分参与者。ADNI1/GO/2 入组者训练（只用其 ADNI1/GO/2 期间的预测点，
    内部按患者分层留出约 1/7 作验证集），ADNI3/4 新入组者测试。两组没有共同的参与者。
    """
    first = samples.sort_values("index_date").groupby("subject_id")["protocol"].first()
    old = {"ADNI1", "ADNIGO", "ADNI2"}
    s = samples.copy()
    s["enrol_old"] = s["subject_id"].map(first).isin(old)
    s = s[(s["enrol_old"] & s["protocol"].isin(old)) | ~s["enrol_old"]].reset_index(drop=True)
    s["split"] = np.where(s["enrol_old"], "train", "test")
    tr = s[s["split"] == "train"]
    inner = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    _, va = next(inner.split(tr, _strata(tr), tr["subject_id"]))
    s.loc[tr.index[va], "split"] = "val"
    return s.drop(columns="enrol_old")
