"""
阶段 3–5 共用的数据准备：把 dataset_v2 的两张表变成模型可直接消费的形式。

产出三种视图（同一批样本、同一套标准化统计量，只是组织方式不同）：
  - 序列张量      —— 给 Transformer / GRU（按自然日的不规则时间步）
  - 年度分箱张量  —— 给 LongForMAD 启发的基线（归到最近的年度访视，起点年 + 前 3 年）
  - 表格特征      —— 给 GBM 基线（起点快照 / 历史聚合统计量）

铁律：所有标准化统计量只用训练集计算，验证集和测试集沿用训练集的均值和标准差。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset.build import SCALES, STATIC_FEATURES
from src.dataset.cohort_v2 import VISIT_VALUE_FEATURES, load_diagnoses

CLASSES = ["CN", "MCI", "Dementia"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DAYS_PER_YEAR = 365.25


@dataclass
class Normalizer:
    """训练集上拟合的标准化参数。"""
    visit_mean: np.ndarray
    visit_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    static_cols: tuple = tuple(STATIC_FEATURES)


def load_dataset(root: Path, variant: str | None = None):
    """variant=None 为主队列（dataset_v2）；all_cause / wide / absorbing 为敏感性分析队列。"""
    d = root / "data/processed/dataset_v2"
    if variant:
        d = d / "variants" / variant
    samples = pd.read_parquet(d / "samples.parquet").sort_values("sample_id").reset_index(drop=True)
    seq = pd.read_parquet(d / "sequences.parquet").sort_values(["sample_id", "time_step"]).reset_index(drop=True)
    samples["y"] = samples["target_diagnosis"].map(CLASS_TO_IDX).astype(int)
    samples["origin_mci"] = (samples["origin_diagnosis"] == "MCI").astype(float)
    return samples, seq


def reversion_flags(root: Path) -> pd.DataFrame:
    """
    每位患者在其全部诊断记录中是否出现过 MCI→CN（相邻两次诊断从 MCI 变为 CN，包括 CN→MCI→CN），
    以及是否出现过痴呆→MCI/CN（仅用于描述，不参与筛选）。

    用于复现 LongForMAD 的队列筛选（它剔除了 CN→MCI→CN 与 MCI 后来被诊断为 CN 的参与者）。注意：这个标记用到了
    患者整个随访期（包括预测起点之后）的信息，做预测时无法事先得知。
    """
    dx = load_diagnoses(root)
    dx["prev"] = dx.groupby("subject_id")["diagnosis"].shift()
    dx["mci_to_cn"] = (dx["prev"] == "MCI") & (dx["diagnosis"] == "CN")
    dx["dem_to_lower"] = (dx["prev"] == "Dementia") & dx["diagnosis"].isin(["CN", "MCI"])
    return dx.groupby("subject_id")[["mci_to_cn", "dem_to_lower"]].any()


def curated_mask(samples: pd.DataFrame, rev: pd.DataFrame) -> pd.Series:
    """LongForMAD 式队列：剔除曾出现 MCI→CN 的参与者，并剔除 CN 起点两年后为痴呆的预测点。"""
    ever_rev = samples["subject_id"].map(rev["mci_to_cn"]).fillna(False).astype(bool)
    cn_to_dem = (samples["origin_diagnosis"] == "CN") & (samples["target_diagnosis"] == "Dementia")
    return ~ever_rev & ~cn_to_dem


def fit_normalizer(samples: pd.DataFrame, seq: pd.DataFrame, static_cols=tuple(STATIC_FEATURES)) -> Normalizer:
    """
    就诊层特征的均值/标准差按训练集参与者的「每人每天一条」计算——同一次就诊会出现在多个预测点的历史里，
    按序列行计算会让随访多的参与者被重复加权。
    """
    tr = samples[samples["split"] == "train"]
    tr_seq = seq[seq["sample_id"].isin(tr["sample_id"])].copy()
    tr_seq["subject_id"] = tr_seq["sample_id"].map(tr.set_index("sample_id")["subject_id"])
    tr_seq = tr_seq.drop_duplicates(["subject_id", "date"])
    tcols = [f"future_{s}" for s in SCALES]
    sc = list(static_cols)
    return Normalizer(
        visit_mean=tr_seq[VISIT_VALUE_FEATURES].mean().to_numpy(float),
        visit_std=tr_seq[VISIT_VALUE_FEATURES].std().to_numpy(float),
        static_mean=tr[sc].mean().to_numpy(float),
        static_std=tr[sc].std().to_numpy(float),
        target_mean=tr[tcols].mean().to_numpy(float),
        target_std=tr[tcols].std().to_numpy(float),
        static_cols=tuple(sc),
    )


def static_matrix(samples: pd.DataFrame, norm: Normalizer) -> np.ndarray:
    """静态输入：年龄、性别、受教育年限（标准化，缺失置 0 并附缺失标志）+ 起点诊断。"""
    raw = samples[list(norm.static_cols)].to_numpy(float)
    miss = np.isnan(raw).astype(float)
    z = np.nan_to_num((raw - norm.static_mean) / norm.static_std)
    return np.concatenate([z, miss, samples[["origin_mci"]].to_numpy(float)], axis=1).astype(np.float32)


def target_matrices(samples: pd.DataFrame, norm: Normalizer):
    """头 B 回归标签（标准化后）与标签 mask（1 = 该量表没有标签，不计入 loss）。"""
    raw = samples[[f"future_{s}" for s in SCALES]].to_numpy(float)
    miss = np.isnan(raw)
    z = np.nan_to_num((raw - norm.target_mean) / norm.target_std)
    return z.astype(np.float32), miss.astype(np.float32)


def sequence_arrays(samples: pd.DataFrame, seq: pd.DataFrame, norm: Normalizer, max_years: float | None = None):
    """
    按自然日时间步组织的变长序列。每个样本返回：
      x    (T, F)  标准化后的特征值，缺失处置 0（真正的缺失信息在 m 里）
      m    (T, F)  缺失标志，1 = 缺失 → 模型侧用可学习的 [MISS] 嵌入替代
      t    (T,)    距预测起点的年数（连续时间，时间编码的输入）
    max_years 用于「回看窗口」消融：只保留起点前 max_years 年内的时间步。
    """
    s = seq[seq["sample_id"].isin(samples["sample_id"])]
    if max_years is not None:
        s = s[s["days_before_index"] <= max_years * DAYS_PER_YEAR]
    vals = s[VISIT_VALUE_FEATURES].to_numpy(float)
    m = np.isnan(vals).astype(np.float32)
    x = np.nan_to_num((vals - norm.visit_mean) / norm.visit_std).astype(np.float32)
    t = (s["days_before_index"].to_numpy(float) / DAYS_PER_YEAR).astype(np.float32)
    ids = s["sample_id"].to_numpy()
    out = {}
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    for a, b in zip(starts, ends):
        out[ids[a]] = (x[a:b], m[a:b], t[a:b])
    xs, ms, ts = [], [], []
    for sid in samples["sample_id"]:
        # 窗口截断后若一个时间步都不剩，保留一个全缺失的占位步（模型仍可依赖静态信息）
        if sid in out:
            xi, mi, ti = out[sid]
        else:
            F = len(VISIT_VALUE_FEATURES)
            xi, mi, ti = np.zeros((1, F), np.float32), np.ones((1, F), np.float32), np.zeros(1, np.float32)
        xs.append(xi); ms.append(mi); ts.append(ti)
    return xs, ms, ts


def annual_bin_arrays(samples: pd.DataFrame, seq: pd.DataFrame, norm: Normalizer, n_years: int = 4):
    """
    LongForMAD 式输入：把历史按「距起点第 k 年」归并为最多 n_years 个年度 token（默认起点年 + 前 3 年），
    同一年内的多次测量取均值；没有任何测量的年份不生成 token。
    返回与 sequence_arrays 相同的格式，t 为年度序号 k（0 = 起点所在年）。
    """
    s = seq[seq["sample_id"].isin(samples["sample_id"])].copy()
    s["k"] = np.round(s["days_before_index"] / DAYS_PER_YEAR).astype(int)  # 归到最近的年度访视
    s = s[s["k"] < n_years]
    g = s.groupby(["sample_id", "k"])[VISIT_VALUE_FEATURES].mean().reset_index()
    g = g.sort_values(["sample_id", "k"], ascending=[True, False])
    return _to_arrays(samples, g, norm)


def _to_arrays(samples, g, norm):
    vals = g[VISIT_VALUE_FEATURES].to_numpy(float)
    m = np.isnan(vals).astype(np.float32)
    x = np.nan_to_num((vals - norm.visit_mean) / norm.visit_std).astype(np.float32)
    k = g["k"].to_numpy(np.float32)
    ids = g["sample_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    out = {ids[a]: (x[a:b], m[a:b], k[a:b]) for a, b in zip(starts, ends)}
    F = len(VISIT_VALUE_FEATURES)
    blank = (np.zeros((1, F), np.float32), np.ones((1, F), np.float32), np.zeros(1, np.float32))
    res = [out.get(sid, blank) for sid in samples["sample_id"]]
    return [r[0] for r in res], [r[1] for r in res], [r[2] for r in res]


# ---------------------------------------------------------------- 表格特征（GBM 基线）

def _slope(t, v):
    ok = ~np.isnan(v)
    if ok.sum() < 2 or np.ptp(t[ok]) < 1e-6:
        return np.nan
    tt, vv = -t[ok], v[ok]  # 取负号：时间轴改为「越近越大」，斜率即每年的变化量
    tt = tt - tt.mean()
    return float((tt * (vv - vv.mean())).sum() / (tt ** 2).sum())


def tabular_features(samples: pd.DataFrame, seq: pd.DataFrame, mode: str, static_cols=tuple(STATIC_FEATURES),
                     drop_features=(), max_years=None, index_dx=True) -> pd.DataFrame:
    """
    mode="snapshot"  —— 起点快照：只用起点前 180 天内每个特征最近一次的测量值（横截面模型的常见做法）
    mode="aggregate" —— 历史聚合：每个特征的最近值/首值/均值/最小/最大/测量次数/年斜率/距最近测量的年数/总变化量
    缺失保留为 NaN，由 LightGBM 原生处理。
    index_dx=False 时不给指数诊断（与 drop_features 去掉各日诊断指示一起，用于"完全不用诊断信息"的消融）。
    """
    s = seq[seq["sample_id"].isin(samples["sample_id"])]
    if max_years is not None:
        s = s[s["days_before_index"] <= max_years * DAYS_PER_YEAR]
    rows = []
    for sid, g in s.groupby("sample_id", sort=False):
        t = g["days_before_index"].to_numpy(float) / DAYS_PER_YEAR
        feat = {"sample_id": sid}
        for c in VISIT_VALUE_FEATURES:
            if c in drop_features:
                continue
            v = g[c].to_numpy(float)
            ok = ~np.isnan(v)
            if mode == "snapshot":
                recent = ok & (t <= 0.5)
                feat[f"{c}_idx"] = v[recent][np.argmin(t[recent])] if recent.any() else np.nan
                continue
            if ok.any():
                order = np.argsort(t[ok])  # t 小 = 离起点近
                vo, to = v[ok][order], t[ok][order]
                feat.update({
                    f"{c}_last": vo[0], f"{c}_first": vo[-1], f"{c}_mean": vo.mean(),
                    f"{c}_min": vo.min(), f"{c}_max": vo.max(), f"{c}_n": ok.sum(),
                    f"{c}_slope": _slope(t, v), f"{c}_tsl": to[0], f"{c}_delta": vo[0] - vo[-1],
                })
            else:
                feat.update({f"{c}_{k}": np.nan for k in
                             ["last", "first", "mean", "min", "max", "slope", "tsl", "delta"]})
                feat[f"{c}_n"] = 0
        if mode == "aggregate":
            feat["n_steps"] = len(g)
            feat["span_years"] = t.max()
        rows.append(feat)
    f = pd.DataFrame(rows)
    base = samples[["sample_id"] + (["origin_mci"] if index_dx else []) + list(static_cols)]
    return base.merge(f, on="sample_id", how="left").set_index("sample_id").loc[samples["sample_id"]]
