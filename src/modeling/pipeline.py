"""
实验编排：把数据视图、模型、训练和预测串起来。scripts/ 与 notebooks/ 都只调用这里，不各写一套逻辑。
"""
import numpy as np
import pandas as pd

from src.dataset.build import SCALES
from src.dataset.cohort_v2 import DX_FEATURES, VISIT_VALUE_FEATURES
from src.modeling.data import (DAYS_PER_YEAR, annual_bin_arrays, sequence_arrays, static_matrix, tabular_features,
                               target_matrices)
from src.modeling.models import CTMT, GRUMultitask, LongForMADStyle
from src.modeling.train import TrainConfig, predict, predict_ensemble, train_ensemble

MODEL_CLASSES = {"CTMT": CTMT, "LongForMAD": LongForMADStyle, "GRU": GRUMultitask}
FEAT = {c: i for i, c in enumerate(VISIT_VALUE_FEATURES)}
COG_IDX = [FEAT[c] for c in SCALES]
DX_IDX = [FEAT[c] for c in DX_FEATURES]
MRI_IDX = [i for c, i in FEAT.items() if c not in SCALES and c not in DX_FEATURES]
CDR_MMSE_IDX = [FEAT["CDRSB"], FEAT["MMSE"]]  # ADNI 诊断分类所依据的两个量表（循环性敏感性分析）
GROUPS = {"MRI": MRI_IDX, "COG": COG_IDX, "DX": DX_IDX, "CDR_MMSE": CDR_MMSE_IDX}


def _drop_features(xs, ms, ts, idx):
    """把指定特征整体移除（置为缺失），并删掉因此变得完全为空的时间步。"""
    out = ([], [], [])
    for x, m, t in zip(xs, ms, ts):
        x, m = x.copy(), m.copy()
        x[:, idx], m[:, idx] = 0.0, 1.0
        keep = m.min(1) < 1
        if not keep.any():
            x, m, t = x[:1] * 0, np.ones_like(m[:1]), t[:1] * 0
        else:
            x, m, t = x[keep], m[keep], t[keep]
        out[0].append(x); out[1].append(m); out[2].append(t)
    return out


def _snapshot(xs, ms, ts, horizon_years=0.5):
    """把历史压成一个时间步：每个特征取起点前 horizon_years 年内最近一次的观测值（与 GBM-snapshot 同口径）。"""
    out = ([], [], [])
    for x, m, t in zip(xs, ms, ts):
        F = x.shape[1]
        xx, mm = np.zeros((1, F), np.float32), np.ones((1, F), np.float32)
        recent = t <= horizon_years
        for j in range(F):
            ok = np.flatnonzero(recent & (m[:, j] == 0))
            if len(ok):
                k = ok[np.argmin(t[ok])]
                xx[0, j], mm[0, j] = x[k, j], 0.0
        out[0].append(xx); out[1].append(mm); out[2].append(np.zeros(1, np.float32))
    return out


def build_view(samples, seq, norm, kind="sequence", window=None, drop=None):
    """
    kind="sequence" 按自然日的不规则时间步；"annual" 年度 token（LongForMAD 启发）；
    "snapshot" 单一时间步（每个特征在起点前半年内最近一次的值，用于「只看起点」的重新训练消融）。
    window 回看窗口（年）；drop 要移除的特征下标。
    """
    if kind == "annual":
        xs, ms, ts = annual_bin_arrays(samples, seq, norm)
    else:
        xs, ms, ts = sequence_arrays(samples, seq, norm, max_years=window)
    if drop:
        xs, ms, ts = _drop_features(xs, ms, ts, drop)
    if kind == "snapshot":
        xs, ms, ts = _snapshot(xs, ms, ts)
    yreg, yreg_miss = target_matrices(samples, norm)
    return {"xs": xs, "ms": ms, "ts": ts, "static": static_matrix(samples, norm),
            "y": samples["y"].to_numpy(), "yreg": yreg, "yreg_miss": yreg_miss,
            "sample_id": samples["sample_id"].to_numpy(), "subject": samples["subject_id"].to_numpy(),
            "origin_mci": samples["origin_mci"].to_numpy()}


def run_neural(model_name, cfg: TrainConfig, train, val, tests: dict, seeds, norm=None):
    """训练多种子集成。返回 {名称: (集成概率, 集成回归)}、平均验证损失、各种子模型。"""
    cls = MODEL_CLASSES[model_name]
    if model_name == "LongForMAD" and norm is not None:
        a = list(norm.static_cols).index("approx_age_years")
        cfg.model_kwargs.update(age_col=a, year_z=float(1.0 / norm.static_std[a]))
    n_feat, n_static = train["xs"][0].shape[1], train["static"].shape[1]
    models, val_loss = train_ensemble(cls, train, val, cfg, seeds, n_feat, n_static)
    preds = {name: predict_ensemble(models, v) for name, v in tests.items()}
    return preds, val_loss, models


def per_seed_predictions(models, view):
    return [predict(m, view)[0] for m in models]


def reg_to_raw(r, norm):
    return r * norm.target_std + norm.target_mean


def run_gbm(samples_tr, samples_va, samples_te_list, seq, mode, seeds=(0,), params=None, static_cols=None,
            drop_features=(), max_years=None, regression=True, index_dx=True):
    """
    LightGBM 基线。seeds 多于一个时做种子集成（bagging 随机性不同），与神经网络的多种子集成对等。
    lightgbm 与 torch 在 macOS 上各带一份 OpenMP，同进程加载会段错误；GBM 实验放在独立进程里跑。
    """
    from src.dataset.build import STATIC_FEATURES
    from src.modeling import gbm
    sc = tuple(static_cols or STATIC_FEATURES)
    kw = dict(mode=mode, static_cols=sc, drop_features=drop_features, max_years=max_years, index_dx=index_dx)
    ftr = tabular_features(samples_tr, seq, **kw)
    fva = tabular_features(samples_va, seq, **kw)
    ftes = [tabular_features(s, seq, **kw) for s in samples_te_list]
    tcols = [f"future_{s}" for s in SCALES]
    probs = [np.zeros((len(f), 3)) for f in ftes]
    regs = [np.zeros((len(f), len(SCALES))) for f in ftes]
    for sd in seeds:
        clf = gbm.fit_classifier(ftr, samples_tr["y"], fva, samples_va["y"], sd, params)
        for k, f in enumerate(ftes):
            probs[k] += clf.predict_proba(f) / len(seeds)
        if regression:
            rm = gbm.fit_regressors(ftr.to_numpy(), samples_tr[tcols].to_numpy(float),
                                    fva.to_numpy(), samples_va[tcols].to_numpy(float), sd, params)
            for k, f in enumerate(ftes):
                regs[k] += gbm.predict_regressors(rm, f.to_numpy()) / len(seeds)
    return [(p, r if regression else None) for p, r in zip(probs, regs)], ftr.shape[1]


def _last_obs(samples, seq):
    s = seq[seq["sample_id"].isin(samples["sample_id"])].sort_values(["sample_id", "time_step"])
    return s.groupby("sample_id")[SCALES].last().reindex(samples["sample_id"])  # groupby.last 跳过 NaN


def locf_regression(samples, seq, train_mean):
    """朴素回归基线「分数不变」：用起点前最近一次测得的该量表分数；从未测过则用训练集均值。"""
    last = _last_obs(samples, seq)
    return last.fillna(pd.Series(train_mean, index=SCALES)).to_numpy(float)


def drift_locf_regression(samples, seq, train_samples):
    """
    漂移校正的「分数不变」：最近一次测得的分数 + 训练集中同一起点诊断、同一量表的平均变化量
    （变化量 = 结局窗分数 − 起点前最近一次分数）。从未测过则用训练集同起点诊断的结局均值。
    """
    tr_last = _last_obs(train_samples, seq)
    out = []
    last = _last_obs(samples, seq)
    for j, sc in enumerate(SCALES):
        fut = train_samples[f"future_{sc}"].to_numpy(float)
        delta = pd.Series(fut - tr_last[sc].to_numpy(), index=train_samples.index)
        by_dx = delta.groupby(train_samples["origin_diagnosis"]).mean()
        mean_fut = pd.Series(fut, index=train_samples.index).groupby(train_samples["origin_diagnosis"]).mean()
        o = samples["origin_diagnosis"]
        pred = last[sc].to_numpy() + o.map(by_dx).to_numpy()
        pred = np.where(np.isnan(last[sc].to_numpy()), o.map(mean_fut).to_numpy(), pred)
        out.append(pred)
    return np.column_stack(out)


def pred_frame(model, samples, p, reg_raw=None, tag="test"):
    df = pd.DataFrame({"model": model, "eval_set": tag, "sample_id": samples["sample_id"].to_numpy(),
                       "p_CN": p[:, 0], "p_MCI": p[:, 1], "p_Dementia": p[:, 2]})
    if reg_raw is not None:
        for j, s in enumerate(SCALES):
            df[f"pred_{s}"] = reg_raw[:, j]
    return df
