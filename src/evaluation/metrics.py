"""
阶段 5 评估指标。

头 A（未来诊断三分类）按临床问题拆成三个二分类 AUROC，另给整体的宏平均 AUROC：
  mci_conv  —— MCI 起点：会不会进展为痴呆（正类 = 痴呆，打分 = P(痴呆)）
  mci_rev   —— MCI 起点：会不会逆转回 CN（正类 = CN，打分 = P(CN)）
  cn_prog   —— CN 起点：会不会进展（正类 = MCI 或痴呆，打分 = 1 − P(CN)）

置信区间：按「患者」重抽样的 bootstrap（cluster bootstrap）。同一患者的多条样本高度相关，
按样本重抽样会低估不确定性，所以整名患者一起抽。两个模型的差值用同一组重抽样计算（配对）。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)

KEY_METRICS = ["macro_auroc", "mci_conv", "mci_rev", "cn_prog", "mci_conv_auprc", "bal_acc", "macro_f1", "brier"]


def _safe_auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def classification_metrics(y, p, origin_mci):
    y = np.asarray(y); p = np.asarray(p); om = np.asarray(origin_mci).astype(bool)
    out = {}
    try:
        out["macro_auroc"] = roc_auc_score(y, p, multi_class="ovr", average="macro", labels=[0, 1, 2])
    except ValueError:
        out["macro_auroc"] = np.nan
    out["mci_conv"] = _safe_auc((y[om] == 2).astype(int), p[om, 2])
    out["mci_rev"] = _safe_auc((y[om] == 0).astype(int), p[om, 0])
    out["cn_prog"] = _safe_auc((y[~om] != 0).astype(int), 1 - p[~om, 0])
    yy = (y[om] == 2).astype(int)
    out["mci_conv_auprc"] = average_precision_score(yy, p[om, 2]) if 0 < yy.sum() < len(yy) else np.nan
    pred = p.argmax(1)
    out["bal_acc"] = balanced_accuracy_score(y, pred)
    out["macro_f1"] = f1_score(y, pred, average="macro", labels=[0, 1, 2])
    onehot = np.eye(3)[y]
    out["brier"] = float(((p - onehot) ** 2).sum(1).mean())
    return out


def regression_metrics(y, yhat):
    """y / yhat 为原始单位；y 中的 NaN 表示无标签。"""
    out = {}
    for j in range(y.shape[1]):
        ok = ~np.isnan(y[:, j])
        e = yhat[ok, j] - y[ok, j]
        out[f"mae_{j}"] = float(np.abs(e).mean()) if ok.any() else np.nan
        out[f"rmse_{j}"] = float(np.sqrt((e ** 2).mean())) if ok.any() else np.nan
    return out


def cluster_bootstrap(fn, subjects, n_boot=2000, seed=0):
    """
    fn(idx) -> dict[str, float]，idx 为样本下标。按患者重抽样 n_boot 次，返回每次的指标表。
    """
    rng = np.random.default_rng(seed)
    subjects = np.asarray(subjects)
    uniq, inv = np.unique(subjects, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rows = []
    for _ in range(n_boot):
        pick = rng.integers(len(uniq), size=len(uniq))
        idx = np.concatenate([groups[k] for k in pick])
        rows.append(fn(idx))
    return pd.DataFrame(rows)


def summarize(point: dict, boot: pd.DataFrame) -> pd.DataFrame:
    lo = boot.quantile(0.025)
    hi = boot.quantile(0.975)
    return pd.DataFrame({"estimate": pd.Series(point), "ci_low": lo, "ci_high": hi})


def evaluate_classifier(y, p, origin_mci, subjects, n_boot=2000, seed=0):
    y, p, om = np.asarray(y), np.asarray(p), np.asarray(origin_mci)
    point = classification_metrics(y, p, om)
    boot = cluster_bootstrap(lambda i: classification_metrics(y[i], p[i], om[i]), subjects, n_boot, seed)
    return summarize(point, boot), boot


def paired_difference(boot_a: pd.DataFrame, boot_b: pd.DataFrame, point_a: dict, point_b: dict):
    """A − B 的点估计、95% CI 与双侧 bootstrap p 值（两组 boot 必须来自同一组重抽样种子）。"""
    rows = {}
    for k in boot_a.columns:
        d = (boot_a[k] - boot_b[k]).dropna()
        if len(d) == 0:
            continue
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        rows[k] = {"diff": point_a[k] - point_b[k], "ci_low": d.quantile(0.025),
                   "ci_high": d.quantile(0.975), "p": min(1.0, p)}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- 第 2 版新增（第 1 轮审稿意见）

TASKS = ("mci_conv", "mci_rev", "cn_prog")


def task_arrays(y, p, origin_mci, task):
    """把三分类输出转成某个临床任务的二分类（结局, 打分, 样本掩码）。"""
    y, p, om = np.asarray(y), np.asarray(p), np.asarray(origin_mci).astype(bool)
    if task == "mci_conv":
        return (y[om] == 2).astype(int), p[om, 2], om
    if task == "mci_rev":
        return (y[om] == 0).astype(int), p[om, 0], om
    return (y[~om] != 0).astype(int), 1 - p[~om, 0], ~om


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _newton_logistic(X, y, offset=None, iters=100):
    """带步长减半的 Newton–Raphson；预测概率极端（如接近 0）时，纯 Newton 会发散。"""
    beta = np.zeros(X.shape[1])
    off = np.zeros(len(y)) if offset is None else offset

    def loglik(b):
        eta = X @ b + off
        return (y * eta - np.logaddexp(0, eta)).sum()

    ll = loglik(beta)
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ beta + off)))
        w = mu * (1 - mu) + 1e-9
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        t = 1.0
        while t > 1e-6 and loglik(beta + t * step) < ll - 1e-12:
            t /= 2
        beta = beta + t * step
        new_ll = loglik(beta)
        if abs(new_ll - ll) < 1e-10 or np.abs(t * step).max() < 1e-8:
            break
        ll = new_ll
    return beta


def calibration_stats(yb, s):
    """
    calibration-in-the-large（截距，以 logit(p) 为 offset）与校准斜率（y ~ a + b·logit(p)）。
    理想值：截距 0、斜率 1。另给观察 / 预测的平均概率。
    """
    yb, s = np.asarray(yb, float), np.asarray(s, float)
    lp = _logit(s)
    ones = np.ones((len(yb), 1))
    intercept = _newton_logistic(ones, yb, offset=lp)[0]
    slope = _newton_logistic(np.column_stack([ones, lp]), yb)[1]
    return {"cal_intercept": intercept, "cal_slope": slope, "mean_pred": s.mean(), "obs_rate": yb.mean()}


def task_metrics(y, p, origin_mci):
    """每个临床任务的 AUROC、AUPRC、Brier、缩放 Brier（1 − Brier / 仅用事件率时的 Brier）。"""
    out = {}
    for t in TASKS:
        yb, s, _ = task_arrays(y, p, origin_mci, t)
        ok = 0 < yb.sum() < len(yb)
        out[t] = roc_auc_score(yb, s) if ok else np.nan
        out[f"{t}_auprc"] = average_precision_score(yb, s) if ok else np.nan
        b = float(((s - yb) ** 2).mean())
        b0 = float(yb.mean() * (1 - yb.mean()))
        out[f"{t}_brier"] = b
        out[f"{t}_scaled_brier"] = 1 - b / b0 if b0 > 0 else np.nan
    return out


def weighted_auc(yb, s, w):
    return roc_auc_score(yb, s, sample_weight=w) if 0 < yb.sum() < len(yb) else np.nan


def corrected_resampled_t(diffs, k=5, test_train_ratio=0.2 / (0.8 * 6 / 7)):
    """
    重复 k 折交叉验证的校正 t 检验（Nadeau & Bengio 2003；Bouckaert & Frank 2004）：
    方差乘以 (1/(k·r) + n_test/n_train)，以计入不同折训练集重叠造成的相关。
    n_train 只算真正用于拟合的部分：训练折（4/5）再留出 1/7 做早停，故比值为 0.2 / (0.8 × 6/7) ≈ 0.29（返修前误用 0.25）。
    diffs：每一折（每次重复 × 每折）上两个模型的指标差。返回均值、95% CI、双侧 P。
    """
    from scipy import stats
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    n = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1 / n + test_train_ratio) * var)
    t = d.mean() / se if se > 0 else np.inf
    q = stats.t.ppf(0.975, n - 1)
    p = 2 * stats.t.sf(abs(t), n - 1)
    return {"diff": d.mean(), "ci_low": d.mean() - q * se, "ci_high": d.mean() + q * se, "p": p, "n_folds": n}


def holm(pvals):
    """Holm–Bonferroni 校正后的 P 值（保持输入顺序）。"""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj
