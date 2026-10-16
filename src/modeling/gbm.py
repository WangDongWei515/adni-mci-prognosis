"""
阶段 3 的表格基线：LightGBM（梯度提升树）。
  - 起点快照 + GBM：代表「只看一次检查」的横截面做法
  - 历史聚合 + GBM：把历史「拍平」成统计量（均值、斜率、最近值……），是时序模型必须超越的强基线
缺失值保留为 NaN，由 LightGBM 在分裂时自行决定缺失样本走向哪一侧。

调参预算与神经网络对等（第 1 轮审稿意见 R2-M3）：同样 8 组配置，在验证集上按多分类对数损失选择，
最终用多个随机种子（bagging 随机性不同）集成。回归用 L1 目标，与评估指标 MAE 一致（R2-M7）。
"""
import lightgbm as lgb
import numpy as np

BASE = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_estimators=2000, verbose=-1)
GRID = [dict(learning_rate=lr, num_leaves=nl, min_child_samples=mc)
        for lr in (0.03, 0.1) for nl in (7, 15) for mc in (10, 40)]
DEFAULT = dict(learning_rate=0.03, num_leaves=15, min_child_samples=20)


def fit_classifier(Xtr, ytr, Xva, yva, seed=0, params=None):
    clf = lgb.LGBMClassifier(objective="multiclass", random_state=seed, **BASE, **(params or DEFAULT))
    clf.fit(Xtr, ytr, eval_X=(Xva,), eval_y=(yva,), callbacks=[lgb.early_stopping(100, verbose=False)])
    return clf


def val_logloss(clf, Xva, yva):
    p = np.clip(clf.predict_proba(Xva)[np.arange(len(yva)), np.asarray(yva)], 1e-8, None)
    return float(-np.log(p).mean())


def fit_regressors(Xtr, Ytr, Xva, Yva, seed=0, params=None):
    """每个量表单独一个回归器，只用有该量表标签的样本训练；L1 目标（中位数型预测，对应 MAE）。"""
    models = []
    for j in range(Ytr.shape[1]):
        ok_tr, ok_va = ~np.isnan(Ytr[:, j]), ~np.isnan(Yva[:, j])
        reg = lgb.LGBMRegressor(objective="l1", random_state=seed, **BASE, **(params or DEFAULT))
        reg.fit(Xtr[ok_tr], Ytr[ok_tr, j], eval_X=(Xva[ok_va],), eval_y=(Yva[ok_va, j],),
                callbacks=[lgb.early_stopping(100, verbose=False)])
        models.append(reg)
    return models


def predict_regressors(models, X):
    return np.column_stack([m.predict(X) for m in models])
