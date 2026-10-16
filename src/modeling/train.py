"""
神经网络模型共用的训练循环：批处理补齐、训练时随机丢弃就诊（数据增强）、早停、多种子集成。
"""
import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 12
    step_dropout: float = 0.2   # 训练时每个历史时间步被随机丢弃的概率（至少保留一步）
    reg_weight: float = 0.5     # 头 B 回归损失的权重；0 = 只训练分类头
    model_kwargs: dict = field(default_factory=dict)


def collate(idx, xs, ms, ts, static, y, yreg, yreg_miss, step_dropout=0.0, rng=None):
    """把若干条变长序列补齐成张量；训练时可按 step_dropout 随机删掉部分时间步。"""
    seqs = []
    for i in idx:
        x, m, t = xs[i], ms[i], ts[i]
        if step_dropout > 0 and len(t) > 1:
            keep = rng.random(len(t)) > step_dropout
            if not keep.any():
                keep[rng.integers(len(t))] = True
            x, m, t = x[keep], m[keep], t[keep]
        seqs.append((x, m, t))
    T = max(len(s[2]) for s in seqs)
    B, Fd = len(idx), xs[0].shape[1]
    X = np.zeros((B, T, Fd), np.float32)
    M = np.ones((B, T, Fd), np.float32)
    Tt = np.zeros((B, T), np.float32)
    G = np.zeros((B, T), np.float32)
    P = np.ones((B, T), bool)
    R = np.zeros((B, T), np.int64)
    for j, (x, m, t) in enumerate(seqs):
        L = len(t)
        X[j, :L], M[j, :L], Tt[j, :L], P[j, :L] = x, m, t, False
        if L > 1:
            G[j, 1:L] = np.abs(t[:-1] - t[1:])
        R[j, :L] = np.arange(L)[::-1]  # 倒数第几步（0 = 最近一步），供「无时间编码」消融使用
    b = {"x": X, "m": M, "t": Tt, "gap": G, "pad": P, "rev_idx": R, "static": static[idx]}
    b = {k: torch.from_numpy(v) for k, v in b.items()}
    b["y"] = torch.from_numpy(y[idx]).long()
    b["yreg"] = torch.from_numpy(yreg[idx])
    b["yreg_miss"] = torch.from_numpy(yreg_miss[idx])
    return b


def loss_fn(out, b, reg_weight):
    loss = F.cross_entropy(out["logits"], b["y"])
    if reg_weight > 0 and "reg" in out:
        w = 1 - b["yreg_miss"]
        # Huber 损失对偶发的极端分数更稳健；只在有标签的量表上计算
        per = F.smooth_l1_loss(out["reg"], b["yreg"], reduction="none") * w
        loss = loss + reg_weight * per.sum() / w.sum().clamp(min=1)
    return loss


@torch.no_grad()
def predict(model, data, batch_size=256):
    model.eval()
    n = len(data["y"])
    probs, regs = [], []
    for a in range(0, n, batch_size):
        idx = np.arange(a, min(a + batch_size, n))
        b = collate(idx, *data_args(data))
        out = model(b)
        probs.append(torch.softmax(out["logits"], -1).numpy())
        if "reg" in out:
            regs.append(out["reg"].numpy())
    return np.concatenate(probs), (np.concatenate(regs) if regs else None)


def data_args(d):
    return d["xs"], d["ms"], d["ts"], d["static"], d["y"], d["yreg"], d["yreg_miss"]


def train_one(model_cls, train, val, cfg: TrainConfig, seed: int, n_feat: int, n_static: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model_cls(n_feat=n_feat, n_static=n_static, **cfg.model_kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = len(train["y"])
    best, best_state, bad = np.inf, None, 0
    history = []
    for ep in range(cfg.max_epochs):
        model.train()
        perm = rng.permutation(n)
        for a in range(0, n, cfg.batch_size):
            b = collate(perm[a:a + cfg.batch_size], *data_args(train), step_dropout=cfg.step_dropout, rng=rng)
            loss = loss_fn(model(b), b, cfg.reg_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # 早停只看验证集的分类交叉熵：头 A 是主结局
        p, _ = predict(model, val)
        vl = float(-np.log(np.clip(p[np.arange(len(p)), val["y"]], 1e-8, None)).mean())
        history.append(vl)
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, best, history


def train_ensemble(model_cls, train, val, cfg, seeds, n_feat, n_static):
    models, vals = [], []
    for s in seeds:
        mdl, vl, _ = train_one(model_cls, train, val, cfg, s, n_feat, n_static)
        models.append(mdl)
        vals.append(vl)
    return models, float(np.mean(vals))


def predict_ensemble(models, data):
    ps, rs = zip(*(predict(m, data) for m in models))
    p = np.mean(ps, axis=0)
    r = np.mean(rs, axis=0) if rs[0] is not None else None
    return p, r
