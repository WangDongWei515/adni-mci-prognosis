"""
阶段 3/4 的神经网络模型。三个模型共享同一套输入约定（见 src/modeling/data.py），便于公平比较：

  CTMT            —— 本文主模型：连续时间多任务 Transformer（按自然日的不规则时间步 + [MISS] 嵌入 + 双头）
  LongForMADStyle —— 复现基线：按 Karaman & Sabuncu (2024) 的做法，年度 token + 可学习位置编码 + 平均池化，仅分类
  GRUMultitask    —— 循环网络基线：与 CTMT 相同的输入和双头，只把注意力换成 GRU

batch 字典约定：
  x (B,T,F) 特征值；m (B,T,F) 缺失标志；t (B,T) 距起点年数；gap (B,T) 距上一步年数；
  pad (B,T) True = 补齐位；static (B,S) 静态输入
"""
import math

import torch
import torch.nn as nn


class ContinuousTimeEncoding(nn.Module):
    """
    把「距起点多少年」和「距上一次就诊多少年」两个连续量编码成向量。
    用多尺度正弦/余弦（周期从 3 个月到 16 年）+ 对数项，让模型既能分辨「5 天前」和「5 年前」，
    又不需要把时间离散成「第几次就诊」。
    """

    def __init__(self, d_model: int, n_freq: int = 8):
        super().__init__()
        periods = torch.logspace(math.log10(0.25), math.log10(16.0), n_freq)
        self.register_buffer("omega", 2 * math.pi / periods)
        self.proj = nn.Linear(2 * n_freq + 3, d_model)

    def forward(self, t, gap):
        ang = t.unsqueeze(-1) * self.omega
        feats = torch.cat([torch.sin(ang), torch.cos(ang), t.unsqueeze(-1),
                           torch.log1p(t).unsqueeze(-1), torch.log1p(gap).unsqueeze(-1)], dim=-1)
        return self.proj(feats)


class CTMT(nn.Module):
    """
    Continuous-Time Multitask Transformer。

    每个时间步的嵌入 = 值嵌入 + 时间嵌入。值嵌入对拼接的 [x·(1−m), m] 做线性映射：
    对第 j 个特征，观测到时贡献 x_j·w_j，缺失时贡献一个可学习的 [MISS]_j 向量——两种写法在数学上等价。
    序列前面拼接 [CLS] 和静态 token，[CLS] 的输出同时接分类头（头 A）和回归头（头 B）。
    """

    def __init__(self, n_feat, n_static, d_model=64, n_heads=4, n_layers=2, dropout=0.2,
                 n_classes=3, n_targets=4, use_time=True, max_len=64):
        super().__init__()
        self.use_time = use_time
        self.value = nn.Linear(2 * n_feat, d_model)
        if use_time:
            self.time = ContinuousTimeEncoding(d_model)
        else:
            # 消融：去掉连续时间，只告诉模型「这是倒数第几次就诊」
            self.pos = nn.Embedding(max_len, d_model)
        self.static = nn.Sequential(nn.Linear(n_static, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_emb = nn.Embedding(3, d_model)  # 0=CLS 1=静态 2=就诊
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head_cls = nn.Linear(d_model, n_classes)
        self.head_reg = nn.Linear(d_model, n_targets)

    def forward(self, b):
        x, m, pad = b["x"], b["m"], b["pad"]
        B, T, _ = x.shape
        h = self.value(torch.cat([x * (1 - m), m], dim=-1))
        if self.use_time:
            h = h + self.time(b["t"], b["gap"])
        else:
            h = h + self.pos(b["rev_idx"])
        h = h + self.type_emb.weight[2]
        s = (self.static(b["static"]) + self.type_emb.weight[1]).unsqueeze(1)
        c = (self.cls + self.type_emb.weight[0]).expand(B, 1, -1)
        z = torch.cat([c, s, h], dim=1)
        key_pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=x.device), pad], dim=1)
        z = self.encoder(z, src_key_padding_mask=key_pad)
        out = self.norm(z[:, 0])
        return {"logits": self.head_cls(out), "reg": self.head_reg(out)}


class LongForMADStyle(nn.Module):
    """
    LongForMAD 的复现（Karaman & Sabuncu, 2024），按其论文描述实现：
      - 每个年度一个 token：[特征均值填补后的值, 缺失掩码, 静态信息] → 线性投影
      - 可学习位置编码（零初始化，按年度序号）+ 线性年龄编码（仿 BEHRT）
      - Transformer 编码器 → 对有效 token 平均池化 → 全连接 + softmax
    本文预测跨度固定为 24 个月，因此原文中拼接在 token 上的跨度 Δt 为常数，省略。
    与原文的其他差异（输入变量、无 CSF/APOE、单一跨度）见补充材料的偏差对照表，因此称为「LongForMAD 启发的」模型。
    """

    def __init__(self, n_feat, n_static, d_model=64, n_heads=4, n_layers=2, dropout=0.2,
                 n_classes=3, n_years=4, age_col=1, year_z=0.14):
        super().__init__()
        # age_col：静态矩阵中标准化年龄所在列；year_z：1 年对应的标准化年龄单位（= 1 / 训练集年龄标准差）
        self.age_col = age_col
        self.year_z = year_z
        self.proj = nn.Linear(2 * n_feat + n_static, d_model)
        self.pos = nn.Parameter(torch.zeros(n_years, d_model))
        self.age = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout, batch_first=True,
                                           norm_first=True, activation="relu")
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, n_classes))

    def forward(self, b):
        x, m, pad, st = b["x"], b["m"], b["pad"], b["static"]
        B, T, _ = x.shape
        k = b["t"].long().clamp(0, self.pos.shape[0] - 1)
        # 年龄编码（仿 BEHRT）：每个年度 token 的年龄 = 起点年龄 − 相隔年数，在标准化年龄尺度上计算
        age = st[:, self.age_col:self.age_col + 1].unsqueeze(1) - self.year_z * b["t"].unsqueeze(-1)
        h = self.proj(torch.cat([x * (1 - m), m, st.unsqueeze(1).expand(B, T, -1)], dim=-1))
        h = h + self.pos[k] + self.age(age)
        z = self.encoder(h, src_key_padding_mask=pad)
        valid = (~pad).unsqueeze(-1).float()
        pooled = (z * valid).sum(1) / valid.sum(1).clamp(min=1)
        return {"logits": self.head(pooled)}


class GRUMultitask(nn.Module):
    """与 CTMT 相同输入和双头的循环网络基线：按时间顺序读入就诊，取最后隐状态。"""

    def __init__(self, n_feat, n_static, d_model=64, dropout=0.2, n_classes=3, n_targets=4, **_):
        super().__init__()
        self.inp = nn.Linear(2 * n_feat + 2, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.static = nn.Linear(n_static, d_model)
        self.drop = nn.Dropout(dropout)
        self.head_cls = nn.Linear(2 * d_model, n_classes)
        self.head_reg = nn.Linear(2 * d_model, n_targets)

    def forward(self, b):
        x, m, pad = b["x"], b["m"], b["pad"]
        inp = torch.cat([x * (1 - m), m, b["t"].unsqueeze(-1), torch.log1p(b["gap"]).unsqueeze(-1)], dim=-1)
        h, _ = self.gru(torch.relu(self.inp(inp)))
        lengths = (~pad).sum(1).clamp(min=1)
        last = h[torch.arange(len(h)), lengths - 1]
        out = self.drop(torch.cat([last, torch.relu(self.static(b["static"]))], dim=-1))
        return {"logits": self.head_cls(out), "reg": self.head_reg(out)}
