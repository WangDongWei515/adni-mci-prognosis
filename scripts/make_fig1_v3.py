"""
Fig. 1 第 3 版（按 docs/fig1_redraw_guide.md 重画）：四个面板，宽度与其他正文图一致（5.1 in，1:1 排入版心）。
  a 队列流程（CONSORT 竖排，数字为主角）
  b 一个预测点的任务定义（时间轴：左侧输入历史、右侧 21–27 个月结局窗；缺测模态留空）
  c 比较的七种方法（模型卡片，颜色与图 2–5 一致）+ 评估协议
  d 剔除逆转者实验（2×2：训练队列 × 评估队列）
数字取自 results/tables/flow_counts.json 与正文结果；画法只用 matplotlib，输出 fig1_overview.{pdf,png,tiff}。
用法：.venv/bin/python scripts/make_fig1_v3.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper/figures"
W, H = 5.1, 5.85

INK, INK2, MUTED, LINE = "#1f2328", "#52514e", "#8a8983", "#c9c8c1"
HIST, HIST_BG = "#3f6690", "#edf2f8"          # 输入历史：石板蓝
OUTC, OUTC_BG = "#7d4e9e", "#f1eaf6"          # 结局：梅紫（与模型五色区分）
COLOR = {"CTMT": "#2a78d6", "GRU": "#eb6834", "LongForMAD": "#1baf7a", "GBM-aggregate": "#eda100",
         "GBM-snapshot": "#e87ba4", "Logistic": "#8a8983", "Index-dx-only": "#b9b8b1"}

plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "text.color": INK, "hatch.color": "#dcdbd4",
                     "hatch.linewidth": 0.5})


def rbox(ax, x, y, w, h, fc="white", ec=LINE, lw=0.6, r=0.012, z=1, ls="-"):
    asp = ax.bbox.height / ax.bbox.width
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=1 / asp,
                                fc=fc, ec=ec, lw=lw, zorder=z, ls=ls, transform=ax.transAxes))


def t(ax, x, y, s, size=6.5, color=INK, weight="normal", ha="left", va="center", maxw=None, **kw):
    """写字；给了 maxw（轴坐标宽度）时自动缩小字号直到放得下，最小 5 pt。"""
    obj = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, transform=ax.transAxes, **kw)
    if maxw is not None:
        r = ax.figure.canvas.get_renderer()
        while obj.get_window_extent(r).width / ax.bbox.width > maxw and size > 6.0:
            size -= 0.2
            obj.set_fontsize(size)
        if obj.get_window_extent(r).width / ax.bbox.width > maxw + 1e-3:
            print("OVERFLOW:", repr(s), round(obj.get_window_extent(r).width / ax.bbox.width, 3), ">", maxw)
    return obj


def width(ax, obj):
    return obj.get_window_extent(ax.figure.canvas.get_renderer()).width / ax.bbox.width


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=0.7, ms=6):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0))


def panel(ax, letter, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    t(ax, -0.02, 1.0, letter, size=9.5, weight="bold", va="bottom")
    t(ax, 0.06, 1.0, title, size=7, weight="bold", va="bottom")


# ---------------------------------------------------------------- a 队列流程

def panel_a(ax, flow):
    panel(ax, "a", "Cohort")
    mx, mw = 0.0, 0.45          # 主列
    ex, ew = 0.49, 0.51         # 排除列
    cx = mx + mw / 2

    def main(y, h, head, num, sub, hl=False):
        rbox(ax, mx, y, mw, h, fc=HIST_BG if hl else "white", ec=HIST if hl else INK2, lw=0.9 if hl else 0.6)
        t(ax, cx, y + h * 0.80, head, size=6, color=INK2, ha="center", maxw=mw - 0.04)
        t(ax, cx, y + h * 0.48, num, size=9.5, weight="bold", color=HIST if hl else INK, ha="center")
        t(ax, cx, y + h * 0.17, sub, size=6, color=INK2, ha="center", maxw=mw - 0.04)

    def excl(y, h, lines):
        rbox(ax, ex, y, ew, h, fc="#f6f6f3", ec=LINE)
        for k, (lab, n) in enumerate(lines):
            yy = y + h - (k + 0.5) * h / len(lines)
            t(ax, ex + 0.03, yy + 0.018, lab, size=6, color=INK2, maxw=ew - 0.10)
            t(ax, ex + 0.03, yy - 0.021, n, size=6.4, weight="bold", color=INK2)

    main(0.83, 0.135, "ADNI participants", f"{flow['adni_participants']:,}", "with a diagnosis")
    excl(0.86, 0.075, [("Never CN or MCI", f"{flow['no_cn_mci']:,}")])
    ax.plot([cx, ex], [0.8975, 0.8975], color=LINE, lw=0.6, transform=ax.transAxes, zorder=0)
    arrow(ax, cx, 0.83, cx, 0.735)

    main(0.60, 0.135, "Candidate points", f"{flow['candidates']:,}", f"{flow['candidate_participants']:,} participants")
    excl(0.40, 0.235, [("Follow-up < 21 mo", f"{flow['short']:,}"), ("No dx at 21–27 mo", f"{flow['no_window']:,}"),
                       ("Outcome visit shared", f"{flow['shared']:,}")])
    ax.plot([cx, ex], [0.555, 0.555], color=LINE, lw=0.6, transform=ax.transAxes, zorder=0)
    arrow(ax, cx, 0.60, cx, 0.505)

    main(0.37, 0.135, "Analysed points", f"{flow['analysed']:,}", f"{flow['analysed_participants']:,} participants", hl=True)

    # 按指数诊断拆分，并给出 21–27 个月时的结局
    for k, (lab, n, outs) in enumerate([("CN index", "1,720", [("CN", "1,554", INK2), ("MCI", "154", INK2), ("Dementia", "12", INK2)]),
                                        ("MCI index", "2,103", [("CN", "101", OUTC), ("MCI", "1,567", INK2), ("Dementia", "435", INK2)])]):
        x = k * 0.52
        arrow(ax, cx, 0.37, x + 0.24, 0.305, color=MUTED, lw=0.6)
        rbox(ax, x, 0.02, 0.48, 0.28, fc="white", ec=LINE)
        t(ax, x + 0.24, 0.258, lab, size=6, color=INK2, ha="center")
        t(ax, x + 0.24, 0.212, n, size=8, weight="bold", ha="center")
        t(ax, x + 0.24, 0.165, "at 21–27 months", size=6, color=MUTED, ha="center", maxw=0.44)
        for j, (o, c, col) in enumerate(outs):
            yy = 0.12 - j * 0.036
            bold = "bold" if col == OUTC else "normal"
            t(ax, x + 0.05, yy, o, size=6, color=col, weight=bold)
            t(ax, x + 0.43, yy, c, size=6, color=col, ha="right", weight=bold)
    t(ax, 0.76, -0.025, "reversions kept", size=6, color=OUTC, ha="center", style="italic")


# ---------------------------------------------------------------- b 任务定义

def panel_b(ax):
    panel(ax, "b", "Prediction task for one prediction point")
    d0, d1 = -1500, 900
    X = lambda d: 0.135 + (d - d0) / (d1 - d0) * 0.80  # noqa: E731
    y_ax, y_c, y_m, y_dx = 0.50, 0.655, 0.565, 0.755
    top = 0.88
    ax.add_patch(Rectangle((X(d0), y_ax), X(0) - X(d0), top - y_ax, fc=HIST_BG, ec="none", transform=ax.transAxes, zorder=0))
    ax.add_patch(Rectangle((X(0), y_ax), X(640) - X(0), top - y_ax, fc="#f7f7f4", ec="none", transform=ax.transAxes, zorder=0,
                           hatch="////", lw=0))
    ax.add_patch(Rectangle((X(640), y_ax), X(820) - X(640), top - y_ax, fc=OUTC_BG, ec="none", transform=ax.transAxes, zorder=0))
    t(ax, X(d0) + 0.012, top - 0.04, "Input: all history to the index date", size=6.2, color=HIST, weight="bold",
      maxw=X(0) - X(d0) - 0.06)
    t(ax, (X(0) + X(640)) / 2, top - 0.04, "not used", size=6, color=MUTED, ha="center", style="italic",
      bbox=dict(fc="#f7f7f4", ec="none", pad=0.6))
    t(ax, (X(640) + X(820)) / 2, top + 0.035, "outcome", size=6, color=OUTC, ha="center", weight="bold")
    for y, lab in [(y_dx, "Diagnosis"), (y_c, "Cognition"), (y_m, "MRI")]:
        t(ax, X(d0) - 0.012, y, lab, size=6, color=INK2, ha="right")
    # 评估日：不规则；同一次访视的认知测试与 MRI 可在不同日，不合并
    days = [(-1390, "CN", 1, 1), (-1030, "CN", 1, 0), (-845, "MCI", 1, 0), (-700, None, 0, 1), (-345, "MCI", 1, 0), (0, "MCI", 1, 1)]
    s = 0.024
    w = s * (ax.bbox.height / ax.bbox.width) * 1.6
    for d, dx, c, m in days:
        x = X(d)
        ax.plot([x, x], [y_ax, y_ax + 0.018], color=INK2, lw=0.6, transform=ax.transAxes)
        for y, on in [(y_c, c), (y_m, m)]:
            ax.add_patch(Rectangle((x - w / 2, y - s), w, 2 * s, fc=HIST if on else "white", ec=HIST if on else MUTED, lw=0.5,
                                   ls="-" if on else (0, (1.5, 1)), transform=ax.transAxes, zorder=3))
        if dx:
            t(ax, x, y_dx, dx, size=6, weight="bold" if d == 0 else "normal", ha="center", color=INK if d == 0 else INK2,
              zorder=5, bbox=dict(fc=HIST_BG, ec="none", pad=0.8) if d == 0 else None)
    ax.plot([X(0), X(0)], [y_ax - 0.02, top + 0.02], color=INK, lw=0.9, ls=(0, (3, 2)), transform=ax.transAxes, zorder=4)
    t(ax, X(0), top + 0.035, "index date", size=6, weight="bold", ha="center")
    ax.plot([X(730)], [y_dx], marker="D", ms=4.6, color=OUTC, transform=ax.transAxes, zorder=4)
    ax.annotate("", xy=(X(d1) + 0.01, y_ax), xytext=(X(d0), y_ax), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0))
    for yr in range(-4, 3):
        x = X(yr * 365.25)
        ax.plot([x, x], [y_ax - 0.015, y_ax], color=INK2, lw=0.5, transform=ax.transAxes)
        t(ax, x, y_ax - 0.05, f"{yr:+d}".replace("-", "−") if yr else "0", size=6, color=INK2, ha="center")
    t(ax, X(d1) + 0.025, y_ax - 0.05, "years", size=6, color=INK2)
    ax.annotate("", xy=(X(820), y_ax - 0.105), xytext=(X(640), y_ax - 0.105), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="|-|", color=OUTC, lw=0.6, mutation_scale=2.5, shrinkA=0, shrinkB=0))
    t(ax, (X(640) + X(820)) / 2, y_ax - 0.15, "21–27 months", size=6, color=OUTC, ha="center")
    # 图例
    lx, ly = X(d0), 0.30
    ax.add_patch(Rectangle((lx, ly - s), w, 2 * s, fc=HIST, ec=HIST, lw=0.5, transform=ax.transAxes))
    o = t(ax, lx + w + 0.012, ly, "measured", size=6, color=INK2)
    x2 = lx + w + 0.012 + width(ax, o) + 0.04
    ax.add_patch(Rectangle((x2, ly - s), w, 2 * s, fc="white", ec=MUTED, lw=0.5, ls=(0, (1.5, 1)), transform=ax.transAxes))
    t(ax, x2 + w + 0.012, ly, "not measured (flagged as missing)", size=6, color=INK2)
    # 一个三分类输出（CN / MCI / 痴呆）覆盖三种转移
    t(ax, lx, 0.17, "One three-class output (CN, MCI or dementia) covers all transitions:", size=6, color=INK2)
    cx = lx
    for lab, col in [("MCI → dementia", INK2), ("MCI → CN", OUTC), ("CN → MCI/dementia", INK2)]:
        a = ax.text(cx + 0.014, 0.08, lab, fontsize=6, color=col, fontweight="bold", va="center", transform=ax.transAxes,
                    zorder=3)
        wbox = width(ax, a) + 0.045
        rbox(ax, cx, 0.042, wbox, 0.076, fc="white", ec=col, lw=0.6, z=2)
        cx += wbox + 0.014


# ---------------------------------------------------------------- c 方法与评估

MODELS = [("Reference", [("Index-dx-only", "Index diagnosis only", "outcome rates per index diagnosis", "A"),
                         ("Logistic", "Logistic regression", "index diagnosis, MMSE, CDR-SB, age", "A")]),
          ("Gradient-boosted models", [("GBM-snapshot", "GBM-snapshot", "latest value of each input", "A+B"),
                                       ("GBM-aggregate", "GBM-aggregate", "150 summaries of the whole history", "A+B")]),
          ("Neural sequence models", [("LongForMAD", "LongForMAD-inspired", "annual tokens, index year and 3 before", "A"),
                                      ("GRU", "GRU", "assessment days in order", "A+B"),
                                      ("CTMT", "CTMT", "assessment days in continuous time", "A+B")])]


def panel_c(ax):
    panel(ax, "c", "Approaches compared")
    cw, ch = 0.82, 0.071
    y = 0.965
    learned = []
    for group, items in MODELS:
        y -= 0.038
        t(ax, 0.0, y, group, size=6, color=INK2, weight="bold")
        y -= 0.008
        for key, name, desc, head in items:
            y -= ch + 0.01
            rbox(ax, 0.0, y, cw, ch, fc="white", ec=LINE)
            ax.add_patch(Rectangle((0.0, y + 0.002), 0.02, ch - 0.004, fc=COLOR[key], ec="none", transform=ax.transAxes, zorder=2))
            t(ax, 0.04, y + ch * 0.68, name, size=6.2, weight="bold")
            t(ax, 0.04, y + ch * 0.28, desc, size=6, color=INK2, maxw=cw - 0.16)
            t(ax, cw - 0.025, y + ch / 2, head, size=6, weight="bold", ha="right", color=INK if head == "A+B" else MUTED)
            if key not in ("Index-dx-only", "Logistic"):
                learned.append((y, y + ch))
    y0, y1 = learned[-1][0], learned[0][1]
    bx = cw + 0.035
    ax.plot([bx - 0.015, bx, bx, bx - 0.015], [y1, y1, y0, y0], color=INK2, lw=0.7, transform=ax.transAxes)
    t(ax, bx + 0.03, (y0 + y1) / 2, "equal tuning budget", size=6, color=INK2, rotation=270, ha="center")
    t(ax, cw, 0.975, "A, diagnosis head · B, score head", size=6, color=MUTED, ha="right")
    # 评估协议：紧接模型卡片
    hb = 0.13
    yb = y0 - 0.035 - hb
    rbox(ax, 0.0, yb, 0.97, hb, fc="#f6f6f3", ec=LINE)
    bw = 0.022
    bh = bw * (ax.bbox.width / ax.bbox.height)
    for r in range(3):
        for c in range(5):
            ax.add_patch(Rectangle((0.025 + c * (bw + 0.006), yb + 0.018 + (2 - r) * (bh + 0.008)), bw, bh,
                                   fc=HIST if c == (r + 1) % 5 else "white", ec=HIST, lw=0.4, transform=ax.transAxes))
    t(ax, 0.19, yb + 0.088, "Cross-validation: 5 folds × 3, by participant", size=6, maxw=0.76)
    t(ax, 0.19, yb + 0.038, "held-out test set · temporal validation", size=6, color=INK2, maxw=0.76)


# ---------------------------------------------------------------- d 剔除逆转者实验

def _rng(vals, signed=True, nd=3):
    """范围写法：同号用 "a to b"（负数）或 "a–b"（正数），避免 "−0.010–0.003" 这种歧义。"""
    lo, hi = min(vals), max(vals)
    f = (lambda v: f"{v:+.{nd}f}") if signed else (lambda v: f"{v:.{nd}f}")
    if abs(hi - lo) < 0.5 * 10 ** -nd:
        s = f(lo)
    elif lo >= 0 and hi >= 0:
        s = f"{f(lo)}–{f(hi).lstrip('+')}"
    else:
        s = f"{f(lo)} to {f(hi)}"
    return s.replace("-", "−")


def curation_numbers():
    """从结果表读 2×2 实验的数字（四个学习模型的范围）。"""
    tab = ROOT / "results/tables"
    import pandas as pd
    models = ["GBM-aggregate", "LongForMAD", "GRU", "CTMT"]
    t2 = pd.read_csv(tab / "t2_discrimination.csv")
    cvm = t2[(t2.setting.str.startswith("Repeated")) & t2.model.isin(models)]
    ev = pd.read_csv(tab / "t3_curation_eval.csv")
    tr = pd.read_csv(tab / "t3_curation_train.csv")
    to = pd.read_csv(tab / "s_total_optimism.csv")
    g = lambda df, k, col="diff": df[(df.model.isin(models)) & (df.metric == k)][col].tolist()  # noqa: E731
    pcn_full = g(tr, "mean_P(CN)_in_MCI", "full_train")
    pcn_cur = g(tr, "mean_P(CN)_in_MCI", "curated_train")
    obs = tr[tr.metric == "mean_P(CN)_in_MCI"]["observed_rate"].iloc[0]
    pct = lambda v: f"{100 * min(v):.1f}–{100 * max(v):.1f}%" if max(v) - min(v) >= 0.0005 else f"{100 * v[0]:.1f}%"  # noqa: E731
    return {
        "main": [("macro-AUROC", _rng(cvm.macro_auroc_est, False, 2)), ("conversion AUROC", _rng(cvm.mci_conv_est, False, 2)),
                 ("P(reversion)", f"{pct(pcn_full)} vs {100 * obs:.1f}%")],
        "eval": [("macro-AUROC", _rng(g(ev, "macro_auroc"))), ("conversion AUROC", _rng(g(ev, "mci_conv"))),
                 ("P(reversion)", "not evaluable")],
        "train": [("macro-AUROC", _rng(g(tr, "macro_auroc"))), ("conversion AUROC", _rng(g(tr, "mci_conv"))),
                  ("P(reversion)", f"{pct(pcn_cur)} vs {100 * obs:.1f}%")],
        "both": [("macro-AUROC", _rng(g(to, "macro_auroc"))), ("conversion AUROC", _rng(g(to, "mci_conv"))),
                 ("P(reversion)", "not evaluable")],
    }


def panel_d(ax):
    panel(ax, "d", "Excluding participants who revert")
    nums = curation_numbers()
    gx, gy, cw, chh, gap = 0.11, 0.25, 0.435, 0.30, 0.02
    CUR = "#a33a2c"
    for k, (lab, col) in enumerate([("full cohort", INK), ("curated cohort", CUR)]):
        xc = gx + cw / 2 + k * (cw + gap)
        t(ax, xc, 0.945, "Trained on", size=6, color=MUTED, ha="center")
        t(ax, xc, 0.9, lab, size=6.4, weight="bold", ha="center", color=col)
    for k, (lab, col) in enumerate([("full data", INK), ("curated data", CUR)]):
        yc = gy + chh * 1.5 + gap - k * (chh + gap)
        t(ax, 0.02, yc, "Evaluated on", size=6, color=MUTED, ha="center", rotation=90)
        t(ax, 0.065, yc, lab, size=6.4, weight="bold", ha="center", rotation=90, color=col)
    cells = [  # (列, 行, 底色, 边框, 标题, 数字键, 加粗的行)
        (0, 0, "white", HIST, "Main analysis", "main", None),
        (1, 0, "#fbe4e0", "#c0503f", "P(reversion) lost", "train", 2),
        (0, 1, "#f6efe0", "#9a7b2f", "Pooled metrics inflated", "eval", 0),
        (1, 1, "#efeeea", INK2, "Both curated", "both", 0),
    ]
    for c, r, fc, ec, head, key, bold in cells:
        x = gx + c * (cw + gap)
        y = gy + (1 - r) * (chh + gap)
        rbox(ax, x, y, cw, chh, fc=fc, ec=ec, lw=0.9 if (c, r) == (0, 0) else 0.7)
        t(ax, x + cw / 2, y + chh * 0.88, head, size=6.4, weight="bold", ha="center", color=ec, maxw=cw - 0.03)
        for j, (lab, val) in enumerate(nums[key]):
            yy = y + chh * (0.735 - j * 0.245)
            wt = "bold" if j == bold else "normal"
            t(ax, x + 0.025, yy, lab, size=6, color=MUTED)
            t(ax, x + 0.025, yy - chh * 0.11, val, size=6.4, weight=wt, color=INK, maxw=cw - 0.04)
    for k, line in enumerate(["Curated: participants with any MCI → CN transition, and CN",
                              "points with a dementia outcome, removed (457 points, 12.0%),",
                              "following LongForMAD. Changes are relative to the main",
                              "analysis; ranges across the four learned models tested."]):
        t(ax, gx, 0.19 - k * 0.046, line, size=6, color=INK2, maxw=0.89)


def main():
    raw = json.loads((ROOT / "results/tables/flow_counts.json").read_text())
    flow = {"adni_participants": raw["participants_with_any_dx"],
            "no_cn_mci": raw["participants_with_any_dx"] - raw["participants_with_cn_mci_dx"],
            "candidates": raw["candidate_points"], "candidate_participants": raw["participants_with_cn_mci_dx"],
            "short": raw["status_insufficient_followup"], "no_window": raw["status_no_dx_in_window"],
            "shared": raw["status_duplicate_outcome"], "analysed": raw["included_points"],
            "analysed_participants": raw["included_participants"]}
    assert flow["short"] + flow["no_window"] + flow["shared"] + flow["analysed"] == flow["candidates"]
    fig = plt.figure(figsize=(W, H))
    fig.canvas.draw()
    ax_a = fig.add_axes([0.035, 0.535, 0.36, 0.43])
    ax_b = fig.add_axes([0.42, 0.555, 0.57, 0.41])
    ax_c = fig.add_axes([0.035, 0.03, 0.44, 0.44])
    ax_d = fig.add_axes([0.53, 0.03, 0.46, 0.44])
    panel_a(ax_a, flow)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)
    for ext, kw in [("pdf", {}), ("png", {"dpi": 600}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(FIG / f"fig1_overview.{ext}", bbox_inches="tight", pad_inches=0.03, **kw)
    plt.close(fig)


if __name__ == "__main__":
    main()
