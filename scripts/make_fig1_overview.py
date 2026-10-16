"""Figure 1 · 总览图（5 panel：队列流程 / 预测任务 / 模型对照 / 评估协议 / 逆转剔除实验）。

本刊 Sci Rep，双栏整页宽 17.8 cm。
输出 paper/figures/fig1_overview.{pdf,png,tiff}；纯 matplotlib 矢量 PDF。

正文引用对应：
  Fig. 1a = 队列流程（候选预测点如何筛选）—— panel a
  Fig. 1b = 一个预测点的输入构成        —— panel b

修订要点（v2）：
- 事实修正 1：Δt 是"距 index date 的天数"，index 处为 0（原图误标为距首次访视）。
- 事实修正 2：head B（4 个量表回归）属于两个 GBM + GRU + CTMT（原图漏了 GBM-snapshot）。
- 事实修正 3：index 之后的随访只提供标签，不作输入；原图把它画成了输入。
- 视觉修正：加深描边、统一圆角与字号阶梯、panel 内分带避免元素互相压盖。

分镜脚本：docs/fig1_overview_storyboard.md
用法：.venv/bin/python scripts/make_fig1_overview.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper/figures"
TAB = ROOT / "results/tables"
FIG.mkdir(parents=True, exist_ok=True)

FLOW = json.loads((TAB / "flow_counts.json").read_text())

# ----- 字号阶梯（只有 5 档）-----
FS_LABEL = 12.0
FS_TITLE = 7.6
FS_BODY = 6.6
FS_SMALL = 5.9
FS_TINY = 5.6

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FS_BODY,
    "axes.labelsize": FS_BODY,
    "xtick.labelsize": FS_SMALL,
    "ytick.labelsize": FS_SMALL,
    "legend.fontsize": FS_SMALL,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ----- 配色 -----
INK = "#1F2933"
INK2 = "#5A6672"
MUTE = "#B4BCC4"
RULE = "#8A949E"
C_INPUT = "#F2F5F8"
C_REF_BG, C_REF_ED = "#EDEFF2", "#7A848E"
C_WARN_BG, C_WARN_ED = "#FDF0D2", "#B87400"
C_ALERT_BG, C_ALERT_ED = "#FBE0E0", "#A8322F"
C_MAIN_BG, C_MAIN_ED = "#DCEBFB", "#2F6FB0"
C_TASK_A = "#B24745"
C_TASK_B = "#D08A22"
C_TASK_C = "#2E7D6B"
C_OK = "#3F8F5B"

# ----- 画布与栅格：a 横贯整幅（流程适合横向），b/c 一行，d/e 一行 -----
W_IN, H_IN = 7.0, 5.75
fig = plt.figure(figsize=(W_IN, H_IN))

MX = 0.026
GY = 0.042
PA_TOP, PA_H = 0.932, 0.168
PB_H = 0.342
PD_H = 0.316
PB_Y = PA_TOP - PA_H - GY - PB_H
PD_Y = PB_Y - GY - PD_H

rects = {
    "a": (MX, PA_TOP - PA_H, 1 - 2 * MX, PA_H),
    "b": (MX, PB_Y, 0.548, PB_H),
    "c": (MX + 0.578, PB_Y, 1 - 2 * MX - 0.578, PB_H),
    "d": (MX, PD_Y, 0.358, PD_H),
    "e": (MX + 0.388, PD_Y, 1 - 2 * MX - 0.388, PD_H),
}


def new_panel(key):
    ax = fig.add_axes(rects[key])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def panel_label(ax, label, subtitle, dy=1.055):
    ax.text(-0.012, dy, label, fontsize=FS_LABEL, fontweight="bold", color=INK,
            ha="left", va="bottom", transform=ax.transAxes)
    ax.text(-0.012, dy - 0.062, subtitle, fontsize=FS_TITLE, color=INK,
            ha="left", va="top", transform=ax.transAxes, fontweight="bold")


def rbox(ax, x, y, w, h, fc, ec, lw=0.8, rounding=0.012, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                zorder=z, alpha=alpha))


def arrow(ax, x1, y1, x2, y2, color=RULE, lw=0.9, style="-|>", rad=0.0, z=4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                                 lw=lw, mutation_scale=7.5, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z))


# ============================================================
# PANEL a · Cohort flow
# ============================================================
ax_a = new_panel("a")
panel_label(ax_a, "a", "Cohort and prediction points")

n_dem_only = FLOW["participants_with_any_dx"] - FLOW["participants_with_cn_mci_dx"]

# 横向流程：起点 → 候选 → 分析队列；三个排除框挂在候选框下方
row_y, row_h = 0.545, 0.355
w_start, w_cand, w_anal = 0.215, 0.300, 0.235
x_start = 0.010
x_cand = x_start + w_start + 0.088
x_anal = x_cand + w_cand + 0.088

# ① 起点
rbox(ax_a, x_start, row_y, w_start, row_h, fc=C_INPUT, ec=RULE, lw=0.85)
ax_a.text(x_start + w_start / 2, row_y + row_h * 0.78, "ADNI participants",
          fontsize=FS_SMALL, color=INK2, ha="center", va="center")
ax_a.text(x_start + w_start / 2, row_y + row_h * 0.56, "with ≥ 1 dated",
          fontsize=FS_SMALL, color=INK2, ha="center", va="center")
ax_a.text(x_start + w_start / 2, row_y + row_h * 0.34, "diagnosis",
          fontsize=FS_SMALL, color=INK2, ha="center", va="center")
ax_a.text(x_start + w_start / 2, row_y + row_h * 0.13, f"n = {FLOW['participants_with_any_dx']:,}",
          fontsize=FS_SMALL, color=INK, ha="center", va="center", fontweight="bold")

# 排除项：底部一排四个框（第 1 个来自参与者层，后 3 个来自候选点层）
exclusions = [
    ("No CN or MCI diagnosis", f"{n_dem_only:,} participants", 0.0),
    ("< 21 months of later follow-up", f"{FLOW['status_insufficient_followup']:,} points", 0.25),
    ("No diagnosis 21–27 months later", f"{FLOW['status_no_dx_in_window']:,} points", 0.52),
    ("Outcome visit shared with a later point", f"{FLOW['status_duplicate_outcome']:,} points", 0.77),
]
ex_y, ex_h = 0.135, 0.230
ex_w = (0.948 - 3 * 0.014) / 4
for k, (t1, t2, frac) in enumerate(exclusions):
    exx = 0.010 + k * (ex_w + 0.014)
    rbox(ax_a, exx, ex_y, ex_w, ex_h, fc=C_REF_BG, ec=C_REF_ED, lw=0.7, rounding=0.010)
    ax_a.text(exx + ex_w / 2, ex_y + ex_h * 0.66, t1, fontsize=FS_TINY, color=INK2,
              ha="center", va="center", wrap=True)
    ax_a.text(exx + ex_w / 2, ex_y + ex_h * 0.26, t2, fontsize=FS_TINY, color=INK2,
              ha="center", va="center")
    if k == 0:
        arrow(ax_a, x_start + w_start / 2, row_y - 0.002, exx + ex_w / 2, ex_y + ex_h + 0.004,
              color=RULE, lw=0.7, rad=0.0)
    else:
        ax_a.plot([x_cand + w_cand * (0.20 + 0.30 * (k - 1))] * 2,
                  [row_y - 0.002, ex_y + ex_h + 0.026], color=RULE, lw=0.6, zorder=1)
        arrow(ax_a, x_cand + w_cand * (0.20 + 0.30 * (k - 1)), ex_y + ex_h + 0.028,
              exx + ex_w / 2, ex_y + ex_h + 0.004, color=RULE, lw=0.7, rad=0.0)

arrow(ax_a, x_start + w_start, row_y + row_h / 2, x_cand - 0.004, row_y + row_h / 2,
      color=RULE, lw=0.9)

# ② 候选预测点
rbox(ax_a, x_cand, row_y, w_cand, row_h, fc=C_INPUT, ec=RULE, lw=0.85)
ax_a.text(x_cand + w_cand / 2, row_y + row_h * 0.85, "Candidate prediction points",
          fontsize=FS_SMALL, color=INK2, ha="center", va="center")
ax_a.text(x_cand + w_cand / 2, row_y + row_h * 0.66, "every CN or MCI diagnosis",
          fontsize=FS_SMALL, color=INK2, ha="center", va="center")
ax_a.text(x_cand + w_cand / 2, row_y + row_h * 0.38,
          f"n = {FLOW['candidate_points']:,}",
          fontsize=FS_BODY, color=INK, ha="center", va="center", fontweight="bold")
ax_a.text(x_cand + w_cand / 2, row_y + row_h * 0.17,
          f"{FLOW['participants_with_cn_mci_dx']:,} participants",
          fontsize=FS_TINY, color=INK2, ha="center", va="center")

# 主流转：候选 → 分析
arrow(ax_a, x_cand + w_cand, row_y + row_h / 2, x_anal - 0.004, row_y + row_h / 2,
      color=RULE, lw=0.9)

# ③ 分析队列
rbox(ax_a, x_anal, row_y, w_anal, row_h, fc=C_MAIN_BG, ec=C_MAIN_ED, lw=1.1)
ax_a.text(x_anal + w_anal / 2, row_y + row_h * 0.80, "Analysed",
          fontsize=FS_SMALL, color=C_MAIN_ED, ha="center", va="center", fontweight="bold")
ax_a.text(x_anal + w_anal / 2, row_y + row_h * 0.55,
          f"{FLOW['included_points']:,} points",
          fontsize=FS_BODY, color=INK, ha="center", va="center", fontweight="bold")
ax_a.text(x_anal + w_anal / 2, row_y + row_h * 0.32,
          f"{FLOW['included_participants']:,} participants",
          fontsize=FS_SMALL, color=INK2, ha="center", va="center")
ax_a.text(x_anal + w_anal / 2, row_y + row_h * 0.13,
          "no participant in two folds",
          fontsize=FS_TINY, color=INK2, ha="center", va="center", style="italic")

# ============================================================
# PANEL b · Realistic two-year prediction task
# ============================================================
ax_b = new_panel("b")
panel_label(ax_b, "b", "Realistic two-year prediction task")

# 3 个输入访视 + index + 1 个未来随访（未来仅提供标签）
visit_x = [0.070, 0.215, 0.360, 0.560]
visit_dx = ["CN", "MCI", "MCI", "MCI"]
visit_mri = [1, 0, 1, 0]
visit_cog = [1, 1, 1, 1]
INDEX = 3
visit_dt = [-730, -490, -220, 0]
FUT_X = 0.790

y_out = 0.870
y_axis = 0.600
y_p = 0.330

# ① outcome 条
out_x, out_w, out_h = 0.430, 0.560, 0.120
rbox(ax_b, out_x, y_out, out_w, out_h, fc=C_WARN_BG, ec=C_WARN_ED, lw=1.0)
ax_b.plot([out_x + 0.018, out_x + 0.018], [y_out + 0.024, y_out + out_h - 0.024],
          color=C_WARN_ED, lw=1.6, solid_capstyle="round", zorder=3)
tx = out_x + 0.038
ax_b.text(tx, y_out + out_h * 0.66, "outcome @ 21–27 mo", fontsize=FS_BODY,
          color=INK, ha="left", va="center", fontweight="bold")
ax_b.text(tx, y_out + out_h * 0.26, "diagnosis = CN  ·  reversion kept",
          fontsize=FS_SMALL, color=C_WARN_ED, ha="left", va="center",
          fontweight="bold", style="italic")

# ② timeline 条
ax_b.add_patch(Rectangle((visit_x[INDEX] + 0.014, y_axis - 0.215), 1.0 - visit_x[INDEX] - 0.014, 0.300,
                         facecolor=C_WARN_BG, edgecolor="none", alpha=0.5, zorder=0))

ax_b.plot([visit_x[0] - 0.040, FUT_X + 0.062], [y_axis, y_axis],
          color=INK, lw=1.0, zorder=1, solid_capstyle="round")
arrow(ax_b, FUT_X + 0.054, y_axis, FUT_X + 0.090, y_axis, color=INK, lw=1.0)
ax_b.text(FUT_X + 0.100, y_axis, "time", fontsize=FS_SMALL, color=INK2,
          ha="left", va="center", style="italic")

ax_b.plot([visit_x[INDEX]] * 2, [y_axis - 0.222, y_axis + 0.096],
          color=C_WARN_ED, lw=1.0, ls=(0, (2.2, 1.8)), zorder=2)
ax_b.text(visit_x[INDEX] + 0.010, y_axis + 0.104, "index date", fontsize=FS_TINY,
          color=C_WARN_ED, ha="left", va="bottom", fontweight="bold", zorder=3)

y_mod = y_axis - 0.108
for i, (x, dx, dt, mri, cog) in enumerate(zip(visit_x, visit_dx, visit_dt, visit_mri, visit_cog)):
    is_index = (i == INDEX)
    ax_b.scatter([x], [y_axis], s=118 if is_index else 60,
                 color=C_WARN_ED if is_index else "#9AA4AE",
                 edgecolor="white", linewidth=0.9, zorder=6)
    if is_index:
        ax_b.text(x, y_axis, "i", fontsize=6.4, color="white", fontweight="bold",
                  ha="center", va="center", zorder=7)
    ax_b.text(x, y_axis + 0.046, dx, fontsize=FS_BODY,
              color=INK if is_index else INK2,
              fontweight="bold" if is_index else "normal", ha="center", va="bottom")
    ax_b.text(x, y_axis - 0.040, f"{dt:+d}" if dt else "0", fontsize=FS_TINY,
              color=C_WARN_ED if is_index else MUTE, ha="center", va="top",
              fontweight="bold" if is_index else "normal")
    for present, dy, fc, ed, tag in ((cog, 0.0, C_INPUT, RULE, "C"),
                                     (mri, -0.050, "#E3E8ED", "#7A848E", "M")):
        if present:
            ax_b.add_patch(Rectangle((x - 0.020, y_mod + dy), 0.040, 0.036,
                                     facecolor=fc, edgecolor=ed, lw=0.65, zorder=3))
            ax_b.text(x, y_mod + dy + 0.018, tag, fontsize=FS_TINY, color=ed,
                      ha="center", va="center", fontweight="bold", zorder=4)
        else:
            ax_b.plot([x - 0.013, x + 0.013], [y_mod + dy + 0.018] * 2,
                      color=MUTE, lw=0.9, ls=(0, (1.4, 1.4)), zorder=3)

# 未来随访：不作输入
ax_b.scatter([FUT_X], [y_axis], s=60, color="white", edgecolor=MUTE, linewidth=1.0, zorder=6)
ax_b.text(FUT_X, y_axis, "?", fontsize=6.4, color=MUTE, fontweight="bold",
          ha="center", va="center", zorder=7)
ax_b.text(FUT_X, y_axis - 0.040, "+250", fontsize=FS_TINY, color=MUTE,
          ha="center", va="top")
ax_b.text(FUT_X, y_mod - 0.004, "not used\nas input", fontsize=FS_TINY, color=MUTE,
          ha="center", va="top", style="italic", linespacing=1.3)
arrow(ax_b, FUT_X, y_axis + 0.028, out_x + out_w * 0.62, y_out - 0.006,
      color=C_WARN_ED, lw=0.9, rad=0.14)

ax_b.text(visit_x[0] - 0.040, y_mod - 0.098,
          "Δt in days from index date  ·  C, cognitive scores  ·  M, MRI  ·  dashed, not measured",
          fontsize=FS_TINY, color=INK2, ha="left", va="top", style="italic")

# ③ 三原则
ax_b.plot([visit_x[0] - 0.040, 0.985], [y_p + 0.036, y_p + 0.036], color=MUTE, lw=0.6, zorder=1)
ax_b.text(visit_x[0] - 0.040, y_p + 0.008, "Design principles", fontsize=FS_TITLE,
          fontweight="bold", color=INK, ha="left", va="top")
principles = [
    "Full history to the index date — no truncation, no annual binning",
    "Missing modalities left missing; absence is itself information",
    "Reversions kept in training and evaluation — one model predicts all",
]
for k, p in enumerate(principles):
    yy = y_p - 0.070 - k * 0.086
    ax_b.scatter([0.020], [yy], s=50, marker="o", color=C_OK,
                 edgecolor="white", linewidth=0.6, zorder=6)
    ax_b.plot([0.0135, 0.0185, 0.0275], [yy + 0.001, yy - 0.009, yy + 0.012],
              color="white", lw=1.25, solid_capstyle="round",
              solid_joinstyle="round", zorder=7)
    ax_b.text(0.046, yy, p, fontsize=FS_BODY, color=INK, ha="left", va="center")

# ============================================================
# PANEL c · Models compared
# ============================================================
ax_c = new_panel("c")
panel_label(ax_c, "c", "Models compared")

# 逐模型配色与 figs 2–5（make_figures_v2.py 的 COLOR 表）严格一致；
# 行尾徽标给出预测头：A = 三分类诊断，A+B = 另带 4 项量表回归（补充表 S3）
MODEL_GROUPS = [
    ("Reference", [("Index diagnosis only", "#b9b8b1", "A"),
                   ("Logistic regression", "#8a8983", "A")]),
    ("Tabular · GBM", [("GBM-snapshot", "#e87ba4", "A+B"),
                       ("GBM-aggregate", "#eda100", "A+B")]),
    ("Sequence · neural", [("LongForMAD-inspired", "#1baf7a", "A"),
                           ("GRU", "#eb6834", "A+B"),
                           ("CTMT", "#2a78d6", "A+B")]),
]

cy = 0.952
for gtitle, models in MODEL_GROUPS:
    ax_c.text(0.012, cy - 0.018, gtitle, fontsize=FS_TINY, fontweight="bold",
              color=INK2, ha="left", va="center")
    cy -= 0.046
    for mname, mcol, heads in models:
        cy -= 0.078
        rbox(ax_c, 0.0, cy, 1.0, 0.078, fc=mcol, ec=mcol, lw=0.9,
             rounding=0.008, alpha=0.12)
        rbox(ax_c, 0.014, cy + 0.023, 0.026, 0.032, fc=mcol, ec=mcol,
             lw=0.5, rounding=0.006, z=4)
        ax_c.text(0.062, cy + 0.039, mname, fontsize=FS_SMALL, color=INK,
                  ha="left", va="center", zorder=5)
        ax_c.text(0.986, cy + 0.039, heads, fontsize=FS_TINY, color=mcol,
                  ha="right", va="center", fontweight="bold", zorder=5)
        cy -= 0.016
    cy -= 0.008

rbox(ax_c, 0.0, 0.018, 1.0, 0.072, fc=C_INPUT, ec=RULE, lw=0.7, rounding=0.008)
ax_c.text(0.5, 0.054, "Same information · equal tuning budget", fontsize=FS_TINY,
          color=INK, ha="center", va="center")

# ============================================================
# PANEL d · Evaluation protocol
# ============================================================
ax_d = new_panel("d")
panel_label(ax_d, "d", "Evaluation protocol")

rbox(ax_d, 0.0, 0.830, 1.0, 0.097, fc=C_INPUT, ec=RULE, lw=0.7, rounding=0.008)
ax_d.text(0.5, 0.890, "3,823 points · 1,449 participants", fontsize=FS_SMALL,
          color=INK, ha="center", va="center", fontweight="bold")
ax_d.text(0.5, 0.852, "split by participant", fontsize=FS_TINY, color=INK2,
          ha="center", va="center", style="italic")

arrow(ax_d, 0.5, 0.826, 0.5, 0.798, color=RULE, lw=0.85)
ax_d.text(0.5, 0.794, "5 folds × 3 repeats", fontsize=FS_SMALL, color=INK,
          ha="center", va="top", fontweight="bold")

fw, fg = 0.170, 0.026
fx0 = 0.020
fy0 = 0.616
for k in range(5):
    xx = fx0 + k * (fw + fg)
    ax_d.add_patch(Rectangle((xx, fy0), fw, 0.124, facecolor="white",
                             edgecolor=C_MAIN_ED, lw=0.85, zorder=3))
    ax_d.text(xx + fw / 2, fy0 + 0.062, f"F{k + 1}", fontsize=FS_SMALL,
              color=C_MAIN_ED, ha="center", va="center", fontweight="bold", zorder=4)

arrow(ax_d, 0.5, 0.612, 0.5, 0.578, color=RULE, lw=0.85)
ax_d.text(0.0, 0.560, "Three per-transition tasks", fontsize=FS_SMALL,
          color=INK, fontweight="bold", ha="left", va="center")

tasks = [(C_TASK_A, "MCI → dementia", "conversion"),
         (C_TASK_B, "MCI → CN", "reversion"),
         (C_TASK_C, "CN → MCI/dementia", "progression")]
chip_h = 0.104
chip_step = 0.128
chip_top = 0.524
for k, (color, name, sub) in enumerate(tasks):
    yy = chip_top - chip_step * k
    rbox(ax_d, 0.0, yy - chip_h, 1.0, chip_h, fc=color, ec=color, lw=0.8, rounding=0.008)
    ax_d.text(0.5, yy - chip_h * 0.33, name, fontsize=FS_SMALL, color="white",
              fontweight="bold", ha="center", va="center")
    ax_d.text(0.5, yy - chip_h * 0.73, sub, fontsize=FS_TINY, color="white",
              ha="center", va="center")

ax_d.plot([0.0, 1.0], [0.120, 0.120], color=MUTE, lw=0.6)
ax_d.text(0.5, 0.089, "±0.02 equivalence margin", fontsize=FS_TINY, color=INK2,
          ha="center", va="center")
ax_d.text(0.5, 0.046, "Holm correction · cluster bootstrap", fontsize=FS_TINY,
          color=INK2, ha="center", va="center")

# ============================================================
# PANEL e · Reverter-exclusion experiment
# ============================================================
ax_e = new_panel("e")
panel_label(ax_e, "e", "Reverter-exclusion experiment")

mx, my = 0.205, 0.300
cw, ch = 0.380, 0.250
gap = 0.014

for k, (label, ed) in enumerate([("Train: FULL", C_MAIN_ED), ("Train: CURATED", C_ALERT_ED)]):
    x = mx + k * (cw + gap)
    rbox(ax_e, x + 0.008, my + 2 * ch + gap + 0.010, cw - 0.016, 0.050,
         fc=ed, ec=ed, lw=0.5, rounding=0.006, z=3)
    ax_e.text(x + cw / 2, my + 2 * ch + gap + 0.035, label, fontsize=FS_TINY,
              fontweight="bold", color="white", ha="center", va="center", zorder=4)

for k, label in enumerate(["Evaluate:\nFULL", "Evaluate:\nCURATED"]):
    ax_e.text(mx - 0.016, my + (1 - k) * (ch + gap) + ch / 2, label, fontsize=FS_TINY,
              fontweight="bold", color=INK, ha="right", va="center", linespacing=1.35)

# (0,0) 主结果
rbox(ax_e, mx, my + ch + gap, cw, ch, fc=C_MAIN_BG, ec=C_MAIN_ED, lw=1.1)
ax_e.text(mx + cw / 2, my + ch + gap + ch * 0.760, "Main results", fontsize=FS_SMALL,
          fontweight="bold", color=C_MAIN_ED, ha="center", va="center")
ax_e.text(mx + cw / 2, my + ch + gap + ch * 0.480, "conversion AUROC 0.87",
          fontsize=FS_TINY, color=INK, ha="center", va="center")
ax_e.text(mx + cw / 2, my + ch + gap + ch * 0.230, "Fig. 2 · Table 2", fontsize=FS_TINY,
          color=INK2, ha="center", va="center", style="italic")

# (0,1) 训练剔除
rbox(ax_e, mx + cw + gap, my + ch + gap, cw, ch, fc=C_ALERT_BG, ec=C_ALERT_ED, lw=1.1)
rbox(ax_e, mx + cw + gap + 0.008, my + ch + gap + ch - 0.044, cw - 0.016, 0.038,
     fc=C_ALERT_ED, ec=C_ALERT_ED, lw=0.5, rounding=0.006, z=3)
ax_e.text(mx + cw + gap + cw / 2, my + ch + gap + ch - 0.025, "↓ reversion detection",
          fontsize=FS_TINY, fontweight="bold", color="white", ha="center",
          va="center", zorder=4)
ax_e.text(mx + cw + gap + cw / 2, my + ch + gap + ch * 0.600, "reversion AUROC",
          fontsize=FS_TINY, color=INK, ha="center", va="center")
ax_e.text(mx + cw + gap + cw / 2, my + ch + gap + ch * 0.410, "− 0.08 to − 0.36",
          fontsize=FS_SMALL, fontweight="bold", color=C_ALERT_ED, ha="center", va="center")
ax_e.text(mx + cw + gap + cw / 2, my + ch + gap + ch * 0.190, "predicted < 1%  (4.8% seen)",
          fontsize=FS_TINY, color=INK2, ha="center", va="center")

# (1,0) 评估剔除
rbox(ax_e, mx, my, cw, ch, fc=C_WARN_BG, ec=C_WARN_ED, lw=1.1)
rbox(ax_e, mx + 0.008, my + ch - 0.044, cw - 0.016, 0.038,
     fc=C_WARN_ED, ec=C_WARN_ED, lw=0.5, rounding=0.006, z=3)
ax_e.text(mx + cw / 2, my + ch - 0.025, "↑ pooled metrics", fontsize=FS_TINY,
          fontweight="bold", color="white", ha="center", va="center", zorder=4)
ax_e.text(mx + cw / 2, my + ch * 0.600, "macro-AUROC", fontsize=FS_TINY,
          color=INK, ha="center", va="center")
ax_e.text(mx + cw / 2, my + ch * 0.410, "+ 0.017 to + 0.019", fontsize=FS_SMALL,
          fontweight="bold", color=C_WARN_ED, ha="center", va="center")
ax_e.text(mx + cw / 2, my + ch * 0.190, "≈ random class removal", fontsize=FS_TINY,
          color=INK2, ha="center", va="center", style="italic")

# (1,1) 双端 curate
rbox(ax_e, mx + cw + gap, my, cw, ch, fc=C_REF_BG, ec=C_REF_ED, lw=1.1)
rbox(ax_e, mx + cw + gap + 0.008, my + ch - 0.044, cw - 0.016, 0.038,
     fc=C_REF_ED, ec=C_REF_ED, lw=0.5, rounding=0.006, z=3)
ax_e.text(mx + cw + gap + cw / 2, my + ch - 0.025, "LongForMAD-style", fontsize=FS_TINY,
          fontweight="bold", color="white", ha="center", va="center", zorder=4)
ax_e.text(mx + cw + gap + cw / 2, my + ch * 0.585, "macro  ↑", fontsize=FS_TINY,
          color=INK, ha="center", va="center")
ax_e.text(mx + cw + gap + cw / 2, my + ch * 0.395, "conversion  ↓", fontsize=FS_TINY,
          color=INK, ha="center", va="center")
ax_e.text(mx + cw + gap + cw / 2, my + ch * 0.185, "reported numbers look better",
          fontsize=FS_TINY, color=INK2, ha="center", va="center", style="italic")

ax_e.text(0.5, 0.185,
          "Inflation comes from removing the outcome class itself;",
          fontsize=FS_TINY, color=INK, ha="center", va="center", style="italic")
ax_e.text(0.5, 0.135,
          "pooled metrics hide the loss of reversion detection.",
          fontsize=FS_TINY, color=INK, ha="center", va="center", style="italic")
ax_e.text(0.5, 0.062, "values are changes in AUROC", fontsize=FS_TINY,
          color=INK2, ha="center", va="center")

# ============================================================
out_base = FIG / "fig1_overview"
fig.savefig(f"{out_base}.pdf")
fig.savefig(f"{out_base}.png", dpi=600)
fig.savefig(f"{out_base}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print(f"已生成 {out_base}.{{pdf,png,tiff}}")
