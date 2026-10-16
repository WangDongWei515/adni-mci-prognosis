"""
论文配图（正文 Fig. 1–5、补充 Fig. S1–S3）。读取 results/predictions 与 results/tables（由 make_report.py 生成），
不重新计算 bootstrap，因此可以单独、快速地重画。

图宽按 Springer Nature 模板（sn-jnl）的版心宽度 372 pt ≈ 5.15 in 设定，放入 LaTeX 时缩放比例≈1，字号保持 6.5–7 pt。
编号按正文中首次引用的顺序：Fig. 4 = 消融，Fig. 5 = 队列筛选；Fig. S1 = 模态缺失，Fig. S2 = CV 的 ROC。

用法：.venv/bin/python scripts/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.build import SCALES  # noqa: E402
from src.evaluation import report as R  # noqa: E402
from src.modeling.data import load_dataset  # noqa: E402

TAB = ROOT / "results/tables"
FIG = ROOT / "paper/figures"
FIG.mkdir(parents=True, exist_ok=True)
W = 5.1  # 英寸，略小于 sn-jnl 版心宽度

MAIN = ["GBM-snapshot", "GBM-aggregate", "LongForMAD", "GRU", "CTMT"]
SHORT = {"GBM-snapshot": "GBM-snapshot", "GBM-aggregate": "GBM-aggregate", "LongForMAD": "LongForMAD-style",
         "GRU": "GRU", "CTMT": "CTMT"}
# 固定配色顺序（dataviz 校验器已验证）；同一模型在所有图中颜色一致，另配线型/标记作为第二编码
COLOR = {"CTMT": "#2a78d6", "GRU": "#eb6834", "LongForMAD": "#1baf7a", "GBM-aggregate": "#eda100",
         "GBM-snapshot": "#e87ba4"}
STYLE = {"CTMT": "-", "GRU": "--", "LongForMAD": "-.", "GBM-aggregate": ":", "GBM-snapshot": (0, (5, 1, 1, 1))}
MARK = {"CTMT": "o", "GRU": "s", "LongForMAD": "^", "GBM-aggregate": "D", "GBM-snapshot": "v"}
TASKS = {"mci_conv": "MCI → dementia", "mci_rev": "MCI → CN (reversion)", "cn_prog": "CN → MCI/dementia"}
# 任务作为类别时用前三个类别色（三色两两之间都通过了 CVD 校验）
TASK_COLOR = {"mci_conv": "#2a78d6", "mci_rev": "#eb6834", "cn_prog": "#1baf7a"}
TASK_MARK = {"mci_conv": "o", "mci_rev": "s", "cn_prog": "^"}
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e5e4df", "#8a8983"

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "legend.frameon": False, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(FIG / f"{name}.tiff", bbox_inches="tight", pad_inches=0.02, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def panel_label(ax, i, dx=-0.16, dy=1.06):
    ax.text(dx, dy, "abcdef"[i], transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def task_scores(d, task):
    om = d["origin_mci"] == 1
    if task == "mci_conv":
        g = d[om]; return (g["y"] == 2).astype(int), g["p_Dementia"]
    if task == "mci_rev":
        g = d[om]; return (g["y"] == 0).astype(int), g["p_CN"]
    g = d[~om]; return (g["y"] != 0).astype(int), 1 - g["p_CN"]


def fig_roc(preds, samples, eval_set, name):
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.15))
    for i, (ax, (task, tname)) in enumerate(zip(axes, TASKS.items())):
        ax.plot([0, 1], [0, 1], color=REF, lw=0.6, ls="--", zorder=0)
        for m in MAIN:
            d = R._join(preds[(preds["model"] == m) & preds["eval_set"].str.startswith(eval_set)], samples)
            y, s = task_scores(d, task)
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=COLOR[m], ls=STYLE[m], lw=1.1, label=SHORT[m])
        ax.set_title(f"{tname}\n{int(y.sum())} events / {len(y)} prediction points", fontsize=6.5, linespacing=1.3)
        ax.set_xlabel("1 − specificity")
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        if i:
            ax.set_yticklabels([])
        panel_label(ax, i, dx=-0.22 if i == 0 else -0.08)
    axes[0].set_ylabel("Sensitivity")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04), handlelength=2.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
    save(fig, name)


def fig_history(strata):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.2), sharey=True)
    for i, (ax, task, tname) in enumerate(zip(axes, ["mci_conv", "cn_prog"], ["MCI → dementia", "CN → MCI/dementia"])):
        for m in MAIN:
            g = strata[strata["model"] == m]
            ax.plot(range(len(g)), g[task], color=COLOR[m], ls=STYLE[m], marker=MARK[m], ms=3.2, lw=1.1, label=SHORT[m])
        g0 = strata[strata["model"] == MAIN[0]]
        ev = "n_mci_conv_events" if task == "mci_conv" else "n_cn_prog_events"
        ax.set_xticks(range(len(g0)))
        ax.set_xticklabels([f"{h}\n({e})" for h, e in zip(g0["history_steps"], g0[ev])])
        ax.set_xlabel("Visit days in history (events)")
        ax.set_title(tname)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        panel_label(ax, i, dx=-0.16 if i == 0 else -0.06)
    axes[0].set_ylabel("AUROC (pooled five-fold CV)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05), handlelength=2.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig3_history_length")


def fig_ablation(abl):
    """Fig. 4：单面板森林图，每个变体一行，三个任务用颜色 + 形状区分。"""
    order = ["no-time-encoding", "window-1y", "window-3y", "cognition-only", "MRI-only", "no-step-dropout",
             "classification-only", "test-time-no-MRI", "test-time-last-step-only"]
    nice = {"no-time-encoding": "Visit order instead of continuous time", "window-1y": "History window: 1 year",
            "window-3y": "History window: 3 years", "cognition-only": "Cognitive scores only (no MRI)",
            "MRI-only": "MRI volumes only (no cognitive scores)", "no-step-dropout": "No visit-dropout augmentation",
            "classification-only": "Classification head only (no head B)",
            "test-time-no-MRI": "MRI withheld", "test-time-last-step-only": "Most recent visit day only"}
    fig, ax = plt.subplots(figsize=(W, 3.3))
    off = {"mci_conv": -0.24, "mci_rev": 0.0, "cn_prog": 0.24}
    for task, tname in TASKS.items():
        for i, v in enumerate(order):
            r = abl[(abl["model"] == f"CTMT[{v}]") & (abl["metric"] == task)].iloc[0]
            ax.errorbar(r["diff"], i + off[task], xerr=[[r["diff"] - r["ci_low"]], [r["ci_high"] - r["diff"]]],
                        fmt=TASK_MARK[task], color=TASK_COLOR[task], ms=3.4, elinewidth=0.8, capsize=1.5,
                        label=tname if i == 0 else None)
    ax.axvline(0, color=REF, lw=0.6, ls="--")
    ax.axhline(6.5, color=INK2, lw=0.5)
    ax.text(0.01, 6.42, "Above: retrained with the modification", transform=ax.get_yaxis_transform(), ha="left",
            va="bottom", fontsize=6, color=INK2)
    ax.text(0.01, 6.58, "Below: full model, reduced input at test time", transform=ax.get_yaxis_transform(), ha="left",
            va="top", fontsize=6, color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([nice[v] for v in order])
    ax.set_ylim(len(order) - 0.5, -0.6)
    ax.set_xlabel("ΔAUROC versus full CTMT (95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, handletextpad=0.3)
    fig.tight_layout()
    save(fig, "fig4_ablation")


def fig_curation(cur):
    """Fig. 5：三个模型上下排列，每个面板 3 个结局 × 4 种训练/评估组合，带 95% CI。"""
    def parse(s):
        if str(s).startswith("nan"):
            return np.nan, np.nan, np.nan
        e, rest = s.split(" (")
        lo, hi = rest.rstrip(")").split("–")
        return float(e), float(lo), float(hi)

    conds = [("full-cohort training", "full test set", "Full-cohort training, full test set", "#2a78d6", None),
             ("full-cohort training", "curated test subset", "Full-cohort training, curated test subset", "#2a78d6", "//////"),
             ("curated training", "full test set", "Curated training, full test set", "#eb6834", None),
             ("curated training", "curated test subset", "Curated training, curated test subset", "#eb6834", "//////")]
    archs = [("CTMT", "CTMT"), ("LongForMAD", "LongForMAD-style Transformer"), ("GBM-aggregate", "GBM-aggregate")]
    fig, axes = plt.subplots(3, 1, figsize=(W, 4.6), sharex=True)
    x = np.arange(len(TASKS))
    w = 0.19
    for k, (ax, (arch, title)) in enumerate(zip(axes, archs)):
        c = cur[cur["architecture"] == arch]
        for j, (tr, ev, lab, col, hatch) in enumerate(conds):
            r = c[(c["training cohort"] == tr) & (c["evaluation set"] == ev)].iloc[0]
            vals = np.array([parse(r[t]) for t in TASKS])
            xs = x + (j - 1.5) * w
            ok = ~np.isnan(vals[:, 0])
            ax.bar(xs[ok], vals[ok, 0], w * 0.9, color=col if hatch is None else "white", edgecolor=col, hatch=hatch,
                   lw=0.7, label=lab)
            ax.errorbar(xs[ok], vals[ok, 0], yerr=[vals[ok, 0] - vals[ok, 1], vals[ok, 2] - vals[ok, 0]], fmt="none",
                        ecolor=INK2, elinewidth=0.6, capsize=1.3)
            for xx in xs[~ok]:
                ax.text(xx, 0.23, "n/a", ha="center", va="bottom", fontsize=5.5, color=INK2)
        ax.axhline(0.5, color=REF, lw=0.6, ls="--")
        ax.set_ylim(0.2, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("AUROC")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=7, fontweight="bold", x=0.02)
        panel_label(ax, k, dx=-0.09, dy=1.0)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(list(TASKS.values()))
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=0.8)
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.53, 1.0), columnspacing=1.5)
    save(fig, "fig5_curation")


def fig_design(samples, seq):
    """Fig. 1：以一名真实患者为例说明样本构造（多个预测起点、全部历史、24±3 月结局窗、缺失模态）。"""
    counts = samples.groupby("subject_id").size()
    cand = samples[samples["subject_id"].isin(counts[counts >= 4].index) & (samples["target_diagnosis"] == "Dementia")]
    sid = int(cand["subject_id"].iloc[0])
    ss = samples[samples["subject_id"] == sid].sort_values("index_date")
    last = ss.iloc[-1]
    q = seq[seq["sample_id"] == last["sample_id"]].sort_values("time_step")
    t0 = last["index_date"]
    yrs = lambda d: (pd.Timestamp(d) - t0).days / 365.25  # noqa: E731
    fig, ax = plt.subplots(figsize=(W, 2.3))
    x0 = yrs(q["date"].min())
    for k, (lab, cols) in enumerate({"Cognitive scores": SCALES, "MRI volumes": ["icv_mm3"]}.items()):
        yy = 2 - k
        ax.text(x0 - 0.3, yy, lab, ha="right", va="center", fontsize=6.5)
        for _, r in q.iterrows():
            obs = r[cols].notna().any()
            ax.plot(yrs(r["date"]), yy, marker="o" if obs else "x", ms=4 if obs else 2.8, lw=0,
                    color="#2a78d6" if obs else REF, mfc="#2a78d6" if obs else "none", mew=0.7, zorder=3 if obs else 2)
    for _, r in ss.iterrows():
        xx = yrs(r["index_date"])
        ax.plot([xx, xx], [0.38, 0.72], color=INK, lw=0.7)
        ax.text(xx - (0.1 if xx == 0 else 0), 0.14, r["origin_diagnosis"], ha="right" if xx == 0 else "center", fontsize=6)
    ax.text(x0 - 0.3, 0.55, "Index visits (diagnosis)", ha="right", va="center", fontsize=6.5)
    ax.axvspan(21 / 12, 27 / 12, ymin=0.04, ymax=0.9, color="#eb6834", alpha=0.15, lw=0)
    ax.axvline(0, color=INK, lw=1.0)
    ax.text(-0.08, 2.72, "Index date", ha="right", fontsize=6.5, fontweight="bold")
    ax.text(2, 2.6, "Outcome window\n24 ± 3 months", ha="center", fontsize=6.5, color="#b04a1f")
    ax.text(yrs(last["target_date"]), 0.55, last["target_diagnosis"], ha="center", va="center", fontsize=6.5,
            fontweight="bold", color="#b04a1f")
    ax.annotate("", xy=(0, 1.38), xytext=(x0, 1.38), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7))
    ax.text(x0, 1.47, f"Model input: all {len(q)} visit days up to the index date\n"
            "(irregular gaps; unmeasured modalities kept as [MISS])", ha="left", va="bottom", fontsize=6,
            color=INK2, linespacing=1.25)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Years relative to the index date")
    ax.set_ylim(0, 3.05)
    ax.plot([], [], "o", color="#2a78d6", ms=4, label="measured")
    ax.plot([], [], "x", color=REF, ms=3, label="not measured that day")
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.04), handletextpad=0.2)
    fig.tight_layout()
    save(fig, "fig1_design")
    return sid, len(ss), len(q)


def fig_missingness(seq):
    """Supplementary Fig. S1：每个就诊日有哪些模态被测量。"""
    mods = {"ADAS-Cog13": "ADAS13", "CDR-SB": "CDRSB", "MMSE": "MMSE", "MoCA": "MoCA", "MRI": "icv_mm3"}
    obs = pd.DataFrame({k: seq[v].notna() for k, v in mods.items()})
    combo = obs.apply(lambda r: " + ".join(k for k in mods if r[k]) or "none", axis=1)
    vc = combo.value_counts()
    top = vc.head(10)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1, 1.5]})
    rate = obs.mean()[::-1] * 100
    axes[0].barh(rate.index, rate.values, color="#2a78d6", height=0.6)
    for i, v in enumerate(rate.values):
        axes[0].text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=6)
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Visit days with the measure (%)")
    share = top.values / len(seq) * 100
    axes[1].barh(range(len(top)), share, color="#2a78d6", height=0.6)
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels(top.index, fontsize=5.8)
    for i, v in enumerate(share):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=6)
    axes[1].set_xlim(0, 36)
    axes[1].set_xlabel(f"Share of {len(seq):,} visit days (%)")
    axes[1].set_title("Ten most frequent same-day combinations")
    panel_label(axes[0], 0, dx=-0.38)
    panel_label(axes[1], 1, dx=-0.95)
    fig.tight_layout(w_pad=1.0)
    save(fig, "figS1_missingness")
    vc.to_csv(TAB / "tableS_modality_combinations.csv")


def fig_calibration(preds, samples):
    """Supplementary Fig. S3：MCI 起点 P(痴呆) 的校准曲线（5 折 CV 汇总，按预测概率十分位分组）。"""
    fig, ax = plt.subplots(figsize=(3.0, 2.9))
    ax.plot([0, 1], [0, 1], color=REF, lw=0.6, ls="--")
    for m in MAIN:
        d = R._join(preds[(preds["model"] == m) & preds["eval_set"].str.startswith("cv")], samples)
        d = d[d["origin_mci"] == 1]
        q = pd.qcut(d["p_Dementia"], 10, duplicates="drop")
        g = d.groupby(q, observed=True).agg(p=("p_Dementia", "mean"), o=("y", lambda y: (y == 2).mean()))
        ax.plot(g["p"], g["o"], color=COLOR[m], ls=STYLE[m], marker=MARK[m], ms=3, lw=1.0, label=SHORT[m])
    ax.set_xlabel("Predicted probability of dementia at 2 years")
    ax.set_ylabel("Observed proportion")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "figS3_calibration")


def main():
    samples, seq = load_dataset(ROOT)
    preds = R.load_predictions(ROOT)
    for old in ["fig4_curation", "fig5_ablation", "figS1_roc_cv", "figS2_missingness"]:
        for ext in ("pdf", "png", "tiff"):
            (FIG / f"{old}.{ext}").unlink(missing_ok=True)
    print("fig1 example:", fig_design(samples, seq))
    fig_roc(preds, samples, "test", "fig2_roc_test")
    fig_history(pd.read_csv(TAB / "table_history_strata_cv.csv"))
    fig_ablation(pd.read_csv(TAB / "table4_ablation_diff.csv"))
    fig_curation(pd.read_csv(TAB / "table5_curation.csv"))
    fig_missingness(seq)
    fig_roc(preds, samples, "cv", "figS2_roc_cv")
    fig_calibration(preds, samples)
    print(sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
