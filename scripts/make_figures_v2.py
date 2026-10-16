"""
第 2 版论文配图（按第 1 轮审稿意见重做）。读取 results/tables（由 make_report_v2.py 生成）与预测文件。

颜色约定（R3-m13）：颜色只表示模型，全文一致；任务用分面板区分；训练方式用实心 / 空心区分。
AUROC 一律用「点 + 置信区间」表示，不用柱状图（R2-m10、R3-m12）。
图宽按 sn-jnl 版心（372 pt ≈ 5.15 in）设定。

用法：.venv/bin/python scripts/make_figures_v2.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.build import SCALES  # noqa: E402
from src.evaluation import report_v2 as R  # noqa: E402
from src.evaluation.metrics import calibration_stats, task_arrays  # noqa: E402
from src.modeling.data import load_dataset  # noqa: E402

TAB = ROOT / "results/tables"
FIG = ROOT / "paper/figures"
FIG.mkdir(parents=True, exist_ok=True)
W = 5.1
NAME = {"Index-dx-only": "Index diagnosis only", "Logistic": "Logistic regression", "GBM-snapshot": "GBM-snapshot",
        "GBM-aggregate": "GBM-aggregate", "LongForMAD": "LongForMAD-inspired", "GRU": "GRU", "CTMT": "CTMT"}
# 固定模型配色（dataviz 校验器已验证前五个；两个简单基线用中性灰）
COLOR = {"CTMT": "#2a78d6", "GRU": "#eb6834", "LongForMAD": "#1baf7a", "GBM-aggregate": "#eda100",
         "GBM-snapshot": "#e87ba4", "Logistic": "#8a8983", "Index-dx-only": "#b9b8b1"}
MARK = {"CTMT": "o", "GRU": "s", "LongForMAD": "^", "GBM-aggregate": "D", "GBM-snapshot": "v", "Logistic": "P",
        "Index-dx-only": "X"}
STYLE = {"CTMT": "-", "GRU": "--", "LongForMAD": "-.", "GBM-aggregate": ":", "GBM-snapshot": (0, (5, 1, 1, 1)),
         "Logistic": (0, (1, 1)), "Index-dx-only": (0, (3, 2))}
TASKS = {"mci_conv": "MCI → dementia", "mci_rev": "MCI → CN (reversion)", "cn_prog": "CN → MCI/dementia"}
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e5e4df", "#8a8983"
MUTED = "#8a8983"

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "legend.frameon": False, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2})


def save(fig, name):
    for ext, kw in [("pdf", {}), ("png", {"dpi": 600}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


def label(ax, s, dx=-0.12, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


# ---------------------------------------------------------------- Fig. 1：流程图 + 一名参与者的样本构造

def box(ax, x, y, w, h, text, fc="white", ec=INK2, fs=6.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01", fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None,
            linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6,
                                                                  mutation_scale=6))


def fig1(samples, seq):
    fc = json.loads((TAB / "flow_counts.json").read_text())
    fig = plt.figure(figsize=(W, 5.1))
    ax = fig.add_axes([0, 0.51, 1, 0.49])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n_dem_only = fc['participants_with_any_dx'] - fc['participants_with_cn_mci_dx']
    box(ax, 0.02, 0.87, 0.40, 0.12, f"ADNI participants with ≥1 dated diagnosis\nn = {fc['participants_with_any_dx']:,}")
    ax.plot([0.22, 0.22], [0.87, 0.80], color=INK2, lw=0.6)
    arrow(ax, 0.22, 0.80, 0.22, 0.73)
    box(ax, 0.54, 0.755, 0.44, 0.09, f"No CN or MCI diagnosis (dementia only)\nn = {n_dem_only:,} participants", fc="#f6f5f1")
    arrow(ax, 0.22, 0.80, 0.54, 0.80)
    box(ax, 0.02, 0.59, 0.40, 0.14, f"Candidate prediction points: every CN or\nMCI diagnosis, n = {fc['candidate_points']:,}\n"
                                    f"({fc['participants_with_cn_mci_dx']:,} participants)")
    ex = [(f"<21 months of later follow-up  n = {fc['status_insufficient_followup']:,}", 0.445),
          (f"Follow-up ≥21 months but no diagnosis\n21–27 months later  n = {fc['status_no_dx_in_window']:,}", 0.315),
          (f"Same outcome visit as a later point of the same\nparticipant (the later was kept)  n = {fc['status_duplicate_outcome']:,}", 0.185)]
    for t, y in ex:
        box(ax, 0.54, y, 0.44, 0.105, t, fc="#f6f5f1")
        arrow(ax, 0.22, y + 0.0525, 0.54, y + 0.0525)
    ax.plot([0.22, 0.22], [0.59, 0.20], color=INK2, lw=0.6)
    arrow(ax, 0.22, 0.20, 0.22, 0.165)
    box(ax, 0.02, 0.015, 0.40, 0.15, f"Analysed: {fc['included_points']:,} prediction points\nfrom {fc['included_participants']:,} participants",
        fc="#eaf2fb", ec="#2a78d6", bold=True)
    s = samples
    box(ax, 0.54, 0.015, 0.44, 0.15,
        f"Held-out split: {int((s.split == 'train').sum()):,} / {int((s.split == 'val').sum()):,} / "
        f"{int((s.split == 'test').sum()):,} points (by participant)\n"
        f"Repeated five-fold CV: {len(s):,} points × 3 partitions\nTemporal validation: ADNI1/GO/2 → ADNI3/4", fc="white")
    arrow(ax, 0.42, 0.09, 0.54, 0.09)
    label(ax, "a", dx=0.0, dy=0.97)

    # b：一名参与者
    axb = fig.add_axes([0.2, 0.05, 0.78, 0.40])
    # 选一名缺失模式有代表性的参与者（R3-m10）：结局为痴呆、≥3 个预测点、历史 10–25 个评估日，
    # 且历史中既有只做 MRI 的日子，也有只做了部分认知量表的日子
    counts = samples.groupby("subject_id").size()
    cand = samples[samples["subject_id"].isin(counts[counts >= 3].index) & (samples["target_diagnosis"] == "Dementia")
                   & samples["n_visits"].between(10, 25)]
    best, sid = -1, None
    for r in cand.itertuples():
        h = seq[seq["sample_id"] == r.sample_id]
        cog = h[["ADAS13", "CDRSB", "MMSE"]].notna()
        mri_only = (h["icv_mm3"].notna() & ~cog.any(axis=1)).sum()
        partial = (cog.any(axis=1) & ~cog.all(axis=1)).sum()
        score = min(mri_only, 3) + min(partial, 3) + (h["MoCA"].notna().sum() > 0)
        if score > best and r.sample_id == samples[samples.subject_id == r.subject_id].sort_values("index_date")["sample_id"].iloc[-1]:
            best, sid = score, r.subject_id
    ss = samples[samples["subject_id"] == sid].sort_values("index_date")
    last = ss.iloc[-1]
    q = seq[seq["sample_id"] == last["sample_id"]].sort_values("time_step")
    t0 = last["index_date"]
    yrs = lambda d: (pd.Timestamp(d) - t0).days / 365.25  # noqa: E731
    rows = [("ADAS-Cog13", ["ADAS13"]), ("CDR-SB", ["CDRSB"]), ("MMSE", ["MMSE"]), ("MoCA", ["MoCA"]),
            ("MRI volumes", ["icv_mm3"]), ("Diagnosis", ["dx_CN"])]
    ys = [5.0, 4.4, 3.8, 3.2, 2.5, 1.7]
    for k, (lab, cols) in enumerate(rows):
        yy = ys[k]
        axb.text(yrs(q["date"].min()) - 0.3, yy, lab, ha="right", va="center", fontsize=6.3)
        for _, r in q.iterrows():
            if not r[cols].notna().any():
                continue
            if lab == "Diagnosis":
                d = "CN" if r["dx_CN"] == 1 else "MCI" if r["dx_MCI"] == 1 else "Dem"
                x = yrs(r["date"])
                at0 = abs(x) < 0.05
                axb.text(x - (0.08 if at0 else 0), yy, d, ha="right" if at0 else "center", va="center", fontsize=5.3,
                         color=INK2)
            else:
                axb.plot(yrs(r["date"]), yy, "o", ms=2.8, color="#2a78d6", lw=0)
    axb.axvspan(21 / 12, 27 / 12, ymin=0.03, ymax=0.9, color="#eb6834", alpha=0.15, lw=0)
    axb.axvline(0, color=INK, lw=1.0)
    axb.text(-0.08, 5.45, "Index date", ha="right", fontsize=6.3, fontweight="bold")
    axb.text(2, 5.35, "Outcome window\n21–27 months", ha="center", fontsize=6.3, color="#b04a1f", va="bottom")
    axb.text(2, 1.7, last["target_diagnosis"], ha="center", va="center", fontsize=6.5, fontweight="bold", color="#b04a1f")
    axb.annotate("", xy=(0, 0.35), xytext=(yrs(q["date"].min()), 0.35), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6))
    axb.text(yrs(q["date"].min()), 0.45, f"Input: all {len(q)} assessment days up to the index date\n"
             "(unmeasured modalities are left missing, not imputed)", ha="left", va="bottom", fontsize=5.8,
             color=INK2, linespacing=1.2)
    for _, r in ss.iterrows():
        axb.plot(yrs(r["index_date"]), 0.05, "|", color="#b04a1f" if r["sample_id"] == last["sample_id"] else INK,
                 ms=5, mew=1.2 if r["sample_id"] == last["sample_id"] else 0.8, zorder=5)
    axb.text(yrs(q["date"].min()) - 0.3, 0.05, f"Prediction points\nof this participant ({len(ss)})", ha="right",
             va="center", fontsize=6.0)
    axb.set_yticks([]); axb.spines["left"].set_visible(False)
    axb.set_ylim(-0.2, 5.9)
    axb.set_xlabel("Years relative to the index date")
    label(axb, "b", dx=-0.27, dy=0.97)
    save(fig, "fig1_design")


# ---------------------------------------------------------------- Fig. 2：相对 GBM-aggregate 的配对差值

def fig2():
    """与 GBM-aggregate 的配对 AUROC 差。仅用指数诊断的参考模型在单项任务上 AUROC 恒为 0.5（见 Table 2），不画，
    以免把横轴拉长到看不清 ±0.02 等价带。逻辑回归作为参考放在顶部，与学习模型之间用细线隔开。"""
    from src.evaluation.metrics import corrected_resampled_t
    fm = pd.read_csv(TAB / "s_fold_metrics.csv")
    order = ["Logistic", "GBM-snapshot", "LongForMAD", "GRU", "CTMT"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.15), sharey=True)
    for k, (ax, (t, tn)) in enumerate(zip(axes, TASKS.items())):
        b = fm[fm.model == "GBM-aggregate"].set_index(["repeat", "fold"])[t]
        for i, m in enumerate(order):
            a = fm[fm.model == m].set_index(["repeat", "fold"])[t]
            r = corrected_resampled_t((a - b.reindex(a.index)).to_numpy())
            ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["ci_low"]], [r["ci_high"] - r["diff"]]], fmt=MARK[m],
                        color=COLOR[m], ms=3.8, elinewidth=1.0, capsize=1.8, zorder=3)
        if t == "mci_conv":
            ax.axvspan(-0.02, 0.02, color="#e4e2da", lw=0, zorder=1)
            for x in (-0.02, 0.02):
                ax.axvline(x, color=MUTED, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
            ax.text(0, 0.56, "±0.02 margin", ha="center", va="top", fontsize=5.6, color=INK2, zorder=4,
                    bbox=dict(fc="#e4e2da", ec="none", pad=0.4))
        ax.axvline(0, color=REF, lw=0.6, ls="--", zorder=2)
        ax.axhline(0.5, color=INK2, lw=0.5, zorder=2)
        ax.set_title(tn)
        ax.set_xlabel("ΔAUROC vs GBM-aggregate")
        ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.tick_params(axis="x", labelsize=6)
        label(ax, "abc"[k], dx=-0.12 if k else -0.62, dy=1.03)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([NAME[m] for m in order])
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig2_comparison")


# ---------------------------------------------------------------- Fig. 3：历史跨度分层

def fig3():
    st = pd.read_csv(TAB / "f3_history_strata.csv")
    mix = pd.read_csv(TAB / "s_history_strata_casemix.csv")
    models = ["GBM-aggregate", "LongForMAD", "GRU", "CTMT"]
    strata = mix["stratum"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True)
    for k, (ax, t) in enumerate(zip(axes, ["mci_conv", "cn_prog"])):
        for j, m in enumerate(models):
            g = st[(st.model == m) & (st.task == t)].set_index("stratum").loc[strata]
            x = np.arange(len(strata)) + (j - 1.5) * 0.14
            ax.errorbar(x, g["diff_vs_snapshot"], yerr=[g["diff_vs_snapshot"] - g["diff_lo"], g["diff_hi"] - g["diff_vs_snapshot"]],
                        fmt=MARK[m], color=COLOR[m], ms=3.2, elinewidth=0.8, capsize=1.2, label=NAME[m])
        ev = "mci_conv_events" if t == "mci_conv" else "cn_prog_events"
        nn = "n_mci" if t == "mci_conv" else "n_cn"
        ax.set_xticks(range(len(strata)))
        ax.set_xticklabels([f"{s}\n{e}/{n}" for s, e, n in zip(strata, mix[ev], mix[nn])], fontsize=6)
        ax.axhline(0, color=REF, lw=0.6, ls="--")
        ax.set_title(TASKS[t])
        ax.set_xlabel("History span (events / prediction points)")
        ax.grid(axis="y", color=GRID, lw=0.4)
        label(ax, "ab"[k], dx=-0.14 if k == 0 else -0.05)
    axes[0].set_ylabel("ΔAUROC vs GBM-snapshot (95% CI)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig3_history")


# ---------------------------------------------------------------- Fig. 4：消融

ABL_NAME = {
    "CTMT[no-time-encoding]": "Visit order instead of continuous time", "CTMT[window-1y]": "History window 1 year",
    "CTMT[window-2y]": "History window 2 years", "CTMT[window-3y]": "History window 3 years",
    "CTMT[snapshot-retrained]": "Index snapshot only (retrained)", "CTMT[no-prior-dx]": "No prior diagnoses",
    "CTMT[no-MRI]": "No MRI volumes", "CTMT[MRI-only]": "MRI volumes only (no cognitive scores)",
    "CTMT[no-CDRSB-MMSE]": "No CDR-SB or MMSE", "CTMT[plus-APOE]": "Adding APOE ε4",
    "CTMT[no-step-dropout]": "No visit-dropout augmentation", "CTMT[classification-only]": "No regression head",
    "CTMT[test-time-no-MRI]": "MRI withheld at test time",
    "GRU[GRU-window-1y]": "History window 1 year", "GRU[GRU-window-2y]": "History window 2 years",
    "GRU[GRU-window-3y]": "History window 3 years",
    "GBM-aggregate[window-1y]": "History window 1 year", "GBM-aggregate[window-2y]": "History window 2 years",
    "GBM-aggregate[window-3y]": "History window 3 years", "GBM-aggregate[no-prior-dx]": "No prior diagnoses",
    "GBM-aggregate[no-MRI]": "No MRI volumes", "GBM-aggregate[no-CDRSB-MMSE]": "No CDR-SB or MMSE",
    "GBM-aggregate[plus-APOE]": "Adding APOE ε4",
    "GBM-aggregate[no-dx-CDRSB-MMSE]": "No diagnoses, CDR-SB or MMSE",
}


def fig4():
    ab = pd.read_csv(TAB / "f4_ablation.csv")
    order = [k for k in ABL_NAME if k in set(ab.variant)]
    fam = lambda v: "CTMT" if v.startswith("CTMT") else "GRU" if v.startswith("GRU") else "GBM-aggregate"  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(W, 4.4), sharey=True)
    for k, (ax, t) in enumerate(zip(axes, TASKS)):
        for i, v in enumerate(order):
            r = ab[(ab.variant == v) & (ab.task == t)].iloc[0]
            m = fam(v)
            ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["t_lo"]], [r["t_hi"] - r["diff"]]], fmt=MARK[m],
                        color=COLOR[m], ms=3.2, elinewidth=0.8, capsize=1.2)
        ax.axvline(0, color=REF, lw=0.6, ls="--")
        ax.set_title(TASKS[t], pad=4)
        ax.set_xlabel("ΔAUROC vs full model")
        ax.grid(axis="x", color=GRID, lw=0.4)
        label(ax, "abc"[k], dx=-0.16 if k else -1.2, dy=1.075)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([ABL_NAME[v] for v in order], fontsize=6)
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    # 分组标签
    groups = [(m, [i for i, v in enumerate(order) if fam(v) == m]) for m in ["CTMT", "GRU", "GBM-aggregate"]]
    for m, idx in groups:
        if idx:
            for ax in axes:
                ax.axhspan(min(idx) - 0.5, max(idx) + 0.5, color=COLOR[m], alpha=0.06, lw=0, zorder=0)
            axes[2].text(1.02, np.mean(idx), NAME[m], transform=axes[2].get_yaxis_transform(), rotation=270,
                         va="center", ha="left", fontsize=6.3, color=INK2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig4_ablation")


# ---------------------------------------------------------------- Fig. 5：队列筛选

def fig5():
    """上排：同一批预测在剔除后的子集上评估，相对零分布（a 宏平均 AUROC，b 均衡准确率）；
    下排：用剔除后的队列训练、在完整测试折上评估——c 逆转（两种打分）与转化 AUROC 的变化，d 平均预测逆转概率与实际比例。"""
    from matplotlib.gridspec import GridSpec
    null = pd.read_csv(TAB / "t3_curation_null.csv")
    ev = pd.read_csv(TAB / "t3_curation_eval.csv")
    tr = pd.read_csv(TAB / "t3_curation_train.csv")
    hb = pd.read_csv(TAB / "s_curation_headB_test.csv")
    models = ["GBM-aggregate", "LongForMAD", "GRU", "CTMT"]
    fig = plt.figure(figsize=(W, 4.3))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.15], hspace=0.95, wspace=0.08)
    top = [fig.add_subplot(gs[0, 0])]
    top.append(fig.add_subplot(gs[0, 1], sharey=top[0]))
    for k, (ax, metric, title) in enumerate(zip(top, ["macro_auroc", "bal_acc"], ["Macro-averaged AUROC", "Balanced accuracy"])):
        for i, m in enumerate(models):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#f4f3ef", lw=0, zorder=0)
            e = ev[(ev.model == m) & (ev.metric == metric)].iloc[0]
            for off, kind, col in [(-0.22, "random participants", "#cfcec5"), (0.0, "class-matched points", "#8a8983")]:
                n = null[(null.model == m) & (null.metric == metric) & (null.null == kind)].iloc[0]
                ax.plot([n.null_lo, n.null_hi], [i + off, i + off], color=col, lw=3.0, solid_capstyle="butt", zorder=2)
            ax.errorbar(e["diff"], i + 0.24, xerr=[[e["diff"] - e["ci_low"]], [e["ci_high"] - e["diff"]]],
                        fmt=MARK[m], color=COLOR[m], ms=3.6, elinewidth=0.9, capsize=1.4, zorder=3)
        ax.axvline(0, color=REF, lw=0.6, ls="--")
        ax.set_title(title)
        ax.set_xlabel("Curated − full evaluation set")
        ax.set_yticks(range(len(models)))
        ax.tick_params(axis="y", labelleft=(k == 0))
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.grid(axis="x", color=GRID, lw=0.4)
        label(ax, "ab"[k], dx=-0.62 if k == 0 else -0.06, dy=1.04)
    top[0].set_yticklabels([NAME[m] for m in models])
    top[1].plot([], [], color="#cfcec5", lw=3.0, label="random participants removed")
    top[1].plot([], [], color="#8a8983", lw=3.0, label="class-matched points removed")
    top[1].plot([], [], "o", color=INK2, ms=3.4, label="curated subset (95% CI)")
    top[1].legend(loc="upper center", bbox_to_anchor=(-0.04, -0.36), ncol=3, columnspacing=1.4, handlelength=1.6)

    # 下排：c 训练集筛选后 AUROC 的变化（逆转用两种打分）；d 平均预测逆转概率 vs 实际
    sc = pd.read_csv(TAB / "r_reversion_scores.csv")
    rows = models + ["CTMT-classification-only"]
    trs = pd.concat([tr, hb])
    ax = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1], sharey=ax)
    obs = tr[tr.metric == "mean_P(CN)_in_MCI"]["observed_rate"].iloc[0]
    for i, m in enumerate(rows):
        if i % 2 == 0:
            for a in (ax, axd):
                a.axhspan(i - 0.5, i + 0.5, color="#f4f3ef", lw=0, zorder=0)
        col = COLOR.get(m, COLOR["CTMT"])
        mk = MARK.get(m, "o")
        g = sc[sc.variant == f"{m}[curated-train]"].set_index("metric")
        items = [(-0.22, "d_rev_PCN_CI", "full"), (0.0, "d_rev_negPdem_CI", "half"), (0.22, "conv", "none")]
        for off, key, fill in items:
            if key == "conv":
                r = trs[(trs.model == m) & (trs.metric == "mci_conv")].iloc[0]
                x, lo, hi = r["diff"], r["ci_low"], r["ci_high"]
            else:
                x, lo, hi = g.loc[key, "diff"], g.loc[key, "lo"], g.loc[key, "hi"]
            kw = dict(mfc=col) if fill == "full" else dict(mfc="white") if fill == "none" else dict(mfc=col, fillstyle="left")
            ax.errorbar(x, i + off, xerr=[[x - lo], [hi - x]], fmt=mk, color=col, ms=3.6, elinewidth=0.9, capsize=1.4,
                        zorder=3, **kw)
        f, c = g.loc["mean_P(CN)|all MCI", "full_train"], g.loc["mean_P(CN)|all MCI", "variant_train"]
        axd.plot([100 * c, 100 * f], [i, i], color=col, lw=1.0, zorder=2)
        axd.plot(100 * f, i, mk, color=col, mfc="white", ms=3.8, zorder=3)
        axd.plot(100 * c, i, mk, color=col, ms=3.8, zorder=3)
    ax.axvline(0, color=REF, lw=0.6, ls="--")
    axd.axvline(100 * obs, color=INK2, lw=0.8, ls=(0, (2, 1.5)))
    axd.text(100 * obs + 0.3, 1.5, f"observed {100 * obs:.1f}%", fontsize=5.8, color=INK2, va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([NAME.get(m, "CTMT without head B") for m in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    axd.tick_params(axis="y", labelleft=False)
    ax.set_title("Curated training: change in AUROC")
    ax.set_xlabel("ΔAUROC, curated − full training")
    axd.set_title("Predicted probability of reversion")
    axd.set_xlabel("Mean P(CN) among MCI points, %")
    axd.set_xlim(-0.3, 8.5)
    for a in (ax, axd):
        a.grid(axis="x", color=GRID, lw=0.4)
    ax.plot([], [], "o", color=INK2, ms=3.6, label="reversion AUROC, scored by P(CN)")
    ax.plot([], [], "o", color=INK2, fillstyle="left", ms=3.6, label="reversion AUROC, scored by −P(dementia)")
    ax.plot([], [], "o", color=INK2, mfc="white", ms=3.6, label="conversion AUROC")
    ax.plot([], [], "s", color=INK2, mfc="white", ms=3.6, label="d: full training (open)")
    ax.plot([], [], "s", color=INK2, ms=3.6, label="d: curated training (filled)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.62, -0.3), ncol=3, columnspacing=1.0, handlelength=1.2, fontsize=6)
    label(ax, "c", dx=-0.62, dy=1.04)
    label(axd, "d", dx=-0.06, dy=1.04)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.94, bottom=0.17)
    save(fig, "fig5_curation")


# ---------------------------------------------------------------- 补充图

def figS_missingness(seq, samples):
    mods = {"ADAS-Cog13": "ADAS13", "CDR-SB": "CDRSB", "MMSE": "MMSE", "MoCA": "MoCA", "MRI": "icv_mm3", "Diagnosis": "dx_CN"}
    # 按参与者-日期去重：同一个就诊日会出现在多个预测点的历史中，只计一次
    u = seq.merge(samples[["sample_id", "subject_id"]], on="sample_id").drop_duplicates(["subject_id", "date"])
    obs = pd.DataFrame({k: u[v].notna() for k, v in mods.items()})
    combo = obs.apply(lambda r: " + ".join(k for k in mods if r[k]) or "none", axis=1)
    vc = combo.value_counts()
    vc.to_csv(TAB / "s_modality_combinations.csv")
    top = vc.head(10)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"width_ratios": [1, 1.6]})
    rate = obs.mean()[::-1] * 100
    axes[0].barh(rate.index, rate.values, color="#2a78d6", height=0.6)
    for i, v in enumerate(rate.values):
        axes[0].text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=6)
    axes[0].set_xlim(0, 100); axes[0].set_xlabel(f"Assessment days with the measure (%)")
    sh = top.values / len(u) * 100
    axes[1].barh(range(len(top)), sh, color="#2a78d6", height=0.6)
    axes[1].set_yticks(range(len(top))); axes[1].set_yticklabels(top.index, fontsize=5.6)
    for i, v in enumerate(sh):
        axes[1].text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=6)
    axes[1].set_xlim(0, sh.max() * 1.25)
    axes[1].set_xlabel(f"Share of {len(u):,} unique participant-days (%)")
    label(axes[0], "a", dx=-0.4); label(axes[1], "b", dx=-1.05)
    fig.tight_layout(w_pad=1.0)
    save(fig, "figS1_missingness")


def figS_roc(preds, samples):
    models = ["Logistic", "GBM-snapshot", "GBM-aggregate", "LongForMAD", "GRU", "CTMT"]
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.0))
    for row, (setting, q) in enumerate([("Held-out test set", preds.eval_set == "test"), ("CV (first repetition)", preds.repeat == 0)]):
        for k, (ax, t) in enumerate(zip(axes[row], TASKS)):
            ax.plot([0, 1], [0, 1], color=REF, lw=0.6, ls="--")
            for m in models:
                d = R.join(preds[q & (preds.model == m)], samples)
                yb, s, _ = task_arrays(d["y"], d[R.PROB].to_numpy(), d["origin_mci"], t)
                f, tp, _ = roc_curve(yb, s)
                ax.plot(f, tp, color=COLOR[m], ls=STYLE[m], lw=1.0, label=NAME[m])
            short = "Test set" if row == 0 else "CV, repetition 1"
            ax.set_title(f"{TASKS[t]}\n{short}: {int(yb.sum())}/{len(yb):,} events", fontsize=6.0)
            ax.set_aspect("equal"); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
            if k:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("1 − specificity")
            ax.text(-0.16, 1.2, chr(97 + row * 3 + k), transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="bottom", ha="left")
        axes[row][0].set_ylabel("Sensitivity")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "figS2_roc")


def figS_calibration(preds, samples):
    models = ["Logistic", "GBM-snapshot", "GBM-aggregate", "LongForMAD", "GRU", "CTMT"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.1))
    for k, (ax, t) in enumerate(zip(axes, TASKS)):
        lim = 0.0
        for m in models:
            d = R.join(preds[(preds.model == m) & preds.repeat.notna()], samples)
            yb, s, _ = task_arrays(d["y"], d[R.PROB].to_numpy(), d["origin_mci"], t)
            qq = pd.qcut(s, 10, duplicates="drop")
            g = pd.DataFrame({"s": s, "y": yb}).groupby(qq, observed=True).mean()
            ax.plot(g["s"], g["y"], color=COLOR[m], ls=STYLE[m], marker=MARK[m], ms=2.4, lw=0.9, label=NAME[m])
            lim = max(lim, g["s"].max(), g["y"].max())
        lim = min(1, lim * 1.08)
        ax.plot([0, lim], [0, lim], color=REF, lw=0.6, ls="--")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.set_title(TASKS[t]); ax.set_xlabel("Predicted probability")
        ax.text(-0.14, 1.03, chr(97 + k), transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left")
    axes[0].set_ylabel("Observed proportion")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "figS3_calibration")


def main(parts):
    samples, seq = load_dataset(ROOT)
    want = lambda k: not parts or k in parts  # noqa: E731
    if want("fig1"):
        fig1(samples, seq)
    if want("figS1"):
        figS_missingness(seq, samples)
    if any(want(k) for k in ["fig2", "fig3", "fig4", "fig5", "figS2", "figS3"]):
        preds = R.load_predictions(ROOT)
        for k, fn in [("fig2", fig2), ("fig3", fig3), ("fig4", fig4), ("fig5", fig5)]:
            if want(k):
                fn()
        if want("figS2"):
            figS_roc(preds, samples)
        if want("figS3"):
            figS_calibration(preds, samples)


if __name__ == "__main__":
    main(sys.argv[1:])
