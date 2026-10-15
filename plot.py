"""Poster figures + summary table from results.jsonl.

  python plot.py --inp results/results.jsonl --outdir figures

Writes (PDF for print, PNG @300dpi for slides/preview):
  fig1_acc_vs_eps    headline curve: accuracy vs epsilon, mean +/- std over seeds, non-private baseline
  fig2_clip          clipping-norm sensitivity at eps~2
  fig3_cost          compute cost: wall time & peak memory, baseline vs DP (two panels, never dual-axis)
  summary.csv / summary.md
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
DP, BASE = "#2a78d6", "#8a8984"          # categorical slot 1; neutral for the reference line
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 19, "axes.titlesize": 22, "axes.labelsize": 19,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1, "axes.axisbelow": True, "figure.dpi": 100, "savefig.bbox": "tight",
})


PREVIEW = False
NOTITLE = False


def save(fig, outdir, name):
    if PREVIEW:
        fig.text(0.5, 0.5, "PREVIEW - SYNTHETIC NUMBERS", ha="center", va="center", rotation=20,
                 fontsize=40, color="#e34948", alpha=0.25, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="results/results.jsonl")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--clip_inp", default="", help="optional jsonl to draw the clipping figure from (default: --inp)")
    p.add_argument("--lr_extra", nargs="*", default=[], help="jsonl files with DP runs at an alternative LR (seed 0)")
    p.add_argument("--no_titles", action="store_true", help="omit chart titles (poster has section headers)")
    p.add_argument("--preview", action="store_true", help="watermark figures (synthetic data)")
    a = p.parse_args()
    global PREVIEW, NOTITLE; PREVIEW = a.preview; NOTITLE = a.no_titles
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.DataFrame([json.loads(l) for l in open(a.inp) if l.strip()])
    df["acc_pct"] = df["acc"] * 100

    main_df = df[df.tag == "main"]
    base = main_df[~main_df.private]
    dp = main_df[main_df.private]
    b_mean, b_std = base.acc_pct.mean(), base.acc_pct.std(ddof=0)
    g = dp.groupby("sigma").agg(eps=("eps", "mean"), acc=("acc_pct", "mean"), acc_sd=("acc_pct", "std"),
                                n=("seed", "count"), wall=("wall_s", "mean"), mem=("peak_mem_gb", "mean"))
    g = g.fillna({"acc_sd": 0}).sort_values("eps")

    # ---- fig 1: the curve
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axhspan(b_mean - b_std, b_mean + b_std, color=BASE, alpha=0.12, lw=0)
    ax.axhline(b_mean, color=BASE, ls="--", lw=2)
    ax.text(g.eps.min() / 1.3, b_mean + 0.35, f"Non-private LoRA baseline: {b_mean:.1f}%", va="bottom", ha="left",
            color=INK2, fontsize=18)
    ax.errorbar(g.eps, g.acc, yerr=g.acc_sd, color=DP, lw=2, marker="o", ms=10, capsize=5,
                mec="white", mew=2, label="DP-SGD + LoRA (Opacus)")
    for e, acc in zip(g.eps, g.acc):
        ax.annotate(f"{acc:.1f}%\n(−{b_mean-acc:.1f} pts)", (e, acc), textcoords="offset points",
                    xytext=(0, -58), ha="center", color=INK, fontsize=16)
    ax.set_xscale("log")
    ax.set_xticks(g.eps); ax.set_xticklabels([f"{e:.2g}" for e in g.eps])
    ax.minorticks_off()
    ax.set_xlim(g.eps.min() / 1.4, g.eps.max() * 1.5)
    lo = min(g.acc.min() - g.acc_sd.max(), b_mean) - 6
    ax.set_ylim(max(lo, 45), min(100, b_mean + 3))
    ax.set_xlabel(f"Privacy budget ε  (δ = {dp.delta.iloc[0]:g}; smaller = more private)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.legend(loc="lower right", frameon=False)
    if not NOTITLE: ax.set_title("Accuracy vs. ε", loc="left", color=INK, fontweight="bold")
    save(fig, a.outdir, "fig1_acc_vs_eps")

    # ---- fig 2: clipping sensitivity at sigma~0.954
    cdf = df
    if a.clip_inp:
        cdf = pd.DataFrame([json.loads(l) for l in open(a.clip_inp) if l.strip()]); cdf["acc_pct"] = cdf["acc"] * 100
    clip = cdf[(cdf.private) & (np.isclose(cdf.sigma, 0.954)) & (cdf.seed == 0)].groupby("clip").acc_pct.mean()
    if len(clip) > 1:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.plot(clip.index, clip.values, color=DP, lw=2, marker="o", ms=10, mec="white", mew=2)
        best = clip.idxmax()
        for c, v in clip.items():
            ax.annotate(f"{v:.1f}%", (c, v), textcoords="offset points", xytext=(0, 12), ha="center",
                        color=INK, fontsize=17, fontweight="bold" if c == best else "normal")
        ax.set_xscale("log"); ax.set_xlim(clip.index.min() / 1.6, clip.index.max() * 1.6); ax.set_xticks(clip.index); ax.set_xticklabels([f"{c:g}" for c in clip.index])
        ax.minorticks_off()
        ax.set_xlabel("Clipping norm C (log scale)"); ax.set_ylabel("Validation accuracy (%)")
        ax.set_ylim(clip.min() - 5, max(clip.max() + 4, b_mean))
        eps2 = cdf[(cdf.private) & np.isclose(cdf.sigma, 0.954)].eps.mean()
        if not NOTITLE: ax.set_title(f"Clipping-norm sensitivity at ε ≈ {eps2:.1f}", loc="left", fontweight="bold")
        save(fig, a.outdir, "fig2_clip")

    # ---- fig 3: compute cost (two panels, shared categories, no dual axis)
    cats = ["Non-private", "DP-SGD"]
    wall = [base.wall_s.mean() / 60, dp.wall_s.mean() / 60]
    mem = [base.peak_mem_gb.mean(), dp.peak_mem_gb.mean()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, vals, lab, fmt in [(axes[0], wall, "Wall time per run (min)", "{:.1f} min"),
                               (axes[1], mem, "Peak GPU memory (GB)", "{:.1f} GB")]:
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
            ax.text(0.5, 0.5, "n/a (CPU run)", ha="center", transform=ax.transAxes, color=INK2); continue
        bars = ax.bar(cats, vals, color=[BASE, DP], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", color=INK)
        ax.set_title(lab, loc="left", fontsize=19); ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.25)
    n_runs = len(df)
    total_h = df.wall_s.sum() / 3600
    fig.suptitle(f"Cost of the sweep: {n_runs} runs, {total_h:.1f} GPU-hours "
                 f"(DP run ≈ {wall[1]/wall[0]:.1f}× a non-private run)", x=0.02, ha="left",
                 fontsize=20, fontweight="bold")
    fig.tight_layout()
    save(fig, a.outdir, "fig3_cost")

    # ---- fig 4: learning-rate sensitivity across eps (seed 0, fixed C)
    if a.lr_extra:
        alt = pd.DataFrame([json.loads(l) for f in a.lr_extra for l in open(f) if l.strip()])
        alt = alt[alt.private]
        alt["acc_pct"] = alt["acc"] * 100
        main_lr = dp.lr.iloc[0]
        fig, ax = plt.subplots(figsize=(10, 5.0))
        ax.axhline(50.9, color=GRID, lw=2, ls=":")
        ax.text(g.eps.min() / 1.3, 51.8, "chance (majority class)", color=INK2, fontsize=15)
        series = [(dp, f"LR {main_lr:g} (used)", DP)]
        for lr_alt, sub in alt.groupby("lr"):
            series.append((sub, f"LR {lr_alt:g}", "#eb6834"))
        for k, (sub, lab, col) in enumerate(series):
            gg = sub.groupby("sigma").agg(eps=("eps", "mean"), m=("acc_pct", "mean"), sd=("acc_pct", "std")).fillna(0).sort_values("eps")
            ax.errorbar(gg.eps, gg.m, yerr=gg.sd, color=col, lw=2, marker="o", ms=10, mec="white", mew=2, capsize=5, label=lab)
        ax.legend(loc="lower right", frameon=False, fontsize=17)
        ax.set_xscale("log"); ax.set_xticks(g.eps); ax.set_xticklabels([f"{e:.2g}" for e in g.eps]); ax.minorticks_off()
        ax.set_xlim(g.eps.min() / 1.6, g.eps.max() * 1.4)
        lo_ = min(alt.acc_pct.min(), dp.acc_pct.min())
        ax.set_ylim(45 if lo_ < 60 else lo_ - 3, 96 if lo_ < 60 else min(100, max(dp.acc_pct.max(), alt.acc_pct.max()) + 2))
        if lo_ >= 60: [l.remove() for l in ax.texts if "chance" in l.get_text()]; [l.remove() for l in ax.lines if l.get_linestyle() == ":"]
        ax.set_xlabel("Privacy budget ε"); ax.set_ylabel("Validation accuracy (%)")
        if not NOTITLE: ax.set_title("Same model, two learning rates", loc="left", fontweight="bold")
        save(fig, a.outdir, "fig4_lr")
        alt[["lr", "sigma", "eps", "acc_pct"]].sort_values("eps").to_csv(os.path.join(a.outdir, "lr_sensitivity.csv"), index=False)

    # ---- summary table
    rows = [{"setting": "non-private", "sigma": 0, "eps": "∞", "acc_mean": round(b_mean, 2), "acc_std": round(b_std, 2),
             "gap_pts": 0.0, "seeds": len(base), "wall_min": round(base.wall_s.mean() / 60, 1),
             "peak_mem_gb": base.peak_mem_gb.mean()}]
    for s, r in g.iterrows():
        rows.append({"setting": "DP-SGD", "sigma": s, "eps": round(r.eps, 2), "acc_mean": round(r.acc, 2),
                     "acc_std": round(r.acc_sd, 2), "gap_pts": round(b_mean - r.acc, 2), "seeds": int(r.n),
                     "wall_min": round(r.wall / 60, 1), "peak_mem_gb": r.mem})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.outdir, "summary.csv"), index=False)
    with open(os.path.join(a.outdir, "summary.md"), "w") as f:
        f.write(t.to_markdown(index=False))
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
