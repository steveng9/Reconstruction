"""The epsilon-sweep figures for the two further quasi-identifier sets.

Draws, for each of QI_large and QI_behavioral, the same two figures that
`fig_eps_curves_full.py` draws for QI_demo, in the same style and palette:

  fig_eps_curves_otherqi.pdf         mean of the three attacks; rows = QI, columns = Adult 10k, CDC 1k
  fig_eps_curves_perattack_<qi>.pdf  one panel per attack, one row per dataset

They replace the appendix tables tab:eps_sweep and tab:eps_full.

AIM is not swept above eps=10 (Adult) or eps=30 (CDC) on these QIs, so its curve
stops there. A pre-binned AIM release (`AIM_eps1_pb`) replaces the SmartNoise one
only for the (dataset, QI) cells where it was attacked; on CDC no pre-binned
release was attacked under these QIs, and there the two pipelines differ by at
most 0.2 pp under QI_demo.

Reads only results.db (RECON_RESULTS_DB overrides the path).

  python experiment_scripts/fig_eps_curves_qi.py [--out DIR]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--out", default=str(HERE.parent / "outfiles" / "figures"),
                    help="output directory (default: outfiles/figures)")
args = parser.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
os.environ["RECON_TABLE_OUT"] = str(OUT)          # must precede the import

import regen_camera_tables as R                    # noqa: E402

import matplotlib; matplotlib.use("Agg")           # noqa: E402
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.lines import Line2D                # noqa: E402

DATASETS = [("adult", 10000, "Adult (10k rows)"), ("cdc_diabetes", 1000, "CDC Diabetes (1k rows)")]
QIS = [("QI_large", "qilarge"), ("QI_behavioral", "qibehavioral")]
QI_LABEL = {"QI_large": r"QI$_{\mathrm{large}}$", "QI_behavioral": r"QI$_{\mathrm{behavioral}}$"}
PREBINNED = {"AIM_eps1_pb": ("AIM", 1.0, "AIM_eps1"), "AIM_eps10_pb": ("AIM", 10.0, "AIM_eps10")}
BASE_C, BASE_LS, BASE_LW = "#555555", (0, (6, 3)), 1.1


def apply_prebinned(df):
    drop = np.zeros(len(df), dtype=bool)
    for pb, (gen, eps, old) in PREBINNED.items():
        for (ds, size, qi), g in df[df.sdg_method == pb].groupby(["dataset", "dataset_size", "qi"]):
            cell = (df.dataset == ds) & (df.dataset_size == size) & (df.qi == qi)
            df.loc[cell & (df.sdg_method == pb), "gen"] = gen
            df.loc[cell & (df.sdg_method == pb), "eps"] = eps
            drop |= (cell & (df.sdg_method == old)).to_numpy()
    return df[~drop]


def mode_baseline(df, ds, size, qi):
    s = df[(df.dataset == ds) & (df.dataset_size == size) & (df.qi == qi)
           & (df.split == "standard") & (df.attack_label == "Mode")]
    return float(s.groupby("sample").ra_mean.mean().mean())


def curve(d, gen, **kw):
    xs, ys, es = [], [], []
    for e in R.EPS:
        m, c, n = R.cell(d, gen=gen, eps=e, **kw)
        if n:
            xs.append(e); ys.append(m); es.append(0 if pd.isna(c) else c)
    return xs, ys, es


def legend_handles(ax):
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([0], [0], color=BASE_C, linestyle=BASE_LS, linewidth=BASE_LW))
    l.append("Mode baseline")
    return h, l


def decorate(ax, base):
    ax.axhline(base, color=BASE_C, linestyle=BASE_LS, linewidth=BASE_LW, zorder=2)
    ax.set_xscale("log")
    ax.grid(alpha=.25, linewidth=.5, zorder=0)
    ax.axvline(10, color="#666666", linestyle=":", linewidth=1, zorder=1)


def main():
    print(f"results.db: {R.DB}")
    df = apply_prebinned(R.load_runs())
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    # ---- one figure for the mean of the three attacks: rows = QI, columns = dataset
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    for r, (qi, _) in enumerate(QIS):
        for c_, (ds, size, title) in enumerate(DATASETS):
            ax = axes[r][c_]
            d = R._eps_frame(df, ds, size, qi)
            base = mode_baseline(df, ds, size, qi)
            print(f"\n--- {qi} {title} (Mode {base:.1f}) ---")
            for gen in R.DP_GENS:
                xs, ys, es = curve(d, gen)
                if not xs:
                    continue
                print("   %-11s %s" % (gen, "  ".join(f"{e:g}:{y:.1f}" for e, y in zip(xs, ys))))
                ax.errorbar(xs, ys, yerr=es, label=R.DP_LABEL[gen], color=R.PALETTE[gen],
                            marker=R.MARKERS[gen], markersize=3.6, linewidth=1.6, capsize=1.8,
                            elinewidth=0.8, zorder=3)
            decorate(ax, base)
            if r == 0:
                ax.set_title(title, fontsize=8.5, pad=4)
            if r == 1:
                ax.set_xlabel(r"privacy budget $\varepsilon$")
            if c_ == 0:
                ax.set_ylabel(QI_LABEL[qi] + "\n" + r"$R_{adv}$ (%)", fontsize=7.5)
    fig.legend(*legend_handles(axes[0][0]), frameon=False, fontsize=7, loc="upper center",
               ncol=7, handlelength=1.6, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_eps_curves_otherqi.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- one per-attack figure per QI: rows = dataset, columns = attack
    for qi, tag in QIS:
        base = {ds: mode_baseline(df, ds, sz, qi) for ds, sz, _ in DATASETS}
        fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True, sharey="row")
        for r, (ds, size, title) in enumerate(DATASETS):
            d = R._eps_frame(df, ds, size, qi)
            for c_, atk in enumerate(R.ATT3):
                ax = axes[r][c_]
                for gen in R.DP_GENS:
                    xs, ys, es = curve(d, gen, attack_label=atk)
                    if not xs:
                        continue
                    ax.errorbar(xs, ys, yerr=es, label=R.DP_LABEL[gen], color=R.PALETTE[gen],
                                marker=R.MARKERS[gen], markersize=3, linewidth=1.4,
                                capsize=1.5, elinewidth=.7, zorder=3)
                decorate(ax, base[ds])
                if r == 0:
                    ax.set_title(R.ATT3_LABEL[atk].replace(r"$^\dagger$", ""), fontsize=8.5, pad=4)
                if c_ == 0:
                    ax.set_ylabel(f"{title}\n" + r"$R_{adv}$ (%)", fontsize=7.5)
                if r == 1:
                    ax.set_xlabel(r"$\varepsilon$")
        fig.legend(*legend_handles(axes[0][0]), frameon=False, fontsize=7, loc="upper center",
                   ncol=7, handlelength=1.6, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94))
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"fig_eps_curves_perattack_{tag}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print("\nwrote to", OUT)


main()
