"""Camera-ready build of fig:eps_curves and the per-attack appendix figure.

Differs from `regen_camera_tables.figure_eps` in three ways:

  1. AIM uses the pre-binned releases end to end. The Feb-2026 `AIM_eps1`
     (and, on CDC, `AIM_eps10`) releases went through SmartNoise's private
     preprocessor instead; they were re-generated with continuous columns
     pre-binned as `AIM_eps1_pb` / `AIM_eps10_pb`. Those suffixed names do not
     parse as an epsilon, so they are mapped onto (AIM, eps) here and the
     superseded rows dropped -- the epsilon trend then rests on one pipeline.
  2. A dashed Mode-baseline rule is drawn in every panel.
  3. The per-attack figure shares its y axis within a row only: the two
     datasets sit at different R_adv levels, and one shared range flattens
     both rows.

The script refuses to draw a curve with a missing budget. AIM's high-budget
points (Adult eps>=30, CDC eps>=100) were added to results.db in Sep 2026; an
older database would otherwise produce a truncated AIM curve with no warning.

Reads only results.db (RECON_RESULTS_DB overrides the path). Writes nothing
into expected_output/.

  python experiment_scripts/fig_eps_curves_full.py [--out DIR]
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
parser.add_argument("--allow-missing", action="store_true",
                    help="draw even if some (generator, epsilon) cell has no runs")
args = parser.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
os.environ["RECON_TABLE_OUT"] = str(OUT)          # must precede the import

import regen_camera_tables as R                    # noqa: E402

import matplotlib; matplotlib.use("Agg")           # noqa: E402
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.lines import Line2D                # noqa: E402

PANELS = [("adult", 10000, "QI1", "Adult (10k rows)"),
          ("cdc_diabetes", 1000, "QI1", "CDC Diabetes (1k rows)")]

# pre-binned release -> (gen, eps, superseded SmartNoise release)
PREBINNED = {("adult", 10000, "AIM_eps1_pb"):        ("AIM", 1.0,  "AIM_eps1"),
             ("cdc_diabetes", 1000, "AIM_eps1_pb"):  ("AIM", 1.0,  "AIM_eps1"),
             ("cdc_diabetes", 1000, "AIM_eps10_pb"): ("AIM", 10.0, "AIM_eps10")}

BASE_C, BASE_LS, BASE_LW = "#555555", (0, (6, 3)), 1.1


def apply_prebinned(df):
    drop = np.zeros(len(df), dtype=bool)
    for (ds, size, pb), (gen, eps, old) in PREBINNED.items():
        here = (df.dataset == ds) & (df.dataset_size == size)
        m = here & (df.sdg_method == pb)
        if not m.any():
            sys.exit(f"{pb} has no rows for {ds} {size}: this results.db predates the "
                     "pre-binned AIM releases")
        df.loc[m, "gen"] = gen
        df.loc[m, "eps"] = eps
        drop |= (here & (df.sdg_method == old)).to_numpy()
        print(f"  {ds} {size}: {pb} -> (AIM, eps={eps:g}); dropping {old}")
    return df[~drop]


def mode_baseline(df, ds, size, qi):
    """Mode is SDG-independent; average it over every generator present."""
    s = df[(df.dataset == ds) & (df.dataset_size == size) & (df.qi == qi)
           & (df.split == "standard") & (df.attack_label == "Mode")]
    return float(s.groupby("sample").ra_mean.mean().mean())


def curve(d, gen, **kw):
    xs, ys, es, ns = [], [], [], []
    for e in R.EPS:
        m, c, n = R.cell(d, gen=gen, eps=e, **kw)
        if n:
            xs.append(e); ys.append(m); es.append(0 if pd.isna(c) else c); ns.append(n)
    return xs, ys, es, ns


def check_complete(df):
    missing = []
    for ds, size, qi, _ in PANELS:
        d = R._eps_frame(df, ds, size, qi)
        for gen in R.DP_GENS:
            for atk in R.ATT3:
                have = set(d[(d.gen == gen) & (d.attack_label == atk)].eps)
                missing += [f"{ds} {gen} {atk} eps={e:g}" for e in R.EPS if e not in have]
    if missing:
        print(f"{len(missing)} empty cells, e.g.:\n  " + "\n  ".join(missing[:10]))
        if not args.allow_missing:
            sys.exit("refusing to draw incomplete curves (pass --allow-missing to override)")


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
    check_complete(df)
    base = {ds: mode_baseline(df, ds, sz, qi) for ds, sz, qi, _ in PANELS}
    for ds, v in base.items():
        print(f"Mode baseline {ds}: {v:.2f}")

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})

    # ---- main figure: mean over the three attacks --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharex=True)
    for ax, (ds, size, qi, title) in zip(axes, PANELS):
        d = R._eps_frame(df, ds, size, qi)
        print(f"\n--- {title} (mean of {', '.join(R.ATT3)}) ---")
        for gen in R.DP_GENS:
            xs, ys, es, ns = curve(d, gen)
            if not xs:
                continue
            print("   %-11s %s" % (gen, "  ".join(f"{e:g}:{y:.1f}(n{n})"
                                                  for e, y, n in zip(xs, ys, ns))))
            ax.errorbar(xs, ys, yerr=es, label=R.DP_LABEL[gen], color=R.PALETTE[gen],
                        marker=R.MARKERS[gen], markersize=3.6, linewidth=1.6, capsize=1.8,
                        elinewidth=0.8, zorder=3)
        decorate(ax, base[ds])
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.set_xlabel(r"privacy budget $\varepsilon$")
    axes[0].set_ylabel(r"$R_{adv}$ (%)")
    axes[1].legend(*legend_handles(axes[1]), frameon=False, fontsize=7, loc="lower right",
                   ncol=2, handlelength=1.6)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_eps_curves.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- per-attack figure: y shared within a row, not across datasets ------
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True, sharey="row")
    for r, (ds, size, qi, title) in enumerate(PANELS):
        d = R._eps_frame(df, ds, size, qi)
        for c_, atk in enumerate(R.ATT3):
            ax = axes[r][c_]
            for gen in R.DP_GENS:
                xs, ys, es, _ = curve(d, gen, attack_label=atk)
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
    # the rows have different ranges, so no panel is guaranteed to have free space;
    # the legend gets its own band above the grid
    fig.legend(*legend_handles(axes[0][0]), frameon=False, fontsize=7, loc="upper center",
               ncol=7, handlelength=1.6, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_eps_curves_perattack.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nwrote fig_eps_curves{,_perattack}.{pdf,png} to", OUT)


main()
