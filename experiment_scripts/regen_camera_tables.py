#!/usr/bin/env python
"""
Regenerate every camera-ready table body from results.db (the source of truth).

Emits LaTeX table bodies + a stats report to manuscript/camera_ready/generated/.
Nothing here edits the manuscript; each output is a drop-in replacement body that
the author pastes (or \input's) into manuscript_CAMERA.tex.

Provenance rule: every number printed by this script comes from results.db or from
a named post-repair CSV. No number is carried over from the .tex.

    python experiment_scripts/regen_camera_tables.py
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib
for _anc in _pathlib.Path(__file__).resolve().parents:
    if (_anc / "paths.py").exists():
        _sys.path.insert(0, str(_anc))
        break
from paths import REPO_ROOT

import os, sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(str(REPO_ROOT))
DB   = Path(os.environ.get("RECON_RESULTS_DB", ROOT / "experiment_scripts" / "results.db"))
# Reviewers get the tables in a tracked directory alongside the committed
# reference copies, so a run can be diffed against the numbers in the paper.
# The author's manuscript build overrides this with RECON_TABLE_OUT.
OUT  = Path(os.environ.get("RECON_TABLE_OUT", ROOT / "expected_output" / "tables"))
OUT.mkdir(parents=True, exist_ok=True)

EPS = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
DP_GENS = ["MST", "AIM", "PrivBayes", "PrivSyn", "MWEMPGM", "PrivateGSD"]
DP_LABEL = {"MST": "MST", "AIM": "AIM", "PrivBayes": "PrivBayes", "PrivSyn": "PrivSyn",
            "MWEMPGM": "MWEM-PGM", "PrivateGSD": "PrivateGSD"}
ATT3 = ["RandomForest", "NaiveBayes", "CoBP-RA"]
ATT3_LABEL = {"RandomForest": "Random Forest", "NaiveBayes": "Naive Bayes",
              "CoBP-RA": r"CoBP-RA$^\dagger$"}

def g(e):  return f"{e:g}"

def load_runs() -> pd.DataFrame:
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT dataset,dataset_size,sample,qi,sdg_method,attack_label,split,ra_mean,run_id,confidence FROM runs", con)
    con.close()
    # `MarginalRF` is the pre-2026-06 label for `CoBP-RA` (same attack). 92 cells carry
    # both labels from separate runs; prefer the post-rename row and drop the duplicate
    # so a cell is never double-counted in a mean.
    df["attack_label"] = df.attack_label.replace({"MarginalRF": "CoBP-RA"})
    key = ["dataset", "dataset_size", "sample", "qi", "sdg_method", "attack_label", "split"]
    df = df.sort_values("run_id").drop_duplicates(key, keep="last")
    df["gen"] = df.sdg_method.str.replace(r"_eps.*", "", regex=True)
    df["eps"] = pd.to_numeric(df.sdg_method.str.extract(r"_eps([0-9.]+)$")[0], errors="coerce")
    return df

def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2: return np.nan
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))

def cell(df, **kw):
    """Mean over samples for one fully-specified cell; returns (mean, ci, n)."""
    s = df
    for k, v in kw.items():
        s = s[s[k] == v]
    if s.empty: return (np.nan, np.nan, 0)
    per_sample = s.groupby("sample").ra_mean.mean()
    return (per_sample.mean(), ci95(per_sample), len(per_sample))

def f1(v):  return "---" if pd.isna(v) else f"{v:.1f}"
def f2(v):  return "---" if pd.isna(v) else f"{v:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — tab:ra_mean_adult   (Adult 10k, QI1/QI_demo, split=standard)
# Column set is UNCHANGED from the printed table: adding the four new DP
# generators here would need 4-24 more columns and crush the layout. The new
# generators live in the eps float (Table 4) and the figure instead.
# ─────────────────────────────────────────────────────────────────────────────
T1_COLS = [("RankSwap","RankSwap"),("CellSuppression","Cell Supp."),("Synthpop","Synthpop"),
           ("TVAE","TVAE"),("CTGAN","CTGAN"),("ARF","ARF"),("TabDDPM","TabDDPM"),
           ("MST_eps0.1",r"MST $\varepsilon{=}0.1$"),("MST_eps1",r"MST $\varepsilon{=}1$"),
           ("MST_eps10",r"MST $\varepsilon{=}10$"),("MST_eps100",r"MST $\varepsilon{=}100$"),
           ("MST_eps1000",r"MST $\varepsilon{=}1000$"),("AIM_eps1",r"AIM $\varepsilon{=}1$")]
# Paper row name -> attack_label as stored in results.db. The DB kept the
# pre-2026-06 names; `Copy` is stored as `MeasureDeid`. Verified by reproducing
# the printed Copy row exactly (RankSwap 49.3 / CellSupp 10.3 / TabDDPM 10.5).
T1_ROWS = [("__grp__", "Reference points (not attacks)"),
           ("Mode","Mode"),("Random","Random"),("MeasureDeid","Copy"),
           ("__grp__", r"Each feature in isolation: classifiers \& column-marginal"),
           ("KNN",r"\textsc{knn}"),("NaiveBayes","Naive Bayes"),("LogisticRegression","Logistic Regression"),
           ("SVM",r"\textsc{svm}"),("RandomForest","Random Forest"),("LightGBM","LightGBM"),
           ("MLP",r"\textsc{mlp}"),("TabPFN","TabPFN"),
           ("__grp__", "Feature-correlated: autoregressive"),
           ("Attention",r"ARFFormer$^\dagger$"),
           ("__grp__", "Feature-correlated: row-wise message passing"),
           ("CoBP-RA",r"CoBP-RA$^\dagger$"),
           ("__grp__", "Feature-correlated: joint generative conditioning"),
           ("JointMLP",r"MultiHeadMLP$^\dagger$"),("PartialMST",r"CondMST$^\dagger$"),
           ("TabDDPM",r"CondDDPM$^\dagger$"),("ConditionedRePaint",r"CondRePaint$^\dagger$")]

def table1(df):
    d = df[(df.dataset=="adult")&(df.dataset_size==10000)&(df.qi=="QI1")&(df.split=="standard")]
    lines, grid = [], {}
    for atk, lab in T1_ROWS:
        if atk == "__grp__":
            lines.append(rf"\multicolumn{{15}}{{l}}{{\emph{{{lab}}}}}\\"); continue
        vals = [cell(d, attack_label=atk, sdg_method=s)[0] for s, _ in T1_COLS]
        grid[atk] = vals
        avg = np.nanmean(vals) if not all(pd.isna(v) for v in vals) else np.nan
        lines.append(f"{lab} & " + " & ".join(f1(v) for v in vals) + f" & {f1(avg)} " + r"\\")
    colavg = [np.nanmean([grid[a][i] for a in grid if a not in ("Mode","Random","Copy")]) for i in range(len(T1_COLS))]
    body = "\n".join(lines) + "\n" + r"\midrule" + "\n" + \
           r"\textbf{Avg.} & " + " & ".join(rf"\textbf{{{f1(v)}}}" for v in colavg) + r" & \\"
    (OUT/"table1_ra_mean_adult.tex").write_text(body + "\n")
    # coverage report: which cells are missing / thin
    cov = []
    for atk, lab in T1_ROWS:
        if atk == "__grp__": continue
        for s, sl in T1_COLS:
            m, c, n = cell(d, attack_label=atk, sdg_method=s)
            if n < 5: cov.append((lab, sl, n))
    return grid, cov


# ─────────────────────────────────────────────────────────────────────────────
# Table 4 — the epsilon float, now six generators (tab:mst_eps_sweep -> tab:eps_sweep)
# Variant (a) full: 6 generators x 3 attacks x 2 QI, all 9 budgets.
# Variant (compact): 6 generators x 2 QI, cell = mean of the 3 attacks.
# AIM stops at eps=10 (adult) / 30 (cdc): the mechanism's runtime and memory blow
# up beyond that. Those cells print "---" and that absence is a finding, not a gap.
# ─────────────────────────────────────────────────────────────────────────────
def _eps_frame(df, dataset, size, qi):
    d = df[(df.dataset==dataset)&(df.dataset_size==size)&(df.qi==qi)&(df.split=="standard")
           &(df.attack_label.isin(ATT3))&(df.gen.isin(DP_GENS))&df.eps.notna()]
    return d

def table4(df):
    full, compact = [], []
    for qi, qilab in [("QI_large", r"$\text{QI}_{\text{large}}$ (10 demographic features)"),
                      ("QI_behavioral", r"$\text{QI}_{\text{behavioral}}$ (6 behavioral/financial features)")]:
        d = _eps_frame(df, "adult", 10000, qi)
        full.append(rf"\multicolumn{{10}}{{l}}{{\emph{{{qilab}}}}}\\")
        compact.append(rf"\multicolumn{{10}}{{l}}{{\emph{{{qilab}}}}}\\")
        for atk in ATT3:
            full.append(rf"\multicolumn{{10}}{{l}}{{\quad {ATT3_LABEL[atk]}}}\\")
            for gen in DP_GENS:
                vals = [cell(d, gen=gen, eps=e, attack_label=atk)[0] for e in EPS]
                full.append(rf"\quad {DP_LABEL[gen]} & " + " & ".join(f1(v) for v in vals) + r" \\")
        for gen in DP_GENS:
            vals = [cell(d, gen=gen, eps=e)[0] for e in EPS]
            compact.append(f"{DP_LABEL[gen]} & " + " & ".join(f1(v) for v in vals) + r" \\")
    (OUT/"table4_eps_sweep_full.tex").write_text("\n".join(full) + "\n")
    (OUT/"table4_eps_sweep_compact.tex").write_text("\n".join(compact) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# The epsilon figure (variant b): risk vs budget, log-x, 95% CI error bars.
# Palette: Okabe-Ito with PrivSyn re-stepped to #4B0092 — all 15 pairs clear the
# CVD floor (worst normal dE 15.6, worst dichromat dE 11.4), verified numerically.
# Each series also carries a distinct marker, so identity never rests on hue alone.
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {"MST":"#0072B2","AIM":"#E69F00","PrivBayes":"#009E73",
           "PrivSyn":"#4B0092","MWEMPGM":"#D55E00","PrivateGSD":"#56B4E9"}
MARKERS = {"MST":"o","AIM":"s","PrivBayes":"^","PrivSyn":"D","MWEMPGM":"v","PrivateGSD":"P"}

def figure_eps(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"ps.fonttype":42})   # Type-1/TrueType, not Type-3

    panels = [("adult",10000,"QI1","Adult (10k rows)"),("cdc_diabetes",1000,"QI1","CDC Diabetes (1k rows)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharex=True)
    for ax,(ds,size,qi,title) in zip(axes, panels):
        d = _eps_frame(df, ds, size, qi)
        for gen in DP_GENS:
            xs, ys, es = [], [], []
            for e in EPS:
                m,c,n = cell(d, gen=gen, eps=e)
                if n: xs.append(e); ys.append(m); es.append(0 if pd.isna(c) else c)
            if not xs: continue
            ax.errorbar(xs, ys, yerr=es, label=DP_LABEL[gen], color=PALETTE[gen],
                        marker=MARKERS[gen], markersize=3.6, linewidth=1.6, capsize=1.8,
                        elinewidth=0.8, zorder=3)
        ax.set_xscale("log"); ax.set_title(title, fontsize=8.5, pad=4)
        ax.set_xlabel(r"privacy budget $\varepsilon$")
        ax.grid(alpha=.25, linewidth=.5, zorder=0)
        ax.axvline(10, color="#666666", linestyle=":", linewidth=1, zorder=1)
    axes[0].set_ylabel(r"$R_{adv}$ (%)")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right", ncol=2, handlelength=1.6)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf","png"):
        fig.savefig(OUT/f"fig_eps_curves.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # per-attack version (appendix): rows = dataset, cols = attack
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True)
    for r,(ds,size,qi,title) in enumerate(panels):
        d = _eps_frame(df, ds, size, qi)
        for c_,atk in enumerate(ATT3):
            ax = axes[r][c_]
            for gen in DP_GENS:
                xs,ys,es=[],[],[]
                for e in EPS:
                    m,ci,n = cell(d, gen=gen, eps=e, attack_label=atk)
                    if n: xs.append(e); ys.append(m); es.append(0 if pd.isna(ci) else ci)
                if not xs: continue
                ax.errorbar(xs,ys,yerr=es,label=DP_LABEL[gen],color=PALETTE[gen],
                            marker=MARKERS[gen],markersize=3,linewidth=1.4,capsize=1.5,elinewidth=.7,zorder=3)
            ax.set_xscale("log"); ax.grid(alpha=.25,linewidth=.5,zorder=0)
            ax.axvline(10,color="#666666",linestyle=":",linewidth=1,zorder=1)
            if r==0: ax.set_title(ATT3_LABEL[atk].replace(r"$^\dagger$",""), fontsize=8.5, pad=4)
            if c_==0: ax.set_ylabel(f"{title}\n"+r"$R_{adv}$ (%)", fontsize=7.5)
            if r==1: ax.set_xlabel(r"$\varepsilon$")
    axes[0][2].legend(frameon=False, fontsize=6.5, loc="lower right", ncol=2, handlelength=1.4)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf","png"):
        fig.savefig(OUT/f"fig_eps_curves_perattack.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 — quality profile (tab:quality_overview), extended with the new DP gens
# ─────────────────────────────────────────────────────────────────────────────
# The printed Table 2 was built from quality_results_merged.csv; it reproduces that
# table exactly, column for column, on every row whose synth was never regenerated.
# The Aug-2026 sweep (synth_quality_full_*.csv) covers the four new DP generators but
# (a) has no `wasserstein_ohe` column and (b) reports a different `sdv_col_pairs`
# (RankSwap 0.845 vs 0.929) — a different SDV metric configuration. Mixing them in one
# table would put two different measurements in the same column, so the new-generator
# rows print "---" until the merged-config metrics are recomputed for them.
QUALITY_CSV = ROOT/"experiment_scripts"/"quality_results_merged.csv"
Q_ROWS = [("MST_eps0.1",r"MST $(\varepsilon{=}0.1)$"),("MST_eps1",r"MST $(\varepsilon{=}1)$"),
          ("MST_eps10",r"MST $(\varepsilon{=}10)$"),("MST_eps1000",r"MST $(\varepsilon{=}1000)$"),
          ("AIM_eps1",r"AIM $(\varepsilon{=}1)$"),
          ("PrivBayes_eps1",r"PrivBayes $(\varepsilon{=}1)$"),("PrivBayes_eps1000",r"PrivBayes $(\varepsilon{=}1000)$"),
          ("PrivSyn_eps1",r"PrivSyn $(\varepsilon{=}1)$"),("PrivSyn_eps1000",r"PrivSyn $(\varepsilon{=}1000)$"),
          ("MWEMPGM_eps1",r"MWEM-PGM $(\varepsilon{=}1)$"),("MWEMPGM_eps1000",r"MWEM-PGM $(\varepsilon{=}1000)$"),
          ("PrivateGSD_eps1",r"PrivateGSD $(\varepsilon{=}1)$"),("PrivateGSD_eps1000",r"PrivateGSD $(\varepsilon{=}1000)$"),
          ("__mid__",""),
          ("RankSwap","RankSwap"),("CellSuppression","Cell Supp."),("Synthpop","Synthpop"),
          ("__mid__",""),
          ("TVAE","TVAE"),("CTGAN","CTGAN"),("ARF","ARF"),("TabDDPM","TabDDPM"),
          ("__mid__",""),
          ("~train_baseline",r"\textit{Train--Train}")]

def table2(df):
    q = pd.read_csv(QUALITY_CSV)
    q = q[(q.dataset=="adult")&(q.size_dir=="size_10000")]
    d = df[(df.dataset=="adult")&(df.dataset_size==10000)&(df.qi=="QI1")&(df.split=="standard")
           &(df.attack_label=="RandomForest")]
    METRICS=["tstr_ratio","mean_jsd","sdv_col_shapes","pairwise_tvd",
             "sdv_col_pairs","corr_diff","wasserstein_ohe"]
    HIGHER_IS_BETTER=[True,False,True,False,True,False,False]

    vals={}
    for meth,_ in Q_ROWS:
        if meth=="__mid__": continue
        s_=q[q.method==meth]
        vals[meth]=[float("nan")]*7 if s_.empty else [s_[c].mean() for c in METRICS]

    # best per column, excluding the real-vs-real baseline (it is a benchmark, not a method)
    best=[]
    for i,hi in enumerate(HIGHER_IS_BETTER):
        col={m:v[i] for m,v in vals.items() if m!="~train_baseline" and not pd.isna(v[i])}
        best.append(max(col,key=col.get) if hi else (min(col,key=col.get) if col else None))

    lines=[]
    for meth,lab in Q_ROWS:
        if meth=="__mid__": lines.append(r"\midrule"); continue
        ra=cell(d, sdg_method=meth)[0]
        cells=[]
        for i,v in enumerate(vals[meth]):
            if pd.isna(v): cells.append("---"); continue
            txt=f"{v:.3f}"
            # ties on a rounded value are bolded together, as in the printed table
            if best[i] is not None and abs(v-vals[best[i]][i])<5e-4: txt=rf"\textbf{{{txt}}}"
            cells.append(txt)
        if meth=="~train_baseline":
            cells[0]="---"; cells[6]="---"      # real-vs-real: no TSTR ratio, no Wass. OHE
        lines.append(f"{lab} & " + " & ".join(cells) + f" & {f1(ra)} " + r"\\")
    (OUT/"table2_quality_overview.tex").write_text("\n".join(lines)+"\n")


# ─────────────────────────────────────────────────────────────────────────────
# Table 7 — memorization (tab:memorization_and_ds_risk), Adult 1k, feature `income`
# The printed table averages RF+KNN+MLP. The four new DP generators were only ever
# run with RF / NaiveBayes / CoBP-RA, so a row for them on the printed attack set
# is impossible. This emits BOTH: `_asprinted` reproduces the current attack set for
# the existing rows, `_common` puts every row (old and new) on RF+NB+CoBP-RA so the
# whole table is one measurement. Only one of them should ship.
# ─────────────────────────────────────────────────────────────────────────────
T7_OLD = [("CellSuppression","Cell Supp."),("RankSwap","RankSwap"),("TabDDPM","TabDDPM"),
          ("__mid__",""),("Synthpop","Synthpop"),("CTGAN","CTGAN"),
          ("AIM_eps1",r"AIM $(\varepsilon{=}1)$"),("MST_eps1",r"MST $(\varepsilon{=}1)$"),
          ("MST_eps10",r"MST $(\varepsilon{=}10)$"),("MST_eps1000",r"MST $(\varepsilon{=}1000)$"),
          ("TVAE","TVAE"),("ARF","ARF")]
T7_NEW = [("PrivBayes_eps1",r"PrivBayes $(\varepsilon{=}1)$"),("PrivBayes_eps1000",r"PrivBayes $(\varepsilon{=}1000)$"),
          ("PrivSyn_eps1",r"PrivSyn $(\varepsilon{=}1)$"),("PrivSyn_eps1000",r"PrivSyn $(\varepsilon{=}1000)$"),
          ("MWEMPGM_eps1",r"MWEM-PGM $(\varepsilon{=}1)$"),("MWEMPGM_eps1000",r"MWEM-PGM $(\varepsilon{=}1000)$"),
          ("PrivateGSD_eps1",r"PrivateGSD $(\varepsilon{=}1)$"),("PrivateGSD_eps1000",r"PrivateGSD $(\varepsilon{=}1000)$")]

MEMO_CSV = ROOT/"experiment_scripts"/"linear_sweep_adult_1000_20260822_182357.csv"

def table7(df):
    """Memorization table. Every row -- the 11 originals and the 8 new DP rows --
    comes from ONE batch (2026-08-22, 285 jobs, 0 errors): Adult 1k, qi=QI_linear
    (which hides exactly `income`), attacks RF+KNN+MLP, 5 disjoint samples with
    round-robin holdout (sample N vs sample (N+1)%5). The originals were re-run
    rather than copied because re-running an untouched synth file still moves a cell
    by ~0.5 pp of attack stochasticity, and this table's claim is a 0.2 pp vs 20 pp
    contrast. Reproduction check against the printed table: Cell Supp. +20.0 vs
    +20.1, RankSwap +13.1 vs +13.5, TabDDPM +13.5 vs +13.4.

    Bold marks a gap significantly greater than zero (one-sample t on the 5 samples).
    On this batch exactly the three printed memorizers clear it, at p <= 0.0001;
    every other row, including ARF (+2.3) and PrivateGSD eps=1 (+2.1), does not.
    """
    ds=pd.read_csv(ROOT/"experiment_scripts"/"ds_risk_scores.csv")
    # The printed d_S column reproduces from the Adult *10k* `_overall` rows (total
    # abs error 0.003 across 11 generators, vs 0.604 for 1k) -- the caption says 1k.
    ds=ds[(ds.dataset=="adult")&(ds["size"]==10000)&(ds.feature=="_overall")]
    ds_map=ds.groupby("sdg").ds.mean().to_dict()

    d=pd.read_csv(MEMO_CSV)
    if d.error.notna().any():
        raise RuntimeError(f"{int(d.error.notna().sum())} memorization jobs errored")
    per=d.groupby(["sdg","sample"])[["train_mean","nontrain_mean","delta_mean"]].mean().reset_index()

    lines, stat_lines, missing = [], [], []
    for meth,lab in T7_OLD + [("__mid__","")] + T7_NEW:
        if meth=="__mid__": lines.append(r"\midrule"); continue
        s_=per[per.sdg==meth]
        dstr=f"{ds_map[meth]:.3f}" if meth in ds_map else "---"
        if s_.empty:
            missing.append(lab); lines.append(f"{lab} & --- & --- & --- & {dstr} "+r"\\"); continue
        t,n=s_.train_mean.mean(), s_.nontrain_mean.mean()
        x=s_.delta_mean.values; delta=x.mean()
        pv=stats.ttest_1samp(x,0).pvalue if len(x)>1 else float("nan")
        h=ci95(x)
        cell_=rf"$\mathbf{{{delta:+.1f}}}$" if pv<0.05 else f"${delta:+.1f}$"
        lines.append(f"{lab} & {t:.1f} & {n:.1f} & {cell_} & {dstr} "+r"\\")
        stat_lines.append(f"| {lab} | {delta:+.2f} | [{delta-h:+.2f}, {delta+h:+.2f}] | {pv:.4f} | {len(x)} |")
    (OUT/"table7_memorization.tex").write_text("\n".join(lines)+"\n")
    (OUT/"STATS_memorization.md").write_text(
        "# Memorization gap: one-sample t-test against zero\n\n"
        "Adult 1k, QI_linear, RF+KNN+MLP, 5 disjoint samples, round-robin holdout.\n"
        "Delta is R_adv(train) - R_adv(holdout) in percentage points.\n\n"
        "| row | Δ pp | 95% CI | p vs 0 | n |\n|---|---:|---|---:|---:|\n"
        + "\n".join(stat_lines) + "\n")
    return missing


# ─────────────────────────────────────────────────────────────────────────────
# Table 9 — disparate impact, from the 2026-08-21 post-repair sweep (not the DB;
# row-level subgroup means are computed by run_per_attack_disparity.py).
# ─────────────────────────────────────────────────────────────────────────────
T9_ROWS=[("CellSuppression","Cell Supp."),("TabDDPM","TabDDPM"),("Synthpop","Synthpop"),
         ("MST_eps1",r"MST $(\varepsilon{=}1)$"),("MST_eps1000",r"MST $(\varepsilon{=}1000)$"),
         ("AIM_eps1",r"AIM $(\varepsilon{=}1)$")]
def table9():
    p=ROOT/"experiment_scripts"/"per_attack_disparity_postrepair.csv"
    d=pd.read_csv(p); d=d[d.attack=="RandomForest"]
    cols=[c for c in d.columns]
    lines=[]
    for meth,lab in T9_ROWS:
        s=d[d.sdg==meth]
        if s.empty: lines.append(f"{lab} & "+" & ".join(["---"]*9)+r" \\"); continue
        r=s.iloc[0]
        def gv(*names):
            for n in names:
                if n in cols and not pd.isna(r[n]): return r[n]*100
            return float("nan")
        out=[f2(gv("mean_ra")), rf"$\times${r['outlier_penalty']:.1f}"]
        out+= [f2(gv(c)) for c in ("race_AI/AN","race_API","race_Black","race_White","race_Other",
                                   "sex_Female","sex_Male")]
        lines.append(f"{lab} & "+" & ".join(out)+r" \\")
    (OUT/"table9_disparate_impact.tex").write_text("\n".join(lines)+"\n")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Statistics for the appendix (E2/E3): paired t-tests on the epsilon curve.
# Paired on the disjoint training sample, so each test has n=5 matched pairs.
# ─────────────────────────────────────────────────────────────────────────────
def stats_report(df):
    out=["# Epsilon-curve statistics (paired on disjoint training sample)","",
         "Cell = mean of RandomForest / NaiveBayes / CoBP-RA. Delta in percentage points.",""]
    for ds,size in [("adult",10000),("cdc_diabetes",1000)]:
        out.append(f"## {ds} {size}"); out.append("")
        out.append("| QI | generator | comparison | n | Δ pp | 95% CI | p |")
        out.append("|---|---|---|---:|---:|---|---:|")
        for qi in ["QI1","QI_large","QI_behavioral"]:
            d=_eps_frame(df,ds,size,qi)
            for gen in DP_GENS:
                s=d[d.gen==gen]
                if s.empty: continue
                per=s.groupby(["eps","sample"]).ra_mean.mean().reset_index()
                for lo,hi in [(0.1,10),(1,10),(10,1000)]:
                    A=per[per.eps==lo].set_index("sample").ra_mean
                    B=per[per.eps==hi].set_index("sample").ra_mean
                    k=A.index.intersection(B.index)
                    if len(k)<3: continue
                    diff=(B[k]-A[k]); t,p=stats.ttest_rel(B[k],A[k])
                    h=stats.t.ppf(.975,len(k)-1)*diff.std(ddof=1)/np.sqrt(len(k))
                    out.append(f"| {qi} | {DP_LABEL[gen]} | ε {g(lo)}→{g(hi)} | {len(k)} | "
                               f"{diff.mean():+.2f} | [{diff.mean()-h:+.2f}, {diff.mean()+h:+.2f}] | {p:.4f} |")
        out.append("")
    (OUT/"STATS_eps_curve.md").write_text("\n".join(out)+"\n")


if __name__ == "__main__":
    runs = load_runs()
    runs = runs[runs.confidence=="certain"]     # never build a table from flagged rows
    grid, cov = table1(runs)
    table4(runs); table2(runs); miss=table7(runs); table9(); figure_eps(runs); stats_report(runs)
    print("wrote:")
    for f in sorted(OUT.iterdir()): print(f"  {f.name:<40}{f.stat().st_size:>8} B")
    if miss:
        print("\nTable 7 rows with NO QI_linear memorization runs (cannot be filled without new runs):")
        for m in miss: print(f"  {m}")
    if cov:
        print("\ncells with n<5 disjoint samples (caption must not claim 5):")
        for a,b,n in cov: print(f"  {a:<24}{b:<28}n={n}")
