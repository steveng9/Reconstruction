#!/usr/bin/env python
"""
The generator functions behind the paper's tables and figures.

Each `tableN` / `figure_*` / `stats_*` function here builds exactly one group of
outputs from results.db or from a named, committed result CSV, and writes them
into the directory named by RECON_TABLE_OUT (default: expected_output/tables/).
Nothing here edits the manuscript; each output is a drop-in replacement body the
author pastes (or \input's) into the .tex.

**Which of these run, and in what order, is decided by the manifest in
`experiment_scripts/paper_objects.py` and dispatched by `reproduce.py`.** Adding
a function here does nothing until a manifest row names it.

Provenance rule: every number printed comes from results.db or from a named
post-repair CSV. No number is carried over from the .tex.

    python reproduce.py                  # the entry point
    python experiment_scripts/regen_camera_tables.py    # equivalent, still supported
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

def _require(path: Path) -> Path:
    """Fail loudly if an input this script needs is missing.

    Tables 2, 7 and 9 read result CSVs alongside results.db. If one is absent the
    table would still be emitted, but with silently different contents -- so stop
    instead, and say which file is missing.
    """
    if not Path(path).exists():
        raise SystemExit(
            f"error: required input not found: {path}\n"
            "It should be committed alongside results.db. If you are working from a\n"
            "partial checkout, run:  git checkout -- experiment_scripts/"
        )
    return Path(path)


def _connect_for_reading(path):
    """Open the results database without needing to write next to it.

    `results.db` is in WAL mode, and reading a WAL database means creating a
    `-shm` and a `-wal` file beside it -- so an ordinary read fails with
    "attempt to write a readonly database" whenever the repository is mounted
    somewhere the current user cannot write. That is the normal case inside a
    container whose user id does not match the host's. Regenerating tables only
    ever reads, so fall back to opening the file as immutable, which tells
    SQLite to skip the WAL and the locking entirely.
    """
    try:
        con = sqlite3.connect(path)
        con.execute("SELECT 1 FROM runs LIMIT 1").fetchone()
        return con
    except sqlite3.Error:
        return sqlite3.connect(f"file:{Path(path).resolve()}?immutable=1", uri=True)


def load_runs() -> pd.DataFrame:
    con = _connect_for_reading(DB)
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
# A row whose label is a pair reads its eight non-MST columns from the first label and
# its five MST columns from the second: CoBP-RA is the QI-graph, entropy-weighted
# variant outside the MST columns and the plain attack inside them, and CondMST is
# the bounded-clique variant outside them and the tree variant inside them.
T1_ROWS = [("__grp__", "Reference points (not attacks)"),
           ("Mode","Mode"),("Random","Random"),("MeasureDeid","Copy"),
           ("__grp__", "Each feature in isolation: classifiers"),
           ("KNN",r"\textsc{knn}"),("NaiveBayes","Naive Bayes"),("LogisticRegression","Logistic Regression"),
           ("SVM",r"\textsc{svm}"),("RandomForest","Random Forest"),("LightGBM","LightGBM"),
           ("MLP",r"\textsc{mlp}"),("TabPFN","TabPFN"),
           ("Ensemble_MRF_5","Best ensemble"),
           ("__grp__", "Feature-correlated: autoregressive"),
           # The causal (autoregressive) transformer at its tuned defaults; label
           # `Attention` is the earlier per-feature model.
           ("AttentionAutoregressiveTuned",r"ARFFormer$^\dagger$"),
           ("CoBP-RA_HardChain","Best chain"),
           ("__grp__", "Feature-correlated: row-wise message passing"),
           (("MarginalRF_graphQI_entropyBP","CoBP-RA"),r"CoBP-RA$^\dagger$"),
           ("__grp__", "Feature-correlated: joint generative conditioning"),
           ("JointMLP",r"MultiHeadMLP$^\dagger$"),(("PartialMSTBounded","PartialMST"),r"CondMST$^\dagger$"),
           ("TabDDPM",r"CondDDPM$^\dagger$"),("ConditionedRePaint",r"CondRePaint$^\dagger$")]
# The two enhancement rows summarise a sweep rather than name an attack, so they are
# left out of the bottom Avg. row.
T1_NOT_AVERAGED = ("Best ensemble", "Best chain")

def _t1_label(atk, sdg):
    if isinstance(atk, tuple):
        return atk[1] if sdg.startswith("MST_") else atk[0]
    return atk

def table1(df):
    d = df[(df.dataset=="adult")&(df.dataset_size==10000)&(df.qi=="QI1")&(df.split=="standard")]
    lines, grid = [], {}
    for atk, lab in T1_ROWS:
        if atk == "__grp__":
            if lines: lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{15}}{{l}}{{\emph{{{lab}}}}}\\"); continue
        vals = [cell(d, attack_label=_t1_label(atk, s), sdg_method=s)[0] for s, _ in T1_COLS]
        grid[lab] = vals
        avg = np.nanmean(vals) if not all(pd.isna(v) for v in vals) else np.nan
        lines.append(f"{lab} & " + " & ".join(f1(v) for v in vals) + f" & {f1(avg)} " + r"\\")
    colavg = [np.nanmean([grid[a][i] for a in grid if a not in T1_NOT_AVERAGED]) for i in range(len(T1_COLS))]
    body = "\n".join(lines) + "\n" + r"\midrule" + "\n" + \
           r"\textbf{Avg.} & " + " & ".join(rf"\textbf{{{f1(v)}}}" for v in colavg) + r" & \\"
    (OUT/"table1_ra_mean_adult.tex").write_text(body + "\n")
    # coverage report: which cells are missing / thin
    cov = []
    for atk, lab in T1_ROWS:
        if atk == "__grp__": continue
        for s, sl in T1_COLS:
            m, c, n = cell(d, attack_label=_t1_label(atk, s), sdg_method=s)
            if n < 5: cov.append((lab, sl, n))
    return grid, cov


# ─────────────────────────────────────────────────────────────────────────────
# The per-dataset attack x SDG tables: CDC Diabetes at 1k and 100k, and NIST SBO.
#
# These three share a shape that differs from Table 1: a Mode baseline row above
# a rule, then the ML attacks, with the best value in each column bolded among
# the ML attacks only. Table 1 instead groups attacks by family and carries an
# average column, so it keeps its own function.
#
# Column and row sets differ per table and are NOT derived from what happens to
# be in the database -- they are transcribed from the printed tables, so a
# regenerated table is directly comparable to the paper. A cell with no runs
# prints "---" rather than silently changing the table's shape.
#
# Output files are named by \label rather than by printed table number: the
# numbering shifts whenever a table moves, and these three sit in the appendix.
# ─────────────────────────────────────────────────────────────────────────────
# The row printed as "CoBP-RA$^\dagger$" in all three of these tables is the
# QIGraph + EntropyBP variant, not the plain attack -- verified by matching every
# non-superseded cell of all three printed tables exactly (e.g. NIST SBO RankSwap
# 80.8, CDC 100k Cell Supp. 85.7, CDC 1k TVAE 46.3), which no other variant does.
# Table 1 (Adult) uses a different variant and keeps its own row spec.
COBP_VARIANT = "MarginalRF_graphQI_entropyBP"

MST_EPS_COLS = [("MST_eps0.1", r"MST $(\varepsilon{=}0.1)$"), ("MST_eps1", r"MST $(\varepsilon{=}1)$"),
                ("MST_eps10", r"MST $(\varepsilon{=}10)$"), ("MST_eps100", r"MST $(\varepsilon{=}100)$"),
                ("MST_eps1000", r"MST $(\varepsilon{=}1000)$")]
DEID_COLS = [("RankSwap", "RankSwap"), ("CellSuppression", "Cell Supp."), ("Synthpop", "Synthpop")]
DEEP_COLS = [("TVAE", "TVAE"), ("CTGAN", "CTGAN"), ("ARF", "ARF"), ("TabDDPM", "TabDDPM")]

PER_DATASET = [
    dict(
        out="table_ra_mean_cdc.tex", label="tab:ra_mean_cdc",
        dataset="cdc_diabetes", size=1000, qi="QI1",
        cols=DEID_COLS + DEEP_COLS + MST_EPS_COLS
             + [("AIM_eps1", r"AIM $(\varepsilon{=}1)$"), ("AIM_eps3", r"AIM $(\varepsilon{=}3)$"),
                ("AIM_eps10", r"AIM $(\varepsilon{=}10)$")],
        # `TabDDPM` as an *attack* label is CondDDPM; as an SDG column it is the
        # generator. The two never collide because they index different axes.
        rows=[("KNN", "KNN"), ("RandomForest", "Random Forest"), ("MLP", "MLP"),
              ("TabDDPM", r"CondDDPM$^\dagger$"), (COBP_VARIANT, r"CoBP-RA$^\dagger$")],
    ),
    dict(
        out="table_cdc_100k.tex", label="tab:cdc_100k",
        dataset="cdc_diabetes", size=100000, qi="QI1",
        cols=DEID_COLS + DEEP_COLS + MST_EPS_COLS + [("AIM_eps1", r"AIM $(\varepsilon{=}1)$")],
        rows=[("RandomForest", "Random Forest"), ("MLP", r"\textsc{mlp}"),
              ("NaiveBayes", "Naive Bayes"), (COBP_VARIANT, r"CoBP-RA$^\dagger$")],
    ),
    dict(
        out="table_ra_mean_nist_sbo.tex", label="tab:ra_mean_nist_sbo",
        dataset="nist_sbo", size=1000, qi="QI1",
        cols=DEID_COLS + DEEP_COLS + MST_EPS_COLS,
        rows=[("KNN", "KNN"), ("NaiveBayes", "Naive Bayes"),
              ("RandomForest", "Random Forest"), (COBP_VARIANT, r"CoBP-RA$^\dagger$")],
    ),
]

def table_perdataset(df):
    """Build all three per-dataset tables. Returns rows with no runs, per table.

    Reproduction against the printed tables, checked cell by cell:

      tab:cdc_100k          exact -- all 52 cells match the paper.
      tab:ra_mean_cdc       9 of 75 cells differ, and every one of them sits in
                            the MST(eps=0.1) or AIM(eps=3) column. Both columns
                            were invalidated by the 2026-08 float-binned encoding
                            repair and re-run; the paper prints the pre-repair
                            values. Largest move 0.9 pp (MLP / AIM eps=3). The
                            superseded rows are still in `runs_superseded`.
      tab:ra_mean_nist_sbo  the eight non-MST columns are exact. The MST columns
                            are not reproducible as printed: MST(eps=0.1) and
                            MST(eps=1) were superseded and never re-run, so they
                            print "---", and the other three MST budgets survive
                            on 2 disjoint samples rather than 5, which moves them
                            by up to 0.4 pp. Filling this needs new runs, not a
                            new generator.

    A cell with no post-repair runs prints "---" rather than falling back to the
    superseded value: an honest gap is better than a stale number.
    """
    gaps = []
    for spec in PER_DATASET:
        d = df[(df.dataset == spec["dataset"]) & (df.dataset_size == spec["size"])
               & (df.qi == spec["qi"]) & (df.split == "standard")]
        # Mode is a reference point, not an attack: it sits above the rule and is
        # never a candidate for the per-column bold.
        mode = [cell(d, attack_label="Mode", sdg_method=s)[0] for s, _ in spec["cols"]]
        ml = {atk: [cell(d, attack_label=atk, sdg_method=s)[0] for s, _ in spec["cols"]]
              for atk, _ in spec["rows"]}

        # Each row ends in its mean over generators, and a closing Avg. row gives
        # each column's mean over the ML attacks (Mode excluded). Both are taken
        # over the cells as printed, to one decimal.
        def row_mean(vals):
            shown = [float(f1(v)) for v in vals if not pd.isna(v)]
            return np.mean(shown) if shown else np.nan
        for atk, _ in spec["rows"]:
            ml[atk] = ml[atk] + [row_mean(ml[atk])]
        mode = mode + [row_mean(mode)]

        best = []
        for i in range(len(spec["cols"]) + 1):
            col = [ml[a][i] for a, _ in spec["rows"] if not pd.isna(ml[a][i])]
            best.append(max(col) if col else np.nan)

        lines = ["Mode & " + " & ".join(f1(v) for v in mode) + r" \\", r"\midrule"]
        for atk, lab in spec["rows"]:
            cells = []
            for i, v in enumerate(ml[atk]):
                t = f1(v)
                # Bold on the rounded value, so a visible tie is bolded twice --
                # as the printed tables do (CDC 100k, RankSwap column).
                if not pd.isna(v) and not pd.isna(best[i]) and t == f1(best[i]):
                    t = rf"\textbf{{{t}}}"
                cells.append(t)
            lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
            if all(pd.isna(v) for v in ml[atk][:-1]):
                gaps.append((spec["label"], lab))
        colavg = [row_mean([ml[a][i] for a, _ in spec["rows"]]) for i in range(len(spec["cols"]))]
        lines += [r"\midrule",
                  r"\textbf{Avg.} & " + " & ".join(rf"\textbf{{{f1(v)}}}" for v in colavg) + r" & \\"]
        (OUT / spec["out"]).write_text("\n".join(lines) + "\n")
    return gaps


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

# AIM at eps=1 (and, on CDC, eps=10) was first generated through SmartNoise's own
# preprocessor and later regenerated with continuous columns pre-binned, like the
# other five mechanisms; those releases are stored as AIM_eps1_pb / AIM_eps10_pb.
# Wherever a pre-binned release was attacked it stands in for the earlier one, so
# an AIM curve rests on one pipeline.
AIM_PREBINNED = {"AIM_eps1_pb": (1.0, "AIM_eps1"), "AIM_eps10_pb": (10.0, "AIM_eps10")}

def _aim_prebinned(df):
    df = df.copy()
    drop = np.zeros(len(df), dtype=bool)
    for pb, (eps, old) in AIM_PREBINNED.items():
        for (ds, size, qi), _ in df[df.sdg_method == pb].groupby(["dataset", "dataset_size", "qi"]):
            here = (df.dataset == ds) & (df.dataset_size == size) & (df.qi == qi)
            df.loc[here & (df.sdg_method == pb), ["gen", "eps"]] = ["AIM", eps]
            drop |= (here & (df.sdg_method == old)).to_numpy()
    return df[~drop]

def table4(df):
    df = _aim_prebinned(df)
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

def figures_paper():
    """Draw the epsilon-sweep figures as they are printed in the paper.

    Two scripts draw them, fig_eps_curves_full.py for the main QI set and
    fig_eps_curves_qi.py for the two further ones. They are run here so that
    `reproduce.py` writes the figures next to the tables.
    """
    import subprocess
    here = ROOT / "experiment_scripts"
    for script in ("fig_eps_curves_full.py", "fig_eps_curves_qi.py"):
        subprocess.run([sys.executable, str(here / script), "--out", str(OUT)],
                       check=True, stdout=subprocess.DEVNULL,
                       env={**os.environ, "RECON_RESULTS_DB": str(DB)})


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 — quality profile (tab:quality_overview), extended with the new DP gens
# ─────────────────────────────────────────────────────────────────────────────
# Every row comes from quality_results_merged.csv. Wass. OHE there is computed on 20
# equal-width bins over the real column's range (compute_wasserstein_ohe.py), and the
# AIM rows are the pre-binned releases, like the other five DP generators;
# historical/update_quality_camera_ready.py records how the CSV was brought to that state.
QUALITY_CSV = ROOT/"experiment_scripts"/"quality_results_merged.csv"
Q_ROWS = [("MST_eps0.1",r"MST $(\varepsilon{=}0.1)$"),("MST_eps1",r"MST $(\varepsilon{=}1)$"),
          ("MST_eps10",r"MST $(\varepsilon{=}10)$"),("MST_eps1000",r"MST $(\varepsilon{=}1000)$"),
          ("AIM_eps1",r"AIM $(\varepsilon{=}1)$"),("AIM_eps10",r"AIM $(\varepsilon{=}10)$"),
          ("AIM_eps1000",r"AIM $(\varepsilon{=}1000)$"),
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
    q = pd.read_csv(_require(QUALITY_CSV))
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
    ds=pd.read_csv(_require(ROOT/"experiment_scripts"/"ds_risk_scores.csv"))
    # The printed d_S column reproduces from the Adult *10k* `_overall` rows (total
    # abs error 0.003 across 11 generators, vs 0.604 for 1k) -- the caption says 1k.
    ds=ds[(ds.dataset=="adult")&(ds["size"]==10000)&(ds.feature=="_overall")]
    ds_map=ds.groupby("sdg").ds.mean().to_dict()

    d=pd.read_csv(_require(MEMO_CSV))
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
# Table 6 — MIA vs RA-as-MIA (not the DB: AUCs, one per dataset and generator).
# Two committed CSVs hold them. mia_comparison_results.csv carries the comparison
# runs of May 2026 and the wider run of 2026-08-21, recovered from the drivers'
# console output by historical/parse_mia_log_to_csv.py. The first six columns use
# the May run where it covered the generator, and the August run otherwise. The
# PrivateGSD columns were added with the August sweep and read its own CSV.
# ─────────────────────────────────────────────────────────────────────────────
MIA_CSV = ROOT/"experiment_scripts"/"mia_comparison_results.csv"
MIA_SWEEP_CSV = ROOT/"experiment_scripts"/"mia_rebuttal_sweep_results.csv"
T6_COLS = [("CellSuppression","Cell Supp."),("RankSwap","RankSwap"),("TabDDPM","TabDDPM"),
           ("Synthpop","Synthpop"),("MST_eps1",r"MST $(\varepsilon{=}1)$"),
           ("MST_eps1000",r"MST $(\varepsilon{=}1000)$"),
           ("PrivateGSD_eps1",r"PrivateGSD $(\varepsilon{=}1)$"),
           ("PrivateGSD_eps1000",r"PrivateGSD $(\varepsilon{=}1000)$")]
T6_BLOCKS = [("adult", "QI1", r"\textbf{Adult} data ($n=10{,}000$, 15 features, $\text{QI}_{\text{demo}}$)"),
             ("cdc_diabetes", "QI1", r"\textbf{CDC Diabetes} data ($n=1{,}000$, 22 features, $\text{QI}_{\text{demo}}$)"),
             ("nist_arizona_25feat", "QI_medium", r"\textbf{NIST Arizona} data ($n=10{,}000$, 25 features, $\text{QI}_{\text{medium}}$)")]
T6_METRICS = [("SynthDistance_auc","SynthDistance"),("NNDR_auc","NNDR"),
              ("RA_as_MIA_auc","RA-as-MIA")]

def _t6_row(runs, sweep, ds, qi, meth):
    """The AUCs for one (dataset, generator) cell group, or None."""
    if meth.startswith("PrivateGSD"):
        eps = float(meth.split("_eps")[1])
        r = sweep[(sweep.dataset==ds)&(sweep.qi==qi)&(sweep.sdg_method=="PrivateGSD")&(sweep.epsilon==eps)]
    else:
        r = runs[(runs.dataset==ds)&(runs.qi==qi)&(runs.sdg_method==meth)].sort_values("run")
    # A run that did not report one of the three AUCs falls through to the next run.
    return None if r.empty else r.bfill().iloc[0]

def table6():
    runs = pd.read_csv(_require(MIA_CSV)); sweep = pd.read_csv(_require(MIA_SWEEP_CSV))
    ncol = len(T6_COLS) + 1
    lines = []
    for bi,(ds,qi,label) in enumerate(T6_BLOCKS):
        if bi: lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{" + label + r"} \\[2pt]")
        cells = {meth: _t6_row(runs, sweep, ds, qi, meth) for meth,_ in T6_COLS}
        for col,mlabel in T6_METRICS:
            if mlabel == "RA-as-MIA": lines.append(rf"\cmidrule(l){{1-{ncol}}}")
            out = []
            for meth,_ in T6_COLS:
                r = cells[meth]
                if r is None or pd.isna(r[col]): out.append("---"); continue
                v = f"{r[col]:.2f}"
                # bold the stronger of the two distance-based MIAs, NNDR on a printed tie
                nn_wins = f"{r['NNDR_auc']:.2f}" >= f"{r['SynthDistance_auc']:.2f}"
                bold = (col=="NNDR_auc" and nn_wins) or (col=="SynthDistance_auc" and not nn_wins)
                out.append(rf"\textbf{{{v}}}" if bold else v)
            lines.append(f"{mlabel:15s} & " + " & ".join(out) + r" \\")
    (OUT/"table6_mia_comparison.tex").write_text("\n".join(lines)+"\n")


# ─────────────────────────────────────────────────────────────────────────────
# Table 9 — disparate impact (not the DB: these are means over individual records).
# The three non-DP rows come from analyze_ra_subgroups.py, reduced to subgroup
# means by historical/summarize_ra_subgroups.py; the two MST rows come from the
# sweep that run_per_attack_disparity.py ran on the MST releases. The MST rows
# print two decimals because their values are below 1%.
# ─────────────────────────────────────────────────────────────────────────────
def _f1_via_2dp(v):
    """One decimal, taken from the two-decimal value analyze_ra_subgroups.py reports."""
    from decimal import Decimal, ROUND_HALF_UP
    two = Decimal(repr(float(v))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return str(two.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

T9_ROWS=[("CellSuppression","Cell Supp.","ra_subgroups_summary.csv",_f1_via_2dp),
         ("TabDDPM","TabDDPM","ra_subgroups_summary.csv",_f1_via_2dp),
         ("Synthpop","Synthpop","ra_subgroups_summary.csv",_f1_via_2dp),
         ("MST_eps1",r"MST $(\varepsilon{=}1)$","per_attack_disparity_postrepair.csv",f2),
         ("MST_eps1000",r"MST $(\varepsilon{=}1000)$","per_attack_disparity_postrepair.csv",f2)]
def table9():
    lines=[]
    for meth,lab,src,fmt in T9_ROWS:
        d=pd.read_csv(_require(ROOT/"experiment_scripts"/src))
        r=d[(d.sdg==meth)&(d.attack=="RandomForest")].iloc[0]
        out=[fmt(r["mean_ra"]*100), rf"$\times${r['outlier_penalty']:.1f}"]
        out+= [fmt(r[c]*100) for c in ("race_AI/AN","race_API","race_Black","race_White","race_Other",
                                       "sex_Female","sex_Male")]
        lines.append(f"{lab} & "+" & ".join(out)+r" \\")
    (OUT/"table9_disparate_impact.tex").write_text("\n".join(lines)+"\n")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics for the appendix (E2/E3): paired t-tests on the epsilon curve.
# Paired on the disjoint training sample, so each test has n=5 matched pairs.
# ─────────────────────────────────────────────────────────────────────────────
def stats_report(df):
    df = _aim_prebinned(df)
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
                    if len(k)<5: continue      # a comparison needs all five training samples
                    diff=(B[k]-A[k]); t,p=stats.ttest_rel(B[k],A[k])
                    h=stats.t.ppf(.975,len(k)-1)*diff.std(ddof=1)/np.sqrt(len(k))
                    out.append(f"| {qi} | {DP_LABEL[gen]} | ε {g(lo)}→{g(hi)} | {len(k)} | "
                               f"{diff.mean():+.2f} | [{diff.mean()-h:+.2f}, {diff.mean()+h:+.2f}] | {p:.4f} |")
        out.append("")
    (OUT/"STATS_eps_curve.md").write_text("\n".join(out)+"\n")


if __name__ == "__main__":
    # The entry point moved to reproduce.py at the repository root, which decides
    # what to build from the manifest in paper_objects.py. This file is now the
    # library of generator functions. Running it directly still works, and does
    # exactly what `python reproduce.py` does.
    sys.path.insert(0, str(ROOT))
    import reproduce
    raise SystemExit(reproduce.main([]))
