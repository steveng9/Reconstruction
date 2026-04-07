#!/usr/bin/env python
"""
QI analysis → side-by-side LaTeX comparison table.

Rows = attack methods (grouped: Baselines / ML Classifiers / Neural).
Cols = QI_tiny | QI1 | QI_large | Δ_size  ‖  QI1 | QI_behavioral | Δ_comp

Two paper angles in one table:
  1. QI size effect:     QI_tiny → QI1 → QI_large (3→6→10 for adult; 4→10→16 for CDC)
                         All three columns scored on features hidden in ALL three QIs
                         (size_intersection).  Δ_size = QI_large − QI_tiny.
  2. Composition effect: QI1 vs QI_behavioral at matched size.
                         Both columns scored on features hidden in BOTH QIs
                         (comp_intersection).  Δ_comp = QI_behavioral − QI1.
                         ≈ 0 means "size is all that matters"; ≠ 0 means content matters too.

QI1 appears TWICE with DIFFERENT scores — once per section — because each section averages
over a different intersection of features.  The two QI1 columns are labeled identically in
the table; section headers make the distinction clear.

Cell values: mean RA over intersection features, averaged over all SDG methods × 5 samples.
Bold = max value per row within each section (size or composition) separately.
Δ columns: signed (+ or −), no bolding.

NOTE: Requires per-feature RA_ columns in the CSV.  Use CSVs from run_qi_sweep.py
(including a QI1 run); old sweep_results_*.csv files lack per-feature columns.

────────────────────────────────────────────────────────────────────────────────
Usage (from-csv, recommended — faster, no WandB API calls):
  python wandb_to_latex_qi.py \\
      --dataset adult \\
      --from-csv experiment_scripts/qi_analysis/qi_adult_QI1_*.csv \\
                 experiment_scripts/qi_analysis/qi_adult_QI_tiny_*.csv \\
                 experiment_scripts/qi_analysis/qi_adult_QI_large_*.csv \\
                 experiment_scripts/qi_analysis/qi_adult_QI_behavioral_*.csv

  python wandb_to_latex_qi.py \\
      --dataset cdc_diabetes --size 1000 \\
      --from-csv experiment_scripts/qi_analysis/qi_cdc_diabetes_QI1_*.csv \\
                 experiment_scripts/qi_analysis/qi_cdc_diabetes_QI_tiny_*.csv \\
                 experiment_scripts/qi_analysis/qi_cdc_diabetes_QI_large_*.csv \\
                 experiment_scripts/qi_analysis/qi_cdc_diabetes_QI_behavioral_*.csv

CLI:
  --dataset    adult | cdc_diabetes                   (required)
  --from-csv   one or more CSV files; 'qi' column distinguishes variants
  --out        output .tex path (default: qi_table_<dataset>.tex in this folder)
  --decimals   decimal places (default 2)
  --sdg        restrict average to these SDG labels (default: all)
  --attacks    restrict rows to these attack names (default: all in ATTACK_GROUPS)
  --size       for CSVs with a 'size' column, keep only this training size
               (useful when CDC CSVs mix 1k and 100k rows)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

WANDB_PROJECT = "tabular-reconstruction-attacks"

# ── Per-dataset config ─────────────────────────────────────────────────────────
#
# qi_known_sizes: how many features the adversary *knows* in each QI variant.
# wandb_groups:   which WandB group holds each QI variant's runs.
#                 (Used only when --from-csv is not provided.)

DATASET_CONFIGS: dict[str, dict] = {
    "adult": {
        "display":  "Adult",
        "size":     10_000,
        "qi_known_sizes": {
            "QI_tiny":       3,
            "QI1":           6,
            "QI_large":     10,
            "QI1_comp":      6,   # same as QI1; separate column in composition section
            "QI_behavioral": 6,
        },
        # Features hidden in ALL of QI_tiny, QI1, QI_large (used for size sweep columns)
        "size_intersection": [
            "fnlwgt", "education-num", "capital-gain", "capital-loss", "income",
        ],
        # Features hidden in BOTH QI1 and QI_behavioral (used for composition columns)
        "comp_intersection": [
            "capital-gain", "capital-loss", "income",
        ],
        # Features hidden in BOTH QI_tiny and QI1 (used for the pairwise Δ†_{t→1} column)
        "tiny_qi1_intersection": [
            "workclass", "fnlwgt", "education-num", "occupation", "relationship",
            "capital-gain", "capital-loss", "hours-per-week", "income",
        ],
        "wandb_groups": {
            "QI_tiny":       "qi-analysis-adult-10000",
            "QI1":           "qi-analysis-adult-10000",
            "QI_large":      "qi-analysis-adult-10000",
            "QI_behavioral": "qi-analysis-adult-10000",
        },
    },
    "cdc_diabetes": {
        "display":  "CDC Diabetes",
        "size":     1_000,
        "qi_known_sizes": {
            "QI_tiny":        4,
            "QI1":           10,
            "QI_large":      16,
            "QI1_comp":      10,   # same as QI1; separate column in composition section
            "QI_behavioral": 10,
        },
        # Features hidden in ALL of QI_tiny, QI1, QI_large (used for size sweep columns)
        "size_intersection": [
            "Diabetes_binary", "Stroke", "HeartDiseaseorAttack",
            "HvyAlcoholConsump", "MentHlth", "PhysHlth",
        ],
        # Features hidden in BOTH QI1 and QI_behavioral (used for composition columns)
        "comp_intersection": [
            "Diabetes_binary", "Stroke", "MentHlth", "PhysHlth",
        ],
        # Features hidden in BOTH QI_tiny and QI1 (used for the pairwise Δ†_{t→1} column)
        "tiny_qi1_intersection": [
            "Diabetes_binary", "Stroke", "HeartDiseaseorAttack", "CholCheck",
            "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
            "NoDocbcCost", "MentHlth", "PhysHlth", "DiffWalk",
        ],
        "wandb_groups": {
            "QI_tiny":       "qi-analysis-cdc_diabetes-1000",
            "QI1":           "qi-analysis-cdc_diabetes-1000",
            "QI_large":      "qi-analysis-cdc_diabetes-1000",
            "QI_behavioral": "qi-analysis-cdc_diabetes-1000",
        },
    },
}

# Ordered QI variants shown left-to-right.
# QI1_comp is QI1 re-scored on comp_intersection features (separate column in comp section).
QI_ORDER = ["QI_tiny", "QI1", "QI_large", "QI1_comp", "QI_behavioral"]

# Δ definitions: (label, minuend_qi, subtrahend_qi)
DELTAS = [
    ("$\\Delta_{\\text{size}}$",   "QI_large",      "QI_tiny"),       # effect of more known features
    ("$\\Delta_{\\text{comp}}$",   "QI_behavioral", "QI1_comp"),       # effect of feature content (at matched size)
]

# ── Attack display order / groupings ──────────────────────────────────────────

ATTACK_GROUPS = [
    ("Baselines",      ["Mode", "Random"]),
    ("ML Classifiers", ["KNN", "NaiveBayes", "RandomForest", "LightGBM"]),
    ("Neural",         ["MLP"]),
]

ATTACK_DISPLAY: dict[str, str] = {
    "Mode":         "Mode",
    "Random":       "Random",
    "KNN":          r"\textsc{knn}",
    "NaiveBayes":   "Naive Bayes",
    "RandomForest": "Random Forest",
    "LightGBM":     "LightGBM",
    "MLP":          r"\textsc{mlp}",
}

QI_DISPLAY: dict[str, str] = {
    "QI_tiny":       r"$\text{QI}_{\text{tiny}}$",
    "QI1":           r"$\text{QI}_{1}$",
    "QI_large":      r"$\text{QI}_{\text{large}}$",
    "QI1_comp":      r"$\text{QI}_{1}$",   # same label, different section
    "QI_behavioral": r"$\text{QI}_{\text{beh.}}$",
}

# Maps internal column name → actual qi value in the raw data (for _n_obs lookup).
QI_DATA_ALIAS: dict[str, str] = {
    "QI1_comp": "QI1",
}

# Column label for the optional pairwise QI_tiny∩QI1 delta column.
# Uses † to signal it is scored on a broader feature set than the three-way size intersection.
TINY_QI1_DELTA_LABEL = r"$\Delta^{\dag}_{t{\to}1}$"

QI_SECTION_LABELS = {
    "size":        r"QI size sweep ($\nearrow$ adversary knowledge)",
    "composition": r"Composition contrast (matched $|\text{QI}|$)",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_from_csv(csv_paths: list[str],
                  dataset: str,
                  qi_variants: list[str],
                  sdg_filter: list[str] | None,
                  size_filter: int | None,
                  attack_filter: list[str] | None) -> pd.DataFrame:
    """Concatenate CSVs, normalise columns, filter, return long DataFrame."""
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        print(f"  Loaded {p}: {len(df)} rows")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {len(df)} rows")

    # Drop failed/incomplete rows (no ra_mean)
    before = len(df)
    df = df[df["ra_mean"].notna()]
    if (dropped := before - len(df)):
        print(f"  Dropped {dropped} rows with missing ra_mean.")

    # Filter by training size if a 'size' column is present and --size was given
    if size_filter is not None and "size" in df.columns:
        df = df[(df["size"] == size_filter) | df["size"].isna()]
        print(f"  After size filter ({size_filter}): {len(df)} rows")

    # Filter to the requested dataset if a 'dataset' column is present
    if "dataset" in df.columns:
        df = df[df["dataset"] == dataset]

    # Keep only the QI variants we care about
    df = df[df["qi"].isin(qi_variants)]

    if sdg_filter:
        df = df[df["sdg"].isin(sdg_filter)]
    if attack_filter:
        df = df[df["attack"].isin(attack_filter)]

    # Deduplicate: keep last (latest file in the list wins for same key)
    key = ["attack", "sdg", "qi", "sample"]
    before = len(df)
    df = df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    if (dropped := before - len(df)):
        print(f"  Deduplicated: dropped {dropped} duplicate rows (kept last).")

    print(f"  Final: {len(df)} rows  |  "
          f"{df['attack'].nunique()} attacks  |  "
          f"{df['sdg'].nunique()} SDG methods  |  "
          f"{df['qi'].nunique()} QI variants  |  "
          f"{df['sample'].nunique()} samples")
    return df


def _sdg_label(method: str, params: dict | None) -> str:
    params = params or {}
    eps = params.get("epsilon") or params.get("eps")
    return f"{method}_eps{float(eps):g}" if eps is not None else method


def fetch_from_wandb(dataset: str,
                     qi_variants: list[str],
                     sdg_filter: list[str] | None,
                     size_filter: int | None,
                     attack_filter: list[str] | None) -> pd.DataFrame:
    """Pull runs from WandB for each QI variant (uses per-QI group mapping)."""
    import wandb
    api    = wandb.Api(timeout=60)
    entity = api.default_entity
    path   = f"{entity}/{WANDB_PROJECT}"

    cfg    = DATASET_CONFIGS[dataset]
    groups = cfg["wandb_groups"]

    all_rows: list[dict] = []
    seen_groups: set[str] = set()

    for qi in qi_variants:
        group = groups.get(qi)
        if group is None:
            print(f"  WARNING: No WandB group configured for {dataset}/{qi} — skipping.")
            continue

        server_filters: dict = {"group": group, "config.dataset": dataset}
        if attack_filter:
            server_filters["config.attack_method"] = {"$in": attack_filter}

        if group not in seen_groups:
            print(f"  Querying group={group!r} (dataset={dataset!r}) ...")
            seen_groups.add(group)

        runs = api.runs(path, filters=server_filters)
        n_added = 0
        for run in runs:
            rcfg  = run.config
            summ  = run.summary
            attack      = rcfg.get("attack_method")
            sdg_method  = rcfg.get("sdg_method")
            sdg_params  = rcfg.get("sdg_params") or {}
            run_qi      = rcfg.get("qi")
            sample      = rcfg.get("sample_idx")
            ra_mean     = summ.get("RA_mean")

            if None in (attack, sdg_method, run_qi, sample) or ra_mean is None:
                continue
            if run_qi != qi:
                continue

            sdg = _sdg_label(sdg_method, sdg_params)
            if sdg_filter and sdg not in sdg_filter:
                continue

            all_rows.append({
                "attack":     attack,
                "sdg":        sdg,
                "qi":         run_qi,
                "sample":     int(sample),
                "ra_mean":    float(ra_mean),
                "created_at": run.created_at,
            })
            n_added += 1

        print(f"    QI={qi}: {n_added} runs")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    key = ["attack", "sdg", "qi", "sample"]
    before = len(df)
    df = (df.sort_values("created_at", ascending=False)
            .drop_duplicates(subset=key)
            .drop(columns=["created_at"])
            .reset_index(drop=True))
    if (dropped := before - len(df)):
        print(f"  Deduplicated: dropped {dropped} older duplicate runs.")

    if attack_filter:
        df = df[df["attack"].isin(attack_filter)]

    print(f"  Total: {len(df)} rows  |  "
          f"{df['attack'].nunique()} attacks  |  "
          f"{df['sdg'].nunique()} SDG methods  |  "
          f"{df['qi'].nunique()} QI variants  |  "
          f"{df['sample'].nunique()} samples")
    return df


# ── Pivot + delta computation ──────────────────────────────────────────────────

def _intersection_mean(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """For each row return mean of RA_{f} columns over the given feature list."""
    ra_cols = [f"RA_{f}" for f in features if f"RA_{f}" in df.columns]
    missing = [f for f in features if f"RA_{f}" not in df.columns]
    if missing:
        print(f"  WARNING: Missing RA_ columns for features: {missing}")
    if not ra_cols:
        raise ValueError(
            "No per-feature RA_ columns found in data. "
            "Re-run QI1 (and other QI variants) through run_qi_sweep.py to produce "
            "CSVs with per-feature scores; old sweep_results_*.csv files lack them."
        )
    return df[ra_cols].mean(axis=1)


def compute_pivot(df: pd.DataFrame, dataset: str,
                  include_tiny_qi1_delta: bool = True) -> pd.DataFrame:
    """Build (attack × qi) pivot using intersection-based RA means.

    Size section (QI_tiny, QI1, QI_large): scored on size_intersection features.
    Comp section (QI1, QI_behavioral):     scored on comp_intersection features.
    QI1 in the comp section is renamed to QI1_comp so both sections can coexist
    as separate columns in the pivot.

    If include_tiny_qi1_delta is True, also computes TINY_QI1_DELTA_LABEL as
    QI1 − QI_tiny scored on the broader tiny_qi1_intersection feature set.
    """
    cfg            = DATASET_CONFIGS[dataset]
    size_feats     = cfg["size_intersection"]
    comp_feats     = cfg["comp_intersection"]

    size_qis = ["QI_tiny", "QI1", "QI_large"]
    comp_qis = ["QI1", "QI_behavioral"]

    size_df = df[df["qi"].isin(size_qis)].copy()
    size_df["_ra"] = _intersection_mean(size_df, size_feats)
    size_avg = (size_df.groupby(["attack", "qi"])["_ra"]
                       .mean()
                       .unstack("qi"))

    comp_df = df[df["qi"].isin(comp_qis)].copy()
    comp_df["_ra"] = _intersection_mean(comp_df, comp_feats)
    comp_df["qi"]  = comp_df["qi"].replace({"QI1": "QI1_comp"})
    comp_avg = (comp_df.groupby(["attack", "qi"])["_ra"]
                       .mean()
                       .unstack("qi"))

    pivot = pd.concat([size_avg, comp_avg], axis=1)

    if include_tiny_qi1_delta:
        pw_feats = cfg.get("tiny_qi1_intersection")
        if pw_feats:
            pw_df = df[df["qi"].isin(["QI_tiny", "QI1"])].copy()
            pw_df["_ra"] = _intersection_mean(pw_df, pw_feats)
            pw_avg = (pw_df.groupby(["attack", "qi"])["_ra"]
                           .mean()
                           .unstack("qi"))
            if "QI1" in pw_avg.columns and "QI_tiny" in pw_avg.columns:
                pivot[TINY_QI1_DELTA_LABEL] = pw_avg["QI1"] - pw_avg["QI_tiny"]

    return pivot


def compute_deltas(pivot: pd.DataFrame) -> pd.DataFrame:
    """Append Δ columns as defined in DELTAS; return copy."""
    p = pivot.copy()
    for label, minuend, subtrahend in DELTAS:
        if minuend in p.columns and subtrahend in p.columns:
            p[label] = p[minuend] - p[subtrahend]
        else:
            missing = [c for c in (minuend, subtrahend) if c not in p.columns]
            print(f"  WARNING: cannot compute {label} — missing QI columns: {missing}")
            p[label] = float("nan")
    return p


def _n_obs(df: pd.DataFrame, attack: str, qi: str) -> int:
    """Number of (sdg × sample) observations averaged for this cell."""
    actual_qi = QI_DATA_ALIAS.get(qi, qi)
    return len(df[(df["attack"] == attack) & (df["qi"] == actual_qi)])


# ── LaTeX generation ───────────────────────────────────────────────────────────

def _fmt_main(val: float, decimals: int, is_bold: bool) -> str:
    s = f"{val:.{decimals}f}"
    return rf"\textbf{{{s}}}" if is_bold else s


def _fmt_delta(val: float, decimals: int) -> str:
    if np.isnan(val):
        return "---"
    sign = "$+$" if val >= 0 else "$-$"
    return f"{sign}{abs(val):.{decimals}f}"


def _ordered_attack_groups(pivot: pd.DataFrame,
                            attack_filter: list[str] | None) -> list[tuple[str, list[str]]]:
    """Return [(group_label, [attack, ...]), ...] in defined order."""
    seen, result = set(), []
    for label, attacks in ATTACK_GROUPS:
        present = [a for a in attacks
                   if a in pivot.index
                   and (attack_filter is None or a in attack_filter)]
        if present:
            result.append((label, present))
            seen.update(present)
    return result


def to_latex(pivot_with_deltas: pd.DataFrame,
             df_raw: pd.DataFrame,
             dataset: str,
             qi_variants: list[str],
             decimals: int = 2,
             sdg_filter: list[str] | None = None,
             include_tiny_qi1_delta: bool = True) -> str:

    cfg         = DATASET_CONFIGS[dataset]
    sizes       = cfg["qi_known_sizes"]
    dataset_dsp = cfg["display"]
    train_size  = cfg["size"]

    atk_groups = _ordered_attack_groups(pivot_with_deltas, attack_filter=None)

    # Collect all delta column labels (fixed deltas + optional pairwise delta)
    delta_labels = [d[0] for d in DELTAS]
    has_tiny_qi1 = (include_tiny_qi1_delta
                    and TINY_QI1_DELTA_LABEL in pivot_with_deltas.columns)
    if has_tiny_qi1:
        delta_labels = delta_labels + [TINY_QI1_DELTA_LABEL]

    # Column order: QI_tiny, QI1, [Δ†_{t→1}], QI_large, Δ_size  |  QI1_comp, QI_behavioral, Δ_comp
    size_qi_cols  = ["QI_tiny", "QI1", "QI_large"]
    comp_qi_cols  = ["QI1_comp", "QI_behavioral"]
    size_delta    = [d[0] for d in DELTAS][0] if DELTAS else None
    comp_delta    = [d[0] for d in DELTAS][1] if len(DELTAS) > 1 else None

    # Build ordered column list for the table
    all_cols: list[str] = []
    for q in size_qi_cols:
        if q in pivot_with_deltas.columns:
            all_cols.append(q)
        # Insert pairwise delta immediately after QI1
        if q == "QI1" and has_tiny_qi1:
            all_cols.append(TINY_QI1_DELTA_LABEL)
    if size_delta and size_delta in pivot_with_deltas.columns:
        all_cols.append(size_delta)
    all_cols += [q for q in comp_qi_cols if q in pivot_with_deltas.columns]
    if comp_delta and comp_delta in pivot_with_deltas.columns:
        all_cols.append(comp_delta)

    n_size_qi = sum(1 for q in size_qi_cols if q in pivot_with_deltas.columns)
    n_size    = (n_size_qi
                 + (1 if has_tiny_qi1 else 0)
                 + (1 if size_delta and size_delta in pivot_with_deltas.columns else 0))
    n_comp_qi = sum(1 for q in comp_qi_cols if q in pivot_with_deltas.columns)
    n_comp    = n_comp_qi + (1 if comp_delta and comp_delta in pivot_with_deltas.columns else 0)

    # tabular column spec: "l" + r-columns + separator after size group
    col_spec_parts = ["l"]
    for i, col in enumerate(all_cols):
        if i == n_size:
            col_spec_parts.append("|")   # vertical rule between size and composition sections
        col_spec_parts.append("r")
    col_spec = "".join(col_spec_parts)

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  % Requires: \usepackage{booktabs, graphicx}")
    lines.append(f"  % Dataset: {dataset}  |  Training size: {train_size:,}")
    if sdg_filter:
        lines.append(f"  % SDG filter: {sdg_filter}")
    else:
        lines.append(r"  % Values averaged over all SDG methods and 5 samples")
    lines.append(r"  \resizebox{\textwidth}{!}{%")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    # Row 1: section span headers
    # "Attack" + \multicolumn for size section + \multicolumn for composition section
    row1_cells = [""]
    if n_size > 0:
        row1_cells.append(
            rf"\multicolumn{{{n_size}}}{{c}}{{{QI_SECTION_LABELS['size']}}}"
        )
    if n_comp > 0:
        row1_cells.append(
            rf"\multicolumn{{{n_comp}}}{{c}}{{{QI_SECTION_LABELS['composition']}}}"
        )
    lines.append("    " + " & ".join(row1_cells) + r" \\")

    # cmidrule under each section
    # column indices (1-based): attack col = 1, size section starts at 2
    size_start = 2
    size_end   = size_start + n_size - 1
    comp_start = size_end + 1
    comp_end   = comp_start + n_comp - 1
    cmid_parts = []
    if n_size > 0:
        cmid_parts.append(rf"\cmidrule(lr){{{size_start}-{size_end}}}")
    if n_comp > 0:
        cmid_parts.append(rf"\cmidrule(l){{{comp_start}-{comp_end}}}")
    if cmid_parts:
        lines.append("    " + "".join(cmid_parts))

    # Row 2: column headers (QI display names + delta labels)
    def _col_header(col: str) -> str:
        if col in QI_DISPLAY:
            return QI_DISPLAY[col]
        return col  # delta labels are already LaTeX

    row2_cells = ["Attack"] + [_col_header(c) for c in all_cols]
    lines.append("    " + " & ".join(row2_cells) + r" \\")

    # Row 3: sub-header with "N known" for QI columns
    def _known_subheader(col: str) -> str:
        k = sizes.get(col)
        if k is not None:
            return f"({k} kn.)"
        if col in delta_labels:
            return ""
        return ""

    row3_cells = [""] + [_known_subheader(c) for c in all_cols]
    lines.append("    " + " & ".join(row3_cells) + r" \\")
    lines.append(r"    \midrule")

    # Per-section max values for bolding (bold within each section separately)
    def _section_max(attack: str, qi_cols: list[str]) -> float:
        vals = [
            pivot_with_deltas.at[attack, q]
            for q in qi_cols
            if attack in pivot_with_deltas.index and q in pivot_with_deltas.columns
            and not np.isnan(pivot_with_deltas.at[attack, q])
        ]
        return max(vals) if vals else float("nan")

    # Data rows
    first_group = True
    for group_label, attacks in atk_groups:
        if not first_group:
            lines.append(r"    \midrule")
        first_group = False

        for attack in attacks:
            display = ATTACK_DISPLAY.get(attack, attack.replace("_", r"\_"))
            cells   = [display]

            size_max = _section_max(attack, size_qi_cols)
            comp_max = _section_max(attack, comp_qi_cols)

            for col in all_cols:
                if attack not in pivot_with_deltas.index or col not in pivot_with_deltas.columns:
                    cells.append("---")
                    continue
                val = pivot_with_deltas.at[attack, col]
                if np.isnan(val):
                    cells.append("---")
                elif col in delta_labels:
                    cells.append(_fmt_delta(val, decimals))
                else:
                    n = _n_obs(df_raw, attack, col)
                    sec_max = size_max if col in size_qi_cols else comp_max
                    is_bold = (not np.isnan(sec_max)) and abs(val - sec_max) < 1e-9
                    fmt = _fmt_main(val, decimals, is_bold)
                    if n < 5:
                        fmt += "$^{*}$"
                    cells.append(fmt)

            lines.append("    " + " & ".join(cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  }% end resizebox")

    # Caption
    size_feats = cfg["size_intersection"]
    comp_feats = cfg["comp_intersection"]
    size_feat_str = ", ".join(r"\textit{" + f.replace("_", r"\_") + "}" for f in size_feats)
    comp_feat_str = ", ".join(r"\textit{" + f.replace("_", r"\_") + "}" for f in comp_feats)
    sdg_str = (f" SDG methods: {', '.join(sdg_filter)}." if sdg_filter
               else " Averaged over all SDG methods.")
    lines.append(
        r"  \caption{Reconstruction accuracy by quasi-identifier setting, "
        f"averaged over 5 samples.{sdg_str} "
        r"Dataset: \textit{" + dataset_dsp + r"}. "
        f"Training size: {train_size:,}. "
        r"Each cell is the mean RA over the \emph{intersection} of hidden features "
        r"across all QI variants in that section, so that scores are comparable. "
        r"\textbf{Size sweep} intersection "
        f"({len(size_feats)} features): {size_feat_str}. "
        r"\textbf{Composition contrast} intersection "
        f"({len(comp_feats)} features): {comp_feat_str}. "
        r"QI$_1$ appears twice (once per section) with different scores because each "
        r"section averages over a different feature set. "
        r"Bold = highest value in row within each section. "
        r"$\Delta_{\text{size}}$ = QI$_{\text{large}}$ $-$ QI$_{\text{tiny}}$; "
        r"$\Delta_{\text{comp}}$ = QI$_{\text{beh.}}$ $-$ QI$_1$ (matched $|\text{QI}|$)."
        + (
            r" $\Delta^{\dag}_{t{\to}1}$ = QI$_1$ $-$ QI$_{\text{tiny}}$ scored on the "
            r"QI$_{\text{tiny}}{\cap}$QI$_1$ intersection "
            f"({len(cfg['tiny_qi1_intersection'])} features), "
            r"a broader feature set than the three-way size-sweep intersection."
            if has_tiny_qi1 else ""
        )
        + r"}"
    )
    lines.append(r"  \label{tab:qi_analysis_" + dataset.replace("_", "") + r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── Plain-text preview ─────────────────────────────────────────────────────────

def print_preview(pivot_with_deltas: pd.DataFrame,
                  df_raw: pd.DataFrame,
                  dataset: str,
                  qi_variants: list[str],
                  decimals: int,
                  include_tiny_qi1_delta: bool = True) -> None:
    cfg   = DATASET_CONFIGS[dataset]
    sizes = cfg["qi_known_sizes"]

    delta_labels  = [d[0] for d in DELTAS]
    has_tiny_qi1  = (include_tiny_qi1_delta
                     and TINY_QI1_DELTA_LABEL in pivot_with_deltas.columns)
    if has_tiny_qi1:
        delta_labels = delta_labels + [TINY_QI1_DELTA_LABEL]

    size_qi_cols  = ["QI_tiny", "QI1", "QI_large"]
    comp_qi_cols  = ["QI1_comp", "QI_behavioral"]
    size_delta    = [d[0] for d in DELTAS][0] if DELTAS else None
    comp_delta    = [d[0] for d in DELTAS][1] if len(DELTAS) > 1 else None

    all_cols: list[str] = []
    for q in size_qi_cols:
        if q in pivot_with_deltas.columns:
            all_cols.append(q)
        if q == "QI1" and has_tiny_qi1:
            all_cols.append(TINY_QI1_DELTA_LABEL)
    if size_delta and size_delta in pivot_with_deltas.columns:
        all_cols.append(size_delta)
    all_cols += [q for q in comp_qi_cols if q in pivot_with_deltas.columns]
    if comp_delta and comp_delta in pivot_with_deltas.columns:
        all_cols.append(comp_delta)

    size_feats = cfg["size_intersection"]
    comp_feats = cfg["comp_intersection"]
    pw_feats   = cfg.get("tiny_qi1_intersection", [])

    atk_groups = _ordered_attack_groups(pivot_with_deltas, attack_filter=None)
    all_attacks = [a for _, attacks in atk_groups for a in attacks]

    # Header
    qi_headers = []
    for col in all_cols:
        k = sizes.get(col)
        if k is not None:
            qi_headers.append(f"{col} ({k}kn.)")
        elif col == TINY_QI1_DELTA_LABEL:
            qi_headers.append(f"Δ†(t→1) ({len(pw_feats)}ft)")
        elif col in delta_labels:
            short = col.replace("$\\Delta_{\\text{", "Δ_").replace("}}$", "")
            qi_headers.append(short)
        else:
            qi_headers.append(col)

    col_w   = max(len(h) for h in qi_headers) + 2
    atk_w   = max(len(a) for a in all_attacks) + 2 if all_attacks else 20
    divider = "-" * (atk_w + col_w * len(all_cols))
    print(f"\n{'='*70}")
    print(f"  QI Analysis — {cfg['display']} (size {cfg['size']:,})")
    print(f"  Size section features ({len(size_feats)}): {', '.join(size_feats)}")
    print(f"  Comp section features ({len(comp_feats)}): {', '.join(comp_feats)}")
    if has_tiny_qi1:
        print(f"  Pairwise Δ†(t→1) features ({len(pw_feats)}): {', '.join(pw_feats)}")
    print(f"  Values = intersection RA averaged over SDG methods × samples")
    print(f"{'='*70}")
    header_row = f"  {'Attack':<{atk_w}}" + "".join(f"{h:>{col_w}}" for h in qi_headers)
    print(header_row)
    print("  " + divider)

    prev_group = None
    for group_label, attacks in atk_groups:
        if prev_group is not None:
            print("  " + divider)
        prev_group = group_label
        for attack in attacks:
            cells = f"  {attack:<{atk_w}}"
            for col in all_cols:
                if attack not in pivot_with_deltas.index or col not in pivot_with_deltas.columns:
                    cells += f"{'---':>{col_w}}"
                    continue
                val = pivot_with_deltas.at[attack, col]
                if np.isnan(val):
                    cells += f"{'---':>{col_w}}"
                elif col in delta_labels:
                    sign = "+" if val >= 0 else "-"
                    cells += f"{sign}{abs(val):.{decimals}f}".rjust(col_w)
                else:
                    cells += f"{val:.{decimals}f}".rjust(col_w)
            print(cells)

    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QI analysis → side-by-side LaTeX table (size effect + composition contrast).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset",  required=True, choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset name.")
    parser.add_argument("--from-csv", nargs="+", default=None, metavar="CSV",
                        help="One or more CSV files (qi column distinguishes variants). "
                             "Glob patterns are expanded.")
    parser.add_argument("--out",      default=None,
                        help="Output .tex path. Default: qi_table_<dataset>.tex in this folder.")
    parser.add_argument("--decimals", type=int, default=2,
                        help="Decimal places (default 2).")
    parser.add_argument("--sdg",      nargs="+", default=None, metavar="SDG",
                        help="Restrict average to these SDG labels (default: all).")
    parser.add_argument("--attacks",  nargs="+", default=None, metavar="ATTACK",
                        help="Restrict rows to these attack names.")
    parser.add_argument("--size",     type=int, default=None,
                        help="Filter rows to this training size (for CSVs with a 'size' column).")
    parser.add_argument("--no-tiny-qi1-delta", action="store_true", default=False,
                        help="Omit the Δ†(t→1) column (QI_tiny→QI1 step on broader feature set).")
    args = parser.parse_args()

    dataset     = args.dataset
    qi_variants = QI_ORDER  # always load all four; render only those present in the data

    print(f"\nQI analysis table: {dataset}")
    print(f"QI variants: {qi_variants}")

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.from_csv:
        # Expand any glob patterns the shell didn't expand (e.g. on Windows)
        expanded: list[str] = []
        for pattern in args.from_csv:
            matches = glob.glob(pattern)
            expanded.extend(sorted(matches) if matches else [pattern])

        print(f"\nLoading {len(expanded)} CSV file(s):")
        df = load_from_csv(
            expanded,
            dataset=dataset,
            qi_variants=qi_variants,
            sdg_filter=args.sdg,
            size_filter=args.size,
            attack_filter=args.attacks,
        )
    else:
        print("\nFetching from WandB ...")
        df = fetch_from_wandb(
            dataset=dataset,
            qi_variants=qi_variants,
            sdg_filter=args.sdg,
            size_filter=args.size,
            attack_filter=args.attacks,
        )

    if df.empty:
        print("No data found.")
        sys.exit(1)

    # Diagnose which QI variants actually have data
    found_qi = set(df["qi"].unique())
    missing  = set(qi_variants) - found_qi
    if missing:
        print(f"\n  WARNING: no data found for QI variants: {sorted(missing)}")
        print("  These columns will show '---' in the table.")

    # ── Build table ────────────────────────────────────────────────────────────
    print(f"\n  Attacks present: {sorted(df['attack'].unique())}")
    print(f"  SDG methods:     {sorted(df['sdg'].unique())}")
    print(f"  QI variants:     {sorted(df['qi'].unique())}")

    include_tiny_qi1_delta = not args.no_tiny_qi1_delta

    pivot              = compute_pivot(df, dataset, include_tiny_qi1_delta)
    pivot_with_deltas  = compute_deltas(pivot)

    # ── Preview ────────────────────────────────────────────────────────────────
    print_preview(pivot_with_deltas, df, dataset, qi_variants, args.decimals,
                  include_tiny_qi1_delta)

    # ── Generate LaTeX ─────────────────────────────────────────────────────────
    latex = to_latex(
        pivot_with_deltas, df,
        dataset=dataset,
        qi_variants=qi_variants,
        decimals=args.decimals,
        sdg_filter=args.sdg,
        include_tiny_qi1_delta=include_tiny_qi1_delta,
    )

    # ── Write output ───────────────────────────────────────────────────────────
    script_dir = Path(__file__).parent
    out_path   = Path(args.out) if args.out else (script_dir / f"qi_table_{dataset}.tex")
    out_path.write_text(latex + "\n")
    print(f"LaTeX written to: {out_path}\n")

    print("─" * 70)
    print(latex)
    print("─" * 70)


if __name__ == "__main__":
    main()
