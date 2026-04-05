#!/usr/bin/env python
"""
Fetch per-feature RA scores from WandB (or local CSV), average across samples
and SDG methods, and write a LaTeX table.

Rows = attack methods (grouped by type, separated by \\midrule).
Cols = individual hidden features (with cardinality in header).
Values = RA per feature, averaged over all SDG methods and samples.

To restrict which SDG methods are averaged, edit SDG_FILTER at the top.

Usage:
    conda activate recon_
    python experiment_scripts/wandb_to_latex_by_feature.py
    python experiment_scripts/wandb_to_latex_by_feature.py --dataset cdc_diabetes --qi QI1
    python experiment_scripts/wandb_to_latex_by_feature.py --from-csv sweep_results.csv
    python experiment_scripts/wandb_to_latex_by_feature.py --out my_table.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── WandB config ───────────────────────────────────────────────────────────────

WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "main attack sweep 1"
DATASET       = "adult"   # default dataset; override with --dataset

# ── SDG filter ─────────────────────────────────────────────────────────────────
# Set to None to average over ALL SDG methods, or a list to restrict:
# SDG_FILTER = ["MST_eps1", "MST_eps3", "MST_eps10", "Synthpop", "TVAE"]
SDG_FILTER: list[str] | None = None

# ── Data paths ─────────────────────────────────────────────────────────────────

DATA_ROOT = Path("~/data/reconstruction_data").expanduser()

# Maps dataset name (as used in configs/WandB) → subdirectory under DATA_ROOT
DATASET_DIR: dict[str, str] = {
    "adult":               "adult",
    "cdc_diabetes":        "cdc_diabetes",
    "california":          "california",
    "nist_arizona_25feat": "nist_arizona_data",
    "nist_arizona_50feat": "nist_arizona_data",
    "nist_arizona_data":   "nist_arizona_data",
    "nist_sbo":            "nist_sbo",
}

# Suffix appended to size_N dir for feature-subset variants
DATASET_FEAT_SUFFIX: dict[str, str] = {
    "nist_arizona_25feat": "_25feat",
    "nist_arizona_50feat": "_50feat",
}

# ── Hidden feature definitions (mirrors minus_QIs in get_data.py) ──────────────

HIDDEN_FEATURES: dict[str, dict[str, list[str]]] = {
    "adult": {
        # Size sweep: QI_tiny (3) → QI1 (6) → QI_large (10)
        "QI_tiny":       ["workclass", "fnlwgt", "education", "education-num",
                          "marital-status", "occupation", "relationship",
                          "capital-gain", "capital-loss", "hours-per-week",
                          "native-country", "income"],
        "QI1":           ["workclass", "fnlwgt", "education-num", "occupation",
                          "relationship", "capital-gain", "capital-loss",
                          "hours-per-week", "income"],
        "QI_large":      ["fnlwgt", "education-num", "capital-gain",
                          "capital-loss", "income"],
        # Composition contrast (same size=6 as QI1): employment prior vs. demographic
        "QI_behavioral": ["age", "sex", "race", "native-country", "education",
                          "marital-status", "capital-gain", "capital-loss", "income"],
        "QI_linear":             ["income"],
        "QI_binary_sex":         ["sex"],
        "QI_linear_lowcard":     ["income"],
        "QI_binary_sex_lowcard": ["sex"],
        "QI_race_lowcard":             ["race"],
        "QI_joint_income_sex_lowcard": ["income", "sex"],
    },
    "cdc_diabetes": {
        # Size sweep: QI_tiny (4) → QI1 (10) → QI_large (16)
        "QI_tiny":       ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
                          "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                          "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
                          "MentHlth", "PhysHlth", "DiffWalk",
                          "Education", "Income", "Diabetes_binary"],
        "QI1":           ["Diabetes_binary", "Stroke", "HeartDiseaseorAttack", "CholCheck",
                          "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
                          "NoDocbcCost", "MentHlth", "PhysHlth", "DiffWalk"],
        "QI_large":      ["Diabetes_binary", "Stroke", "HeartDiseaseorAttack",
                          "HvyAlcoholConsump", "MentHlth", "PhysHlth"],
        # Composition contrast (same size=10 as QI1): lifestyle prior vs. demographic
        "QI_behavioral": ["Diabetes_binary", "Stroke", "HighBP", "HighChol", "BMI",
                          "Sex", "Age", "Education", "Income", "GenHlth",
                          "MentHlth", "PhysHlth"],
        "QI_linear":         ["Diabetes_binary"],
        "QI_binary_HighBP":  ["HighBP"],
        "QI_binary_Stroke":  ["Stroke"],
        "QI_joint_Diabetes_HighBP_Stroke_lowcard": ["Diabetes_binary", "HighBP", "Stroke"],
    },
    "california": {
        "QI1": ["MedInc", "AveRooms", "AveBedrms", "Population",
                "AveOccup", "MedHouseVal"],
    },
    "nist_arizona_25feat": {
        # Size sweep: QI1 (7) → QI_medium (12) → QI3 (18)
        "QI1":    ["AGE", "BPL", "CITIZEN", "DURUNEMP", "EDUC", "EMPSTAT",
                   "FAMSIZE", "FARM", "GQ", "HISPAN", "INCWAGE", "LABFORCE",
                   "MARST", "MIGRATE5", "NATIVITY", "OWNERSHP", "URBAN", "WKSWORK1"],
        "QI_medium": ["CITIZEN", "DURUNEMP", "EDUC", "FAMSIZE", "FARM", "GQ",
                      "INCWAGE", "MARST", "MIGRATE5", "NATIVITY", "OWNERSHP",
                      "URBAN", "WKSWORK1"],
        "QI3":    ["EDUC", "FARM", "GQ", "INCWAGE", "LABFORCE", "MIGRATE5", "NATIVITY"],
        "QI_binary_SEX":  ["SEX"],
        "QI_binary_FARM": ["FARM"],
        "QI_binary_URBAN": ["URBAN"],
    },
    "nist_arizona_50feat": {
        "QI1": ["INCWAGE", "VALUEH", "RENT", "OCC", "WKSWORK1",
                "HRSWORK1", "CLASSWKR", "EDUC"],
    },
    "nist_arizona_data": {
        "QI1": ["INCWAGE", "VALUEH", "RENT", "OCC", "WKSWORK1",
                "HRSWORK1", "CLASSWKR", "EDUC"],
        "QI2": ["INCWAGE", "VALUEH", "RENT", "OCC", "IND", "WKSWORK1",
                "HRSWORK1", "SEI", "OCCSCORE", "CLASSWKR"],
    },
}

# ── Display order & groupings ──────────────────────────────────────────────────

ATTACK_GROUPS = [
    ("Baselines",      ["Mode", "Random", "MeasureDeid"]),
    ("ML Classifiers", ["KNN", "NaiveBayes", "LogisticRegression", "SVM",
                        "RandomForest", "LightGBM"]),
    ("Neural",         ["MLP", "Attention", "AttentionAutoregressive"]),
    ("Partial SDG",    ["TabDDPM", "TabDDPMWithMLP", "ConditionedRePaint", "RePaint",
                        "PartialMST", "PartialMSTIndependent", "PartialMSTBounded"]),
    ("SOTA",           ["LinearReconstruction"]),
]

ATTACK_DISPLAY: dict[str, str] = {
    "Mode":                    "Mode",
    "Random":                  "Random",
    "MeasureDeid":             "MeasureDeid",
    "KNN":                     r"\textsc{knn}",
    "NaiveBayes":              "Naive Bayes",
    "LogisticRegression":      "Logistic Regression",
    "SVM":                     r"\textsc{svm}",
    "RandomForest":            "Random Forest",
    "LightGBM":                "LightGBM",
    "MLP":                     r"\textsc{mlp}",
    "Attention":               "Attention",
    "AttentionAutoregressive": "Attention (AR)",
    "TabDDPM":                 "TabDDPM",
    "ConditionedRePaint":      r"Cond.\ RePaint",
    "TabDDPMWithMLP":          r"TabDDPM+MLP",
    "LinearReconstruction":    "Linear Recon.",
    "PartialMST":              "MST",
    "PartialMSTIndependent":   "MST (1 ft./time)",
    "PartialMSTBounded":       r"MST $k{=}3$",
}

DATASET_DISPLAY: dict[str, str] = {
    "adult":               "Adult",
    "cdc_diabetes":        "CDC Diabetes",
    "california":          "California Housing",
    "nist_arizona_data":   "NIST Arizona (full)",
    "nist_arizona_25feat": "NIST Arizona (25 feat.)",
    "nist_arizona_50feat": "NIST Arizona (50 feat.)",
    "nist_sbo":            "NIST SBO",
}


# ── Cardinality helper ─────────────────────────────────────────────────────────

def get_cardinalities(dataset: str, size: int | None = None) -> dict[str, int]:
    """Return {col: nunique} reflecting what attacks actually see.

    Prefers sample_00/train.csv (preprocessing already applied, e.g. deduped
    income labels in adult) over full_data.csv.  Falls back to full_data.csv
    when no size is given or the sample dir doesn't exist.
    """
    subdir = DATASET_DIR.get(dataset, dataset)

    if size is not None:
        feat_suffix = DATASET_FEAT_SUFFIX.get(dataset, "")
        train_path = DATA_ROOT / subdir / f"size_{size}{feat_suffix}" / "sample_00" / "train.csv"
        if train_path.exists():
            df = pd.read_csv(train_path)
            print(f"  Cardinalities from: {train_path}")
            return {col: int(df[col].nunique()) for col in df.columns}
        print(f"  Warning: {train_path} not found, falling back to full_data.csv")

    path = DATA_ROOT / subdir / "full_data.csv"
    if not path.exists():
        print(f"  Warning: {path} not found — cardinalities will be omitted.")
        return {}
    df = pd.read_csv(path)
    print(f"  Cardinalities from: {path}")
    return {col: int(df[col].nunique()) for col in df.columns}


# ── WandB helpers ──────────────────────────────────────────────────────────────

def _sdg_label(method: str, params: dict | None) -> str:
    params = params or {}
    eps = params.get("epsilon") or params.get("eps")
    return f"{method}_eps{float(eps):g}" if eps is not None else method


def _fetch_group_long(api, path: str, group: str,
                      effective_dataset: str,
                      attack_filter: list[str] | None,
                      qi_filter: str | None,
                      sdg_filter: list[str] | None) -> tuple[list[dict], int]:
    """Fetch one WandB group; return (rows, n_skipped)."""
    server_filters: dict = {"group": group}
    if attack_filter:
        server_filters["config.attack_method"] = {"$in": attack_filter}
    if effective_dataset:
        server_filters["config.dataset"] = effective_dataset

    print(f"  Querying group={group!r} ...")
    runs = api.runs(path, filters=server_filters)

    rows, skipped = [], 0
    for run in runs:
        cfg  = run.config
        summ = run.summary

        attack      = cfg.get("attack_method")
        sdg_method  = cfg.get("sdg_method")
        sdg_params  = cfg.get("sdg_params") or {}
        qi          = cfg.get("qi")
        sample      = cfg.get("sample_idx")
        dataset_cfg = cfg.get("dataset") or {}
        dataset     = dataset_cfg.get("name") if isinstance(dataset_cfg, dict) else dataset_cfg

        if None in (attack, sdg_method, qi, sample):
            skipped += 1
            continue
        if effective_dataset and dataset != effective_dataset:
            skipped += 1
            continue
        if qi_filter and qi != qi_filter:
            continue
        if attack_filter and attack not in attack_filter:
            skipped += 1
            continue

        sdg = _sdg_label(sdg_method, sdg_params)
        if sdg_filter and sdg not in sdg_filter:
            continue

        feat_scores = {
            k[3:]: float(v)
            for k, v in summ.items()
            if (k.startswith("RA_")
                and k not in ("RA_mean",)
                and not k.startswith("RA_train_")
                and not k.startswith("RA_nontraining_")
                and not k.startswith("RA_delta_")
                and not k.startswith("RA_row_"))
        }
        if not feat_scores:
            skipped += 1
            continue

        for feat, score in feat_scores.items():
            rows.append({
                "attack":     attack,
                "sdg":        sdg,
                "qi":         qi,
                "sample":     int(sample),
                "feature":    feat,
                "ra_score":   score,
                "created_at": run.created_at,
            })

    return rows, skipped


def fetch_runs_long(groups: list[str], qi_filter: str | None,
                    attack_filter: list[str] | None,
                    dataset_filter: str | None,
                    sdg_filter: list[str] | None) -> pd.DataFrame:
    """Pull finished runs from one or more WandB groups; deduplicate by most recent run."""
    import wandb
    api    = wandb.Api(timeout=60)
    entity = api.default_entity
    path   = f"{entity}/{WANDB_PROJECT}"

    effective_dataset = dataset_filter if dataset_filter is not None else DATASET
    if effective_dataset:
        print(f"Dataset filter: {effective_dataset!r}")

    all_rows, total_skipped = [], 0
    for group in groups:
        rows, skipped = _fetch_group_long(
            api, path, group, effective_dataset,
            attack_filter, qi_filter, sdg_filter,
        )
        print(f"    → {len(rows)} (run, feature) records  ({skipped} skipped)")
        all_rows.extend(rows)
        total_skipped += skipped

    if total_skipped:
        print(f"  Total skipped: {total_skipped} runs (missing fields or no per-feature scores).")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # Deduplicate across groups: keep most recent run per (attack, sdg, qi, sample, feature)
    key = ["attack", "sdg", "qi", "sample", "feature"]
    before = len(df)
    df = (df.sort_values("created_at", ascending=False)
            .drop_duplicates(subset=key)
            .drop(columns=["created_at"])
            .reset_index(drop=True))
    if (dropped := before - len(df)):
        print(f"  Deduplication across groups: dropped {dropped} older duplicate entries.")

    print(f"  {len(df)} (run, feature) records remaining after dedup.")
    return df


# ── CSV helpers ────────────────────────────────────────────────────────────────

def load_csv_long(csv_paths: list[str],
                  qi_filter: str | None,
                  attack_filter: list[str] | None,
                  sdg_filter: list[str] | None,
                  size_filter: int | None) -> pd.DataFrame:
    """Load one or more sweep CSVs with per-feature columns; return long DataFrame."""
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        print(f"Loaded {p}: {len(df)} rows")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Drop rows with no ra_mean (failed runs)
    before = len(df)
    df = df[df["ra_mean"].notna()]
    if (dropped := before - len(df)):
        print(f"  Dropped {dropped} rows with missing ra_mean (failed runs).")

    if qi_filter:
        df = df[df["qi"] == qi_filter]
    if size_filter is not None and "size" in df.columns:
        df = df[(df["size"] == size_filter) | df["size"].isna()]
    if attack_filter:
        df = df[df["attack"].isin(attack_filter)]
    if sdg_filter:
        df = df[df["sdg"].isin(sdg_filter)]

    # Find per-feature columns: RA_{feat} (not RA_mean, RA_train_*, etc.)
    feat_cols = [
        c for c in df.columns
        if (c.startswith("RA_")
            and c != "RA_mean"
            and not c.startswith("RA_train_")
            and not c.startswith("RA_nontraining_")
            and not c.startswith("RA_delta_"))
    ]
    if not feat_cols:
        print("  ERROR: No per-feature RA_ columns found in CSV. "
              "Re-run the sweep to generate CSVs with per-feature scores "
              "(run_production_sweep.py and run_cdc_sweep.py now write these).")
        return pd.DataFrame()

    # Melt to long format
    id_vars = ["attack", "sdg", "qi", "sample"]
    available_id = [c for c in id_vars if c in df.columns]
    long = df[available_id + feat_cols].melt(
        id_vars=available_id,
        value_vars=feat_cols,
        var_name="feature",
        value_name="ra_score",
    )
    long["feature"] = long["feature"].str[3:]  # strip "RA_" prefix
    long = long[long["ra_score"].notna()].reset_index(drop=True)

    # Dedup: for same (attack, sdg, qi, sample, feature), keep last (latest CSV wins)
    key = ["attack", "sdg", "qi", "sample", "feature"]
    before = len(long)
    long = long.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    if (dropped := before - len(long)):
        print(f"  Deduplicated: dropped {dropped} duplicate (run, feature) entries.")

    print(f"  {len(long)} (run, feature) records from {long['attack'].nunique()} attacks.")
    return long


# ── Table construction ─────────────────────────────────────────────────────────


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Average ra_score over (sdg × sample); return pivot (index=attack, columns=feature)."""
    avg   = df.groupby(["attack", "feature"])["ra_score"].mean()
    pivot = avg.unstack("feature")
    return pivot


def _ordered_features(pivot: pd.DataFrame, dataset: str, qi: str) -> list[str]:
    """Features in the preferred order from HIDDEN_FEATURES, extras appended alphabetically."""
    preferred = HIDDEN_FEATURES.get(dataset, {}).get(qi, [])
    ordered = [f for f in preferred if f in pivot.columns]
    extra   = sorted(f for f in pivot.columns if f not in ordered)
    return ordered + extra


def _ordered_attack_groups(pivot: pd.DataFrame) -> list[tuple[str, list[str]]]:
    seen, result = set(), []
    for label, attacks in ATTACK_GROUPS:
        present = [a for a in attacks if a in pivot.index]
        if present:
            result.append((label, present))
            seen.update(present)
    leftover = [a for a in pivot.index if a not in seen]
    if leftover:
        result.append(("Other", sorted(leftover)))
    return result


def _n_obs(df: pd.DataFrame, attack: str, feature: str) -> int:
    """Number of (sdg, sample) observations averaged for this cell."""
    sub = df[(df["attack"] == attack) & (df["feature"] == feature)]
    return len(sub)


# ── LaTeX generation ───────────────────────────────────────────────────────────

def to_latex(pivot: pd.DataFrame, df_raw: pd.DataFrame,
             group: str, qi: str, dataset: str,
             cardinalities: dict[str, int],
             decimals: int = 3, size: int | None = None) -> str:

    features   = _ordered_features(pivot, dataset, qi)
    atk_groups = _ordered_attack_groups(pivot)

    n_cols   = len(features)
    col_spec = "l" + "r" * n_cols

    # Column headers: "feature\n(k=N)" or just "feature" if cardinality unavailable
    def _feat_header(f: str) -> str:
        k = cardinalities.get(f)
        label = f.replace("_", r"\_").replace("-", r"\mbox{-}")
        return f"{label} ($k{{=}}{k}$)" if k is not None else label

    col_headers = [_feat_header(f) for f in features]

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  % Requires: \usepackage{booktabs,rotating,graphicx}")
    lines.append(f"  % WandB group: {group!r}   QI: {qi}   Dataset: {dataset}")
    if SDG_FILTER:
        lines.append(f"  % SDG filter: {SDG_FILTER}")
    lines.append(r"  \resizebox{\textwidth}{!}{%")
    lines.append(r"  \begin{tabular}{" + col_spec + r"}")
    lines.append(r"    \toprule")

    header_cells = ["Attack"] + [
        r"\rotatebox{60}{" + h + r"}" for h in col_headers
    ]
    lines.append("    " + " & ".join(header_cells) + r" \\")
    lines.append(r"    \midrule")

    # Determine minimum expected observations (samples × SDG methods)
    # Use median n_obs across non-NaN cells as a heuristic threshold
    all_n = [
        _n_obs(df_raw, atk, f)
        for _, attacks in atk_groups for atk in attacks
        for f in features
        if not np.isnan(
            pivot.at[atk, f]
            if (atk in pivot.index and f in pivot.columns)
            else float("nan")
        )
    ]
    median_n = int(np.median(all_n)) if all_n else 1
    low_threshold = max(1, median_n // 2)

    first_group = True
    for _group_label, attacks in atk_groups:
        if not first_group:
            lines.append(r"    \midrule")
        first_group = False

        for attack in attacks:
            display_name = ATTACK_DISPLAY.get(attack, attack.replace("_", r"\_"))
            cells = [display_name]
            for feat in features:
                val = (
                    pivot.at[attack, feat]
                    if (attack in pivot.index and feat in pivot.columns)
                    else float("nan")
                )
                n = _n_obs(df_raw, attack, feat)
                fmt = f".{decimals}f"
                if np.isnan(val):
                    cells.append("---")
                elif n < low_threshold:
                    cells.append(f"{val:{fmt}}$^*$")
                else:
                    cells.append(f"{val:{fmt}}")
            lines.append("    " + " & ".join(cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  }% end resizebox")

    any_flagged = any(
        _n_obs(df_raw, atk, f) < low_threshold
        for _, attacks in atk_groups for atk in attacks
        for f in features
        if not np.isnan(
            pivot.at[atk, f]
            if (atk in pivot.index and f in pivot.columns)
            else float("nan")
        )
    )

    dataset_display = DATASET_DISPLAY.get(dataset, dataset)
    size_str = f" Training size: {size:,}." if size else ""
    sdg_str  = (f" SDG methods included: {', '.join(SDG_FILTER)}."
                if SDG_FILTER else " Averaged over all SDG methods.")
    lines.append(
        r"  \caption{Mean reconstruction accuracy (\texttt{RA}) per feature, "
        r"averaged over samples and SDG methods. "
        f"Dataset: \\textit{{{dataset_display}}}.{size_str}{sdg_str}"
    )
    lines.append(f"           QI variant: {qi}.")
    if any_flagged:
        lines.append(r"           $^*$Fewer observations than expected for this cell.}")
    else:
        lines.append(r"           }")
    lines.append(r"  \label{tab:ra_by_feature}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WandB/CSV → LaTeX table of RA per feature (rows=attacks, cols=features)."
    )
    parser.add_argument("--group", nargs="+", default=[WANDB_GROUP],
                        help=f"WandB run group(s) to fetch from (default: {WANDB_GROUP!r}). "
                             "Pass multiple to merge across groups, e.g. --group cdc-sweep-1 'main attack sweep 1'.")
    parser.add_argument("--qi", default="QI1",
                        help="QI variant to include (default: QI1).")
    parser.add_argument("--dataset", default=DATASET,
                        help=f"Dataset name (default: {DATASET!r}).")
    parser.add_argument("--out", default=None,
                        help="Output .tex path (default: experiment_scripts/ra_by_feature_<dataset>_<qi>.tex).")
    parser.add_argument("--decimals", type=int, default=3,
                        help="Decimal places (default: 3).")
    parser.add_argument("--attacks", nargs="+", default=None, metavar="ATTACK",
                        help="Only include these attack methods.")
    parser.add_argument("--from-csv", nargs="+", default=None, metavar="CSV",
                        help="Load from local sweep CSV(s) instead of querying WandB. "
                             "Expected columns: sample, sdg, attack, qi, ra_mean, RA_<feat>...")
    parser.add_argument("--size", type=int, default=None,
                        help="Filter to rows where 'size' column equals this value.")
    args = parser.parse_args()

    qi_filter = args.qi

    # ── Load data ──
    if args.from_csv:
        df = load_csv_long(
            args.from_csv,
            qi_filter=qi_filter,
            attack_filter=args.attacks,
            sdg_filter=SDG_FILTER,
            size_filter=args.size,
        )
    else:
        dataset_filter = args.dataset if args.dataset.lower() != "all" else ""
        df = fetch_runs_long(
            args.group,   # list of groups
            qi_filter=qi_filter,
            attack_filter=args.attacks,
            dataset_filter=dataset_filter,
            sdg_filter=SDG_FILTER,
        )

    if df.empty:
        print("No data found.")
        return

    print(f"\n  {len(df)} (run, feature) records  |  "
          f"{df['attack'].nunique()} attacks  |  "
          f"{df['feature'].nunique()} features  |  "
          f"{df['sdg'].nunique()} SDG methods  |  "
          f"{df['sample'].nunique()} samples")
    print(f"  Attacks : {sorted(df['attack'].unique())}")
    print(f"  Features: {sorted(df['feature'].unique())}")

    # ── Cardinalities ──
    cardinalities = get_cardinalities(args.dataset, size=args.size)

    # ── Build table ──
    pivot = build_pivot(df)
    latex = to_latex(
        pivot, df,
        group=", ".join(args.group), qi=args.qi, dataset=args.dataset,
        cardinalities=cardinalities,
        decimals=args.decimals, size=args.size,
    )

    # ── Write output ──
    out_path = Path(args.out) if args.out else (
        Path(__file__).parent / f"ra_by_feature_{args.dataset}_{args.qi}.tex"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex + "\n")
    print(f"\nLaTeX table written to: {out_path}")

    print("\n" + "─" * 70)
    print(latex)
    print("─" * 70)

    # Plain-text pivot preview
    print("\nPivot (plain text preview):")
    features = _ordered_features(pivot, args.dataset, args.qi)
    avail = [f for f in features if f in pivot.columns]
    print(pivot[avail].to_string(float_format=lambda x: f"{x:.{args.decimals}f}"))


if __name__ == "__main__":
    main()
