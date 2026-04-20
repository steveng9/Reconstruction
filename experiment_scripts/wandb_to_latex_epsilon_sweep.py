#!/usr/bin/env python
"""
RA per feature vs. privacy budget (epsilon).

Rows    = hidden features (with cardinality in header)
Columns = MST epsilon variants | AIM epsilon variants  (two-level grouped header)
Values  = RA per (feature, SDG) averaged over samples and the selected attacks

Use --attacks to fix a single attack or average over several.
Use --from-csv to load local results without hitting WandB.

Usage:
    conda activate recon_
    python experiment_scripts/wandb_to_latex_epsilon_sweep.py
    python experiment_scripts/wandb_to_latex_epsilon_sweep.py \\
        --from-csv experiment_scripts/sweep_results_adult_20260401_214300.csv \\
        --dataset adult --qi QI1 --size 10000 --attacks RandomForest
    python experiment_scripts/wandb_to_latex_epsilon_sweep.py \\
        --attacks RandomForest LightGBM MLP \\
        --dataset adult --qi QI1
    python experiment_scripts/wandb_to_latex_epsilon_sweep.py \\
        --from-csv sweep.csv --dataset adult --size 10000 \\
        --attacks RandomForest --out my_eps_table.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ── WandB config ──────────────────────────────────────────────────────────────

WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "main attack sweep 1"
DATASET       = "adult"

# ── Default attacks to average over ──────────────────────────────────────────

DEFAULT_ATTACKS = ["RandomForest"]

# ── Epsilon column ordering (MST then AIM, by increasing ε) ──────────────────
# Columns absent from the data are silently skipped.

EPS_COL_ORDER = [
    "MST_eps0.1", "MST_eps0.3", "MST_eps1",  "MST_eps3",
    "MST_eps10",  "MST_eps30",  "MST_eps100", "MST_eps300", "MST_eps1000",
    "AIM_eps0.3", "AIM_eps1",   "AIM_eps3",   "AIM_eps10",  "AIM_eps100",
]

# ── Label remapping (mirrors the other scripts) ───────────────────────────────

LABEL_REMAP: dict[str, str] = {
    "MarginalRF_global":         "MarginalRF_mst_global",
    "MarginalRF_local_k50":      "MarginalRF_mst_local_50",
    "MarginalRF_local_k100":     "MarginalRF_mst_local_100",
    "MarginalRF_local_k200":     "MarginalRF_mst_local_200",
    "MarginalRF_mst_local":      "MarginalRF_mst_local_100",
    "MarginalRF_complete_local": "MarginalRF_complete_local_100",
    "MarginalRF_topk_local":     "MarginalRF_topk_local_100",
}

# ── Data path helpers ─────────────────────────────────────────────────────────

DATA_ROOT = Path("~/data/reconstruction_data").expanduser()

DATASET_DIR: dict[str, str] = {
    "adult":               "adult",
    "cdc_diabetes":        "cdc_diabetes",
    "california":          "california",
    "nist_arizona_25feat": "nist_arizona_data",
    "nist_arizona_50feat": "nist_arizona_data",
    "nist_arizona_data":   "nist_arizona_data",
    "nist_sbo":            "nist_sbo",
}

DATASET_FEAT_SUFFIX: dict[str, str] = {
    "nist_arizona_25feat": "_25feat",
    "nist_arizona_50feat": "_50feat",
}

# ── Hidden feature definitions (mirrors wandb_to_latex_by_feature.py) ─────────

HIDDEN_FEATURES: dict[str, dict[str, list[str]]] = {
    "adult": {
        "QI_tiny":       ["workclass", "fnlwgt", "education", "education-num",
                          "marital-status", "occupation", "relationship",
                          "capital-gain", "capital-loss", "hours-per-week",
                          "native-country", "income"],
        "QI1":           ["workclass", "fnlwgt", "education-num", "occupation",
                          "relationship", "capital-gain", "capital-loss",
                          "hours-per-week", "income"],
        "QI_large":      ["fnlwgt", "education-num", "capital-gain",
                          "capital-loss", "income"],
        "QI_behavioral": ["age", "sex", "race", "native-country", "education",
                          "marital-status", "capital-gain", "capital-loss", "income"],
    },
    "cdc_diabetes": {
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
    },
    "california": {
        "QI1": ["MedInc", "AveRooms", "AveBedrms", "Population",
                "AveOccup", "MedHouseVal"],
    },
    "nist_arizona_25feat": {
        "QI1":    ["AGE", "BPL", "CITIZEN", "DURUNEMP", "EDUC", "EMPSTAT",
                   "FAMSIZE", "FARM", "GQ", "HISPAN", "INCWAGE", "LABFORCE",
                   "MARST", "MIGRATE5", "NATIVITY", "OWNERSHP", "URBAN", "WKSWORK1"],
        "QI_medium": ["CITIZEN", "DURUNEMP", "EDUC", "FAMSIZE", "FARM", "GQ",
                      "INCWAGE", "MARST", "MIGRATE5", "NATIVITY", "OWNERSHP",
                      "URBAN", "WKSWORK1"],
        "QI3":    ["EDUC", "FARM", "GQ", "INCWAGE", "LABFORCE", "MIGRATE5", "NATIVITY"],
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

DATASET_DISPLAY: dict[str, str] = {
    "adult":               "Adult",
    "cdc_diabetes":        "CDC Diabetes",
    "california":          "California Housing",
    "nist_arizona_data":   "NIST Arizona (full)",
    "nist_arizona_25feat": "NIST Arizona (25 feat.)",
    "nist_arizona_50feat": "NIST Arizona (50 feat.)",
    "nist_sbo":            "NIST SBO",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sdg_label(method: str, params: dict | None) -> str:
    params = params or {}
    eps = params.get("epsilon") or params.get("eps")
    return f"{method}_eps{float(eps):g}" if eps is not None else method


def get_cardinalities(dataset: str, size: int | None = None) -> dict[str, int]:
    """Return {col: nunique} from sample_00/train.csv, or full_data.csv as fallback."""
    subdir = DATASET_DIR.get(dataset, dataset)
    if size is not None:
        feat_suffix = DATASET_FEAT_SUFFIX.get(dataset, "")
        train_path = (DATA_ROOT / subdir
                      / f"size_{size}{feat_suffix}" / "sample_00" / "train.csv")
        if train_path.exists():
            df = pd.read_csv(train_path)
            print(f"  Cardinalities from: {train_path}")
            return {col: int(df[col].nunique()) for col in df.columns}
        print(f"  Warning: {train_path} not found, falling back to full_data.csv")
    path = DATA_ROOT / subdir / "full_data.csv"
    if not path.exists():
        print(f"  Warning: {path} not found — cardinalities omitted.")
        return {}
    df = pd.read_csv(path)
    print(f"  Cardinalities from: {path}")
    return {col: int(df[col].nunique()) for col in df.columns}


def _ordered_cols(pivot: pd.DataFrame) -> list[str]:
    """EPS_COL_ORDER columns present in the pivot (silently skip absent ones)."""
    return [c for c in EPS_COL_ORDER if c in pivot.columns]


def _ordered_features(pivot: pd.DataFrame, dataset: str, qi: str) -> list[str]:
    preferred = HIDDEN_FEATURES.get(dataset, {}).get(qi, [])
    ordered = [f for f in preferred if f in pivot.index]
    extra   = sorted(f for f in pivot.index if f not in ordered)
    return ordered + extra


def _n_obs(df: pd.DataFrame, feature: str, sdg: str) -> int:
    """Number of (attack, sample) observations for this (feature, sdg) cell."""
    return len(df[(df["feature"] == feature) & (df["sdg"] == sdg)])


# ── WandB fetch ───────────────────────────────────────────────────────────────

def fetch_runs_long(groups: list[str],
                    qi_filter: str | None,
                    attack_filter: list[str] | None,
                    dataset_filter: str | None) -> pd.DataFrame:
    import wandb
    api    = wandb.Api(timeout=60)
    entity = api.default_entity
    path   = f"{entity}/{WANDB_PROJECT}"

    effective_dataset = dataset_filter if dataset_filter is not None else DATASET
    if effective_dataset:
        print(f"Dataset filter: {effective_dataset!r}")

    all_rows, total_skipped = [], 0
    for group in groups:
        server_filters: dict = {
            "group":           group,
            "config.sdg_method": {"$in": ["MST", "AIM"]},
        }
        if attack_filter:
            server_filters["config.attack_method"] = {"$in": attack_filter}
        if effective_dataset:
            server_filters["config.dataset"] = effective_dataset

        print(f"  Querying group={group!r} (MST/AIM only) ...")
        runs = api.runs(path, filters=server_filters)

        rows, skipped = [], 0
        for run in runs:
            cfg  = run.config
            summ = run.summary

            attack      = cfg.get("attack_label") or cfg.get("attack_method")
            attack      = LABEL_REMAP.get(attack, attack)
            sdg_method  = cfg.get("sdg_method")
            sdg_params  = cfg.get("sdg_params") or {}
            qi          = cfg.get("qi")
            sample      = cfg.get("sample_idx")
            dataset_cfg = cfg.get("dataset") or {}
            dataset     = (dataset_cfg.get("name") if isinstance(dataset_cfg, dict)
                           else dataset_cfg)

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
            if not (sdg.startswith("MST_") or sdg.startswith("AIM_")):
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

        print(f"    → {len(rows)} (run, feature) records  ({skipped} skipped)")
        all_rows.extend(rows)
        total_skipped += skipped

    if total_skipped:
        print(f"  Total skipped: {total_skipped} runs.")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    key = ["attack", "sdg", "qi", "sample", "feature"]
    before = len(df)
    df = (df.sort_values("created_at", ascending=False)
            .drop_duplicates(subset=key)
            .drop(columns=["created_at"])
            .reset_index(drop=True))
    if (dropped := before - len(df)):
        print(f"  Deduplication: dropped {dropped} older duplicate entries.")
    return df


# ── CSV load ──────────────────────────────────────────────────────────────────

def load_csv_long(csv_paths: list[str],
                  qi_filter: str | None,
                  attack_filter: list[str] | None,
                  size_filter: int | None,
                  dataset_filter: str | None) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        print(f"Loaded {p}: {len(df)} rows")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    before = len(df)
    df = df[df["ra_mean"].notna()]
    if (dropped := before - len(df)):
        print(f"  Dropped {dropped} rows with missing ra_mean (failed runs).")

    # label → attack substitution (for MarginalRF variant labels, etc.)
    if "label" in df.columns:
        mask = df["label"].notna() & (df["label"].astype(str).str.strip() != "")
        df.loc[mask, "attack"] = df.loc[mask, "label"]
        df = df.drop(columns=["label"])
    df["attack"] = df["attack"].map(lambda x: LABEL_REMAP.get(x, x))

    if qi_filter:
        df = df[df["qi"] == qi_filter]
    if size_filter is not None and "size" in df.columns:
        # NaN passthrough ONLY for truly old single-context CSVs (no 'dataset' col → NaN
        # after concat). Rows that have an explicit dataset but no size column are from
        # newer multi-context CSVs and must match explicitly — we don't know their size.
        has_explicit_size = df["size"].notna()
        is_old_csv = (
            df["dataset"].isna()
            if "dataset" in df.columns
            else pd.Series(True, index=df.index)
        )
        df = df[(has_explicit_size & (df["size"] == size_filter)) | (~has_explicit_size & is_old_csv)]
    if dataset_filter and dataset_filter.lower() != "all" and "dataset" in df.columns:
        # Keep rows matching the requested dataset; also keep rows from CSVs that
        # had no 'dataset' column (old single-dataset files that predate the column).
        df = df[(df["dataset"] == dataset_filter) | df["dataset"].isna()]
    if attack_filter:
        df = df[df["attack"].isin(attack_filter)]

    # Keep only MST/AIM rows
    df = df[df["sdg"].str.startswith("MST_") | df["sdg"].str.startswith("AIM_")]

    # Find per-feature columns: RA_{feat} (exclude RA_mean, memorisation metrics)
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
              "Re-run the sweep so CSVs contain per-feature scores.")
        return pd.DataFrame()

    # Carry 'dataset' through the melt so it can anchor the dedup key.
    id_vars = [c for c in ["dataset", "attack", "sdg", "qi", "sample"] if c in df.columns]
    long = df[id_vars + feat_cols].melt(
        id_vars=id_vars,
        value_vars=feat_cols,
        var_name="feature",
        value_name="ra_score",
    )
    long["feature"] = long["feature"].str[3:]   # strip "RA_" prefix
    long = long[long["ra_score"].notna()].reset_index(drop=True)

    # Include dataset in the key so rows from different datasets never overwrite
    # each other even when both survive the (optional) filter above.
    key = [c for c in ["dataset", "attack", "sdg", "qi", "sample", "feature"] if c in long.columns]
    before = len(long)
    long = long.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    if (dropped := before - len(long)):
        print(f"  Deduplicated: dropped {dropped} duplicate (run, feature) entries.")

    print(f"  {len(long)} (run, feature) records | "
          f"{long['attack'].nunique()} attacks | "
          f"{long['sdg'].nunique()} SDGs | "
          f"{long['feature'].nunique()} features")
    return long


# ── Pivot ─────────────────────────────────────────────────────────────────────

def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Average ra_score over (attack × sample); rows = feature, cols = sdg."""
    avg = df.groupby(["feature", "sdg"])["ra_score"].mean()
    return avg.unstack("sdg")


# ── LaTeX generation ──────────────────────────────────────────────────────────

def to_latex(pivot: pd.DataFrame, df_raw: pd.DataFrame,
             group: str, qi: str, dataset: str,
             cardinalities: dict[str, int],
             attacks: list[str],
             decimals: int = 3,
             size: int | None = None) -> str:

    cols     = _ordered_cols(pivot)
    features = _ordered_features(pivot, dataset, qi)

    mst_cols = [c for c in cols if c.startswith("MST_")]
    aim_cols = [c for c in cols if c.startswith("AIM_")]
    n_mst, n_aim = len(mst_cols), len(aim_cols)

    col_spec = "l" + "r" * (n_mst + n_aim)

    def _eps_label(c: str) -> str:
        eps = c.split("_eps")[1]
        return f"$\\varepsilon{{=}}{eps}$"

    col_headers = [_eps_label(c) for c in cols]

    # Flagging threshold: flag cells with fewer observations than half the median
    all_n = [
        _n_obs(df_raw, f, s)
        for f in features for s in cols
        if not np.isnan(
            pivot.at[f, s]
            if (f in pivot.index and s in pivot.columns) else float("nan")
        )
    ]
    median_n = int(np.median(all_n)) if all_n else 1
    low_threshold = max(1, median_n // 2)
    any_flagged = any(
        _n_obs(df_raw, f, s) < low_threshold
        for f in features for s in cols
        if not np.isnan(
            pivot.at[f, s]
            if (f in pivot.index and s in pivot.columns) else float("nan")
        )
    )

    # Caption (placed above the table)
    dataset_display = DATASET_DISPLAY.get(dataset, dataset)
    _sz   = f" ($N={size:,}$ rows)" if size else ""
    _atks = ", ".join(f"\\texttt{{{a}}}" for a in attacks)
    _flg  = r" $^*$Fewer than expected observations." if any_flagged else ""
    caption = (
        f"Reconstruction accuracy (\\texttt{{RA}}) per feature "
        f"as a function of privacy budget $\\varepsilon$. "
        f"Dataset: \\textit{{{dataset_display}}}{_sz}. "
        f"QI variant: \\texttt{{{qi}}}. "
        f"Averaged over {len(attacks)} attack(s): {_atks}.{_flg}"
    )

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  % Requires: \usepackage{booktabs,graphicx}")
    lines.append(f"  % WandB group: {group!r}   QI: {qi}   Dataset: {dataset}")
    lines.append(f"  % Attacks averaged: {', '.join(attacks)}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(r"  \label{tab:ra_vs_epsilon}")
    lines.append(r"  \resizebox{\textwidth}{!}{%")
    lines.append(r"  \begin{tabular}{" + col_spec + r"}")
    lines.append(r"    \toprule")

    # Level-1 header: MST span | AIM span
    lvl1 = [""]
    if n_mst:
        lvl1.append(f"\\multicolumn{{{n_mst}}}{{c}}{{MST (DP)}}")
    if n_aim:
        lvl1.append(f"\\multicolumn{{{n_aim}}}{{c}}{{AIM (DP)}}")
    lines.append("    " + " & ".join(lvl1) + r" \\")

    # cmidrules under the group headers
    cmidrules = []
    col_idx = 2   # 1-indexed; column 1 is the feature label
    if n_mst:
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n_mst - 1}}}")
        col_idx += n_mst
    if n_aim:
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n_aim - 1}}}")
    if cmidrules:
        lines.append("    " + " ".join(cmidrules))

    # Level-2 header: feature label + epsilon values
    hdr_cells = [r"Feature ($k$)"] + col_headers
    lines.append("    " + " & ".join(hdr_cells) + r" \\")
    lines.append(r"    \midrule")

    # Data rows: one per feature
    for feat in features:
        k = cardinalities.get(feat)
        feat_label   = feat.replace("_", r"\_").replace("-", r"\mbox{-}")
        feat_display = f"{feat_label} ($k{{=}}{k}$)" if k is not None else feat_label

        cells = [feat_display]
        for sdg in cols:
            val = (
                pivot.at[feat, sdg]
                if (feat in pivot.index and sdg in pivot.columns)
                else float("nan")
            )
            n = _n_obs(df_raw, feat, sdg)
            if np.isnan(val):
                cells.append("---")
            elif n < low_threshold:
                cells.append(f"{val:.{decimals}f}$^*$")
            else:
                cells.append(f"{val:.{decimals}f}")
        lines.append("    " + " & ".join(cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  }% end resizebox")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WandB/CSV → LaTeX table: RA per feature vs. epsilon (MST & AIM)."
    )
    parser.add_argument("--group", nargs="+", default=[WANDB_GROUP],
                        help=f"WandB run group(s) (default: {WANDB_GROUP!r}).")
    parser.add_argument("--qi", default="QI1",
                        help="QI variant (default: QI1).")
    parser.add_argument("--dataset", default=DATASET,
                        help=f"Dataset name (default: {DATASET!r}).")
    parser.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS,
                        metavar="ATTACK",
                        help=f"Attack(s) to average over (default: {DEFAULT_ATTACKS}). "
                             "Pass multiple to average, e.g. --attacks RandomForest LightGBM MLP")
    parser.add_argument("--out", default=None,
                        help="Output .tex path (default: experiment_scripts/ra_vs_epsilon_<dataset>_<qi>.tex).")
    parser.add_argument("--decimals", type=int, default=3,
                        help="Decimal places (default: 3).")
    parser.add_argument("--from-csv", nargs="+", default=None, metavar="CSV",
                        help="Load from local sweep CSV(s) instead of querying WandB. "
                             "Expected columns: sample, sdg, attack, qi, ra_mean, RA_<feat>...")
    parser.add_argument("--size", type=int, default=None,
                        help="Filter CSV rows where 'size' == this value (e.g. 10000).")
    args = parser.parse_args()

    qi_filter = args.qi

    if args.from_csv:
        dataset_filter = args.dataset if args.dataset.lower() != "all" else None
        df = load_csv_long(
            args.from_csv,
            qi_filter=qi_filter,
            attack_filter=args.attacks,
            size_filter=args.size,
            dataset_filter=dataset_filter,
        )
    else:
        dataset_filter = args.dataset if args.dataset.lower() != "all" else ""
        df = fetch_runs_long(
            args.group,
            qi_filter=qi_filter,
            attack_filter=args.attacks,
            dataset_filter=dataset_filter,
        )

    if df.empty:
        print("No data found.")
        return

    print(f"\n  {len(df)} (run, feature) records | "
          f"{df['attack'].nunique()} attacks | "
          f"{df['sdg'].nunique()} SDGs | "
          f"{df['feature'].nunique()} features | "
          f"{df['sample'].nunique()} samples")
    print(f"  Attacks : {sorted(df['attack'].unique())}")
    print(f"  SDGs    : {sorted(df['sdg'].unique())}")
    print(f"  Features: {sorted(df['feature'].unique())}")

    cardinalities = get_cardinalities(args.dataset, size=args.size)

    pivot = build_pivot(df)
    latex = to_latex(
        pivot, df,
        group=", ".join(args.group),
        qi=args.qi,
        dataset=args.dataset,
        cardinalities=cardinalities,
        attacks=args.attacks,
        decimals=args.decimals,
        size=args.size,
    )

    out_path = Path(args.out) if args.out else (
        Path(__file__).parent / f"ra_vs_epsilon_{args.dataset}_{args.qi}.tex"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex + "\n")
    print(f"\nLaTeX table written to: {out_path}")

    print("\n" + "─" * 70)
    print(latex)
    print("─" * 70)

    print("\nPivot (plain text preview):")
    cols     = _ordered_cols(pivot)
    features = _ordered_features(pivot, args.dataset, args.qi)
    avail_f  = [f for f in features if f in pivot.index]
    avail_c  = [c for c in cols if c in pivot.columns]
    print(pivot.loc[avail_f, avail_c].to_string(
        float_format=lambda x: f"{x:.{args.decimals}f}"
    ))


if __name__ == "__main__":
    main()
