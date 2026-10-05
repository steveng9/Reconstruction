#!/usr/bin/env python
"""
summarize_ra_subgroups.py — subgroup means behind tab:disparate_impact.

`analyze_ra_subgroups.py` scores every training record under a Random Forest
attack and writes one row per record (analysis_outputs/ra_subgroups/). Those
files hold record-level data and are not shipped. This script reduces them to
the per-generator means the table prints, in the same columns as
`per_attack_disparity_postrepair.csv`, and writes
`experiment_scripts/ra_subgroups_summary.csv`, which is shipped.

Usage:
    python experiment_scripts/historical/summarize_ra_subgroups.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "analysis_outputs" / "ra_subgroups"
GENERATORS = ["CellSuppression", "TabDDPM", "Synthpop"]
RACE = {"White": "race_White", "Asian-Pac-Islander": "race_API", "Amer-Indian-Eskimo": "race_AI/AN",
        "Black": "race_Black", "Other": "race_Other"}


def main():
    rows = []
    for gen in GENERATORS:
        d = pd.read_csv(SRC / f"adult_{gen}_QI1.csv")
        score = d["RA_row_mean"]
        out, non = score[d.is_outlier].mean(), score[~d.is_outlier].mean()
        row = {"sdg": gen, "attack": "RandomForest", "mean_ra": score.mean(),
               "outlier_mean": out, "non_outlier_mean": non, "outlier_penalty": out / non}
        row.update({col: score[d.race == name].mean() for name, col in RACE.items()})
        row.update({f"sex_{s}": score[d.sex == s].mean() for s in ("Male", "Female")})
        rows.append(row)
    out_path = ROOT / "experiment_scripts" / "ra_subgroups_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {len(rows)} rows to {out_path.name}")


if __name__ == "__main__":
    main()
