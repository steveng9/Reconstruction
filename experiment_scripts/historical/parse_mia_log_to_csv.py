"""Extract MIA-vs-RA-as-MIA comparison results from sweep logs into one CSV.

`compare_mia_ra.py`, `run_mia_comparison_*.py` and `run_mia_rebuttal_sweep.py`
print their results rather than writing them, so the numbers in the paper's MIA
comparison table were read off console output. This recovers them into a
machine-readable file so the table can be checked and regenerated like the rest.

Two log formats exist and both are handled:

  "summary" (May 2026 runs)  a "Method / AUC / Advantage / TPR@FPR1%" block
  "compact" (Aug 2026 re-run) a single "SynthDistance AUC=... NNDR AUC=..." line

Run once; the output (`experiment_scripts/mia_comparison_results.csv`) is
committed. Kept here for provenance.

    python experiment_scripts/historical/parse_mia_log_to_csv.py \
        -o experiment_scripts/mia_comparison_results.csv \
        experiment_scripts/raw_logs/*.log experiment_scripts/raw_logs/*.txt
"""
import sys as _sys, pathlib as _pathlib
for _anc in _pathlib.Path(__file__).resolve().parents:
    if (_anc / "paths.py").exists():
        _sys.path.insert(0, str(_anc))
        break

import argparse
import csv
import re
from pathlib import Path

BLOCK = re.compile(r"MIA vs RA-as-MIA\s+[-—–]\s+(\w+)\s*/\s*(\S+)")
# compact form: "  adult / MST_eps1" followed later by the one-line AUC summary
HEADER = re.compile(r"^\s{2}(\w+)\s*/\s*(\S+)\s*$")
QI = re.compile(r"\[RA-as-MIA\s*/\s*QI=(\S+?)\]")
COMPACT = re.compile(
    r"SynthDistance\s+AUC=([\d.]+)\s+NNDR\s+AUC=([\d.]+)\s+RA-as-MIA\s+AUC=([\d.]+)"
)
# summary form: "  SynthDistance   0.769   0.510   0.485"
SUMMARY = re.compile(
    r"^\s{2}(SynthDistance|NNDR|RA-as-MIA \(RandomForest, QI1\))\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)
KEY = {"SynthDistance": "SynthDistance", "NNDR": "NNDR",
       "RA-as-MIA (RandomForest, QI1)": "RA_as_MIA"}


def parse(path, run_label):
    """Return one dict per (dataset, sdg_method) block found in `path`."""
    text = Path(path).read_text(errors="replace").splitlines()
    out, cur, seen_summary = [], None, False

    def flush():
        nonlocal cur
        if cur and "SynthDistance_auc" in cur:
            out.append(cur)
        cur = None

    for line in text:
        b = BLOCK.search(line)
        if b:
            # A "MIA vs RA-as-MIA - ds / sdg" banner appears twice per block in
            # the summary logs; only start a new record on the first.
            if cur is None or (cur["dataset"], cur["sdg_method"]) != (b.group(1), b.group(2)):
                flush()
                cur = {"run": run_label, "source_log": Path(path).name,
                       "dataset": b.group(1), "sdg_method": b.group(2), "qi": ""}
            continue

        h = HEADER.match(line)
        if h and "=" not in line and "wandb" not in line and not BLOCK.search(line):
            flush()
            cur = {"run": run_label, "source_log": Path(path).name,
                   "dataset": h.group(1), "sdg_method": h.group(2), "qi": ""}
            continue

        if cur is None:
            continue

        q = QI.search(line)
        if q:
            cur["qi"] = q.group(1)
            continue

        c = COMPACT.search(line)
        if c:
            cur["SynthDistance_auc"] = float(c.group(1))
            cur["NNDR_auc"] = float(c.group(2))
            cur["RA_as_MIA_auc"] = float(c.group(3))
            flush()
            continue

        s = SUMMARY.match(line)
        if s:
            k = KEY[s.group(1)]
            cur[f"{k}_auc"] = float(s.group(2))
            cur[f"{k}_advantage"] = float(s.group(3))
            cur[f"{k}_tpr_at_fpr001"] = float(s.group(4))
            if k == "RA_as_MIA":
                cur.setdefault("qi", "QI1")
                flush()
            continue

    flush()
    return out


FIELDS = ["run", "source_log", "dataset", "sdg_method", "qi",
          "SynthDistance_auc", "NNDR_auc", "RA_as_MIA_auc",
          "SynthDistance_advantage", "NNDR_advantage", "RA_as_MIA_advantage",
          "SynthDistance_tpr_at_fpr001", "NNDR_tpr_at_fpr001",
          "RA_as_MIA_tpr_at_fpr001"]

# Which repair generation each source log belongs to. The August re-run is the
# only one made against the post-encoding-fix synthetic data.
RUN_OF = {
    "mia_rerun_20260821.log": "2026-08-21_postrepair",
    "mia_comparison_adult_20260519.txt": "2026-05-19_prerepair",
    "mia_comparison_cdc_20260526.log": "2026-05-26_prerepair",
    "mia_comparison_arizona_20260526.log": "2026-05-26_prerepair",
    "mia_comparison_arizona_qimed_20260527.log": "2026-05-27_prerepair",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    rows = []
    for p in args.log:
        label = RUN_OF.get(Path(p).name, "unknown")
        got = parse(p, label)
        print(f"{Path(p).name}: {len(got)} blocks  [{label}]")
        rows.extend(got)

    rows.sort(key=lambda r: (r["run"], r["dataset"], r["sdg_method"]))
    out = Path(args.out)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
