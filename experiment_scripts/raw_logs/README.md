# Raw console logs kept as primary sources

Most experiment drivers write a CSV. The MIA comparison drivers
(`compare_mia_ra.py`, `run_mia_comparison_cdc.py`,
`run_mia_comparison_arizona.py`) instead *print* their results, so for those the
console log **is** the primary record. The logs are kept here so the extracted
CSV can be audited against them.

| Log | Run | Covers |
|---|---|---|
| `mia_comparison_adult_20260519.txt` | 2026-05-19, pre-repair | Adult: TabDDPM, Synthpop, MST ε=1000 |
| `mia_comparison_cdc_20260526.log` | 2026-05-26, pre-repair | CDC Diabetes: all six SDG methods |
| `mia_comparison_arizona_20260526.log` | 2026-05-26, pre-repair | NIST Arizona: all six |
| `mia_comparison_arizona_qimed_20260527.log` | 2026-05-27, pre-repair | NIST Arizona under $\text{QI}_\text{medium}$ |
| `mia_rerun_20260821.log` | 2026-08-21, **post-repair** | All three datasets × 14 SDG configurations |

"Pre-repair" and "post-repair" refer to the float-binned encoding fix of August
2026, which is why the August run supersedes the others.

`historical/parse_mia_log_to_csv.py` turns all of these into
`../mia_comparison_results.csv`, tagging every row with the run it came from:

```bash
python experiment_scripts/historical/parse_mia_log_to_csv.py \
    -o experiment_scripts/mia_comparison_results.csv \
    experiment_scripts/raw_logs/*.log experiment_scripts/raw_logs/*.txt
```
