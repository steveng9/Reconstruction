#!/usr/bin/env python
"""
The manifest: every table and figure in the paper, and how it is produced.

This file is DATA, not logic. It is the single place that answers, for each
labelled object in the manuscript:

    which file regenerates it, from what source, with which generator,
    and does the smoke test verify it byte-for-byte?

Everything else derives from it, so nothing can drift out of sync:

    reproduce.py            dispatches the generators listed here
    test.sh                 byte-diffs exactly the outputs marked check=True
    TODO-TABLE-COVERAGE.md  its tier tables are regenerated from here
    experiment_scripts/README.md   its coverage table likewise

To add a paper object: add a row, write the generator in regen_camera_tables.py,
flip `status` to "done", and run `python reproduce.py --docs`. Nothing else
needs editing by hand.

    python reproduce.py --list      # human-readable coverage report
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PaperObject:
    """One table or figure in the manuscript."""

    key: str                    # short stable id, used by --only
    title: str                  # what it is, in words
    label: str | None = None    # \label{...} in the manuscript; None = no label
    output: str | None = None   # filename written into expected_output/tables/
    source: str = ""            # where the numbers come from
    generator: str | None = None  # function name in regen_camera_tables.py
    status: str = "todo"        # done | todo | external
    tier: str = ""              # A/B/C/D for todo+external, per TODO-TABLE-COVERAGE.md
    check: bool = False         # test.sh byte-diffs this output
    note: str = ""              # why it is not done, or what to know about it

    @property
    def is_labelled(self) -> bool:
        return self.label is not None


# ─────────────────────────────────────────────────────────────────────────────
# Sources, spelled once so the rows stay short and a rename is a single edit.
# ─────────────────────────────────────────────────────────────────────────────
DB      = "results.db"
QUALITY = "quality_results_merged.csv"
MEMO    = "linear_sweep_adult_1000_20260822_182357.csv + ds_risk_scores.csv"
MIA     = "mia_comparison_results.csv"
DISP    = "per_attack_disparity_postrepair.csv"
DISP_PF = "per_attack_disparity_postrepair_perfeat.csv"

BINARY_NOTE = ("PDF/PNG are not byte-stable across matplotlib and freetype "
               "versions, so the smoke test does not diff them. The numbers "
               "behind them are checked via STATS_eps_curve.md.")


MANIFEST: list[PaperObject] = [

    # ── Done: regenerate today from committed data ───────────────────────────
    PaperObject(
        key="table1", title="Attack x SDG, Adult 10k",
        label="tab:ra_mean_adult", output="table1_ra_mean_adult.tex",
        source=DB, generator="table1", status="done", check=True,
    ),
    PaperObject(
        key="table2", title="Synthetic-data quality profile, Adult",
        label="tab:quality_overview", output="table2_quality_overview.tex",
        source=QUALITY, generator="table2", status="done", check=True,
    ),
    PaperObject(
        key="table4_compact", title="Epsilon sweep, compact (mean of 3 attacks)",
        label="tab:eps_sweep", output="table4_eps_sweep_compact.tex",
        source=DB, generator="table4", status="done", check=True,
    ),
    PaperObject(
        key="table4_full", title="Epsilon sweep, full (6 generators x 3 attacks x 2 QI)",
        label="tab:eps_full", output="table4_eps_sweep_full.tex",
        source=DB, generator="table4", status="done", check=True,
    ),
    PaperObject(
        key="stats_eps", title="Epsilon-curve paired t-tests",
        label="tab:eps_stats", output="STATS_eps_curve.md",
        source=DB, generator="stats_report", status="done", check=True,
    ),
    PaperObject(
        key="table6", title="MIA baselines vs RA-as-MIA",
        label="tab:mia_comparison", output="table6_mia_comparison.tex",
        source=MIA, generator="table6", status="done", check=True,
        note="The printed table mixes pre- and post-repair runs; this output is "
             "the post-repair version. See TODO-TABLE-COVERAGE.md.",
    ),
    PaperObject(
        key="table7", title="Memorization gap and d_S disclosure risk",
        label="tab:memorization_and_ds_risk", output="table7_memorization.tex",
        source=MEMO, generator="table7", status="done", check=True,
    ),
    PaperObject(
        key="stats_memo", title="Memorization one-sample t-tests",
        label=None, output="STATS_memorization.md",
        source=MEMO, generator="table7", status="done", check=True,
        note="Supporting statistics for tab:memorization_and_ds_risk; no label of its own.",
    ),
    PaperObject(
        key="table9", title="Disparate impact by race and sex",
        label="tab:disparate_impact", output="table9_disparate_impact.tex",
        source=DISP, generator="table9", status="done", check=True,
    ),
    PaperObject(
        key="fig_eps", title="Reconstruction risk vs privacy budget",
        label="fig:eps_curves", output="fig_eps_curves.pdf",
        source=DB, generator="figure_eps", status="done", check=False,
        note="A .png companion is written alongside the .pdf. " + BINARY_NOTE,
    ),
    PaperObject(
        key="fig_eps_perattack", title="Epsilon curves broken out per attack",
        label=None, output="fig_eps_curves_perattack.pdf",
        source=DB, generator="figure_eps", status="done", check=False,
        note="Appendix figure, no label. A .png companion is written alongside. " + BINARY_NOTE,
    ),

    # ── Tier A: data committed, generator missing ────────────────────────────
    PaperObject(
        key="table_cdc_1k", title="Attack x SDG, CDC Diabetes 1k",
        label="tab:ra_mean_cdc", source=DB + " (cdc_diabetes, 1k)",
        status="todo", tier="A",
        note="Same shape as Table 1; parameterise table1() by dataset and size "
             "rather than duplicating it.",
    ),
    PaperObject(
        key="table_cdc_100k", title="Attack x SDG, CDC Diabetes 100k",
        label="tab:cdc_100k", source=DB + " (cdc_diabetes, 100k)",
        status="todo", tier="A",
        note="Same generator as tab:ra_mean_cdc, different size filter.",
    ),
    PaperObject(
        key="table_sbo", title="Attack x SDG, NIST SBO 1k",
        label="tab:ra_mean_nist_sbo", source=DB + " (nist_sbo, 1k)",
        status="todo", tier="A",
        note="Same generator as tab:ra_mean_cdc.",
    ),
    PaperObject(
        key="table_feature_eps", title="Per-feature breakdown across epsilon",
        label="tab:feature_eps_breakdown", source=DB + " (feature_scores)",
        status="todo", tier="A",
        note="wandb_to_latex_epsilon_sweep.py already builds this from the DB -- "
             "fold that logic in rather than rewriting it.",
    ),
    PaperObject(
        key="table_memo_california", title="Memorization on California Housing",
        label="tab:memorization_california", source=DB + " (california, train/nontraining)",
        status="todo", tier="A",
        note="Continuous dataset: the metric is NRMSE, not R_adv, so it needs its "
             "own formatter rather than reusing table7().",
    ),
    PaperObject(
        key="table_quality_arizona", title="Quality profile, NIST Arizona",
        label="tab:quality_arizona", source=QUALITY, status="todo", tier="A",
        note="One parameterised generator covers all four per-dataset quality "
             "tables; synth_quality_to_latex.py has the column logic.",
    ),
    PaperObject(
        key="table_quality_cdc", title="Quality profile, CDC Diabetes",
        label="tab:quality_cdc", source=QUALITY, status="todo", tier="A",
        note="See tab:quality_arizona.",
    ),
    PaperObject(
        key="table_quality_sbo", title="Quality profile, NIST SBO",
        label="tab:quality_sbo", source=QUALITY, status="todo", tier="A",
        note="See tab:quality_arizona.",
    ),
    PaperObject(
        key="table_quality_california", title="Quality profile, California Housing",
        label="tab:quality_california", source=QUALITY, status="todo", tier="A",
        note="See tab:quality_arizona.",
    ),
    PaperObject(
        key="table_disparity_perfeature", title="Disparity broken out per feature",
        label="tab:disparity_perfeature", source=DISP_PF, status="todo", tier="A",
        note="CSV is committed.",
    ),
    PaperObject(
        key="table_disparity_perattack", title="Outlier penalty per attack",
        label="tab:disparity_perattack", source=DISP, status="todo", tier="A",
        note="Ratio of outlier to non-outlier row-level R_adv; the CSV has both columns.",
    ),

    # ── Tier B: data committed, only formatter is WandB-only ─────────────────
    PaperObject(
        key="table_qi_adult", title="QI-set sensitivity, Adult",
        label="tab:qi_analysis_adult", source=DB, status="todo", tier="B",
        note="qi_analysis/wandb_to_latex_qi.py needs our Weights & Biases "
             "credentials, so a reviewer cannot run it. The DB has every QI "
             "variant; port the formatter the way wandb_to_latex.py already reads the DB.",
    ),
    PaperObject(
        key="table_qi_cdc", title="QI-set sensitivity, CDC Diabetes",
        label="tab:qi_analysis_cdcdiabetes", source=DB, status="todo", tier="B",
        note="See tab:qi_analysis_adult.",
    ),
    PaperObject(
        key="table_linear_sweep", title="LinearReconstruction sweep summary",
        label="tab:linear_sweep_summary", source=DB + " (914 LinearReconstruction runs)",
        status="todo", tier="B",
        note="linear_sweep_to_latex.py is WandB-only. Port it to read the DB.",
    ),

    # ── Tier C: needs a result file that is not committed yet ────────────────
    PaperObject(
        key="fig_heatmap_ensemble", title="Ensembling heatmap",
        label="fig:heatmap_ensemble", source="ensembling_heatmap_results_*.csv",
        status="todo", tier="C",
        note="17 timestamped candidate CSVs exist and none is marked authoritative. "
             "Decide which run produced the printed figure, commit that one, then "
             "wire plot_ensembling_heatmap.py in. Files are <=300 KB; the work is "
             "the provenance decision, not the plumbing.",
    ),

    # ── Tier D: not derived from this repository ─────────────────────────────
    PaperObject(
        key="table_nist", title="NIST CRC scoreboard",
        label="tab:nist_results", source="NIST's published CRC scoreboard",
        status="external", tier="D",
        note="Transcribed from NIST's official results, not computed here.",
    ),
    PaperObject(
        key="table_datasets", title="Dataset descriptions",
        label="tab:datasets", source="hand-written", status="external", tier="D",
        note="Hand-written prose description of the five datasets.",
    ),
    PaperObject(
        key="table_qi_def_adult", title="QI definitions, Adult",
        label="tab:qi_def_adult", source="hand-written from get_data.py QIs",
        status="external", tier="D",
        note="Hand-written QI membership matrix -- documentation, not a result. "
             "Worth cross-checking against the QIs dict in get_data.py.",
    ),
    PaperObject(
        key="table_qi_def_cdc", title="QI definitions, CDC Diabetes",
        label="tab:qi_def_cdc", source="hand-written from get_data.py QIs",
        status="external", tier="D",
        note="Hand-written QI membership matrix. See tab:qi_def_adult.",
    ),
    PaperObject(
        key="table_qi_def_arizona", title="QI definitions, NIST Arizona",
        label="tab:qi_def_arizona", source="hand-written from get_data.py QIs",
        status="external", tier="D",
        note="Hand-written QI membership matrix. See tab:qi_def_adult.",
    ),
    PaperObject(
        key="table_qi_def_california", title="QI definitions, California Housing",
        label="tab:qi_def_california", source="hand-written from get_data.py QIs",
        status="external", tier="D",
        note="Hand-written QI membership matrix. See tab:qi_def_adult.",
    ),
    PaperObject(
        key="table_qi_def_sbo", title="QI definitions, NIST SBO",
        label="tab:qi_def_sbo", source="hand-written from get_data.py QIs",
        status="external", tier="D",
        note="Hand-written QI membership matrix. See tab:qi_def_adult.",
    ),
    PaperObject(
        key="fig_threat_model", title="Threat model diagram",
        label="fig:threat_model", source="hand-drawn", status="external", tier="D",
        note="Hand-drawn diagram, not generated from data.",
    ),
    PaperObject(
        key="fig_taxonomy", title="Attack taxonomy diagram",
        label="fig:taxonomy", source="hand-drawn", status="external", tier="D",
        note="Hand-drawn diagram, not generated from data.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Views over the manifest. Everything downstream goes through these, so the
# selection rules live in one place too.
# ─────────────────────────────────────────────────────────────────────────────

def by_status(status: str) -> list[PaperObject]:
    return [o for o in MANIFEST if o.status == status]


def by_tier(tier: str) -> list[PaperObject]:
    return [o for o in MANIFEST if o.tier == tier]


def generators() -> list[str]:
    """Generator function names to run, in manifest order, each exactly once."""
    seen, out = set(), []
    for o in MANIFEST:
        if o.status == "done" and o.generator and o.generator not in seen:
            seen.add(o.generator)
            out.append(o.generator)
    return out


def checked_outputs() -> list[str]:
    """Output files the smoke test byte-diffs against expected_output/tables/."""
    return [o.output for o in MANIFEST if o.check and o.output]


def expected_outputs() -> list[str]:
    """Every file a full run should produce."""
    return [o.output for o in MANIFEST if o.status == "done" and o.output]


def find(key: str) -> PaperObject | None:
    for o in MANIFEST:
        if o.key == key:
            return o
    return None


def counts() -> dict[str, int]:
    labelled = [o for o in MANIFEST if o.is_labelled]
    return {
        "labelled": len(labelled),
        "done": len([o for o in labelled if o.status == "done"]),
        "todo": len([o for o in labelled if o.status == "todo"]),
        "external": len([o for o in labelled if o.status == "external"]),
        "unlabelled": len(MANIFEST) - len(labelled),
    }


def _self_check() -> list[str]:
    """Invariants that would otherwise be caught only by a confusing failure."""
    problems, keys, labels, outputs = [], set(), set(), set()
    for o in MANIFEST:
        if o.key in keys:
            problems.append(f"duplicate key: {o.key}")
        keys.add(o.key)
        if o.label:
            if o.label in labels:
                problems.append(f"duplicate label: {o.label}")
            labels.add(o.label)
        if o.status not in ("done", "todo", "external"):
            problems.append(f"{o.key}: unknown status {o.status!r}")
        if o.status == "done":
            if not o.generator:
                problems.append(f"{o.key}: status=done but no generator")
            if not o.output:
                problems.append(f"{o.key}: status=done but no output")
        else:
            if o.output:
                problems.append(f"{o.key}: status={o.status} must not claim an output")
            if o.check:
                problems.append(f"{o.key}: status={o.status} cannot be checked")
            if not o.tier:
                problems.append(f"{o.key}: status={o.status} needs a tier")
        if o.check and not o.output:
            problems.append(f"{o.key}: check=True but no output")
        if o.output:
            # Two objects may share a generator, but never an output file.
            if o.output in outputs:
                problems.append(f"duplicate output: {o.output}")
            outputs.add(o.output)
    return problems


if __name__ == "__main__":
    import sys
    errs = _self_check()
    for e in errs:
        print(f"manifest error: {e}", file=sys.stderr)
    c = counts()
    print(f"{len(MANIFEST)} entries: {c['labelled']} labelled paper objects "
          f"({c['done']} done, {c['todo']} to do, {c['external']} external) "
          f"+ {c['unlabelled']} unlabelled outputs")
    sys.exit(1 if errs else 0)
