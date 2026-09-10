"""
Worked queries against `results.db`, the paper's results database.

Run it from the repository root:

    python examples/query_results_db.py

It needs only Python's standard library — no dependency on this repository, and
nothing to install. The database is plain SQLite, and this script is meant to be
read and copied from as much as run: each query below is a starting point for
reusing the 49,126 scored runs as a dataset in their own right.

Full documentation of the schema, the label conventions and the caveats is in
DATABASE.md.
"""

import sqlite3
import sys
import textwrap
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "experiment_scripts" / "results.db"


# Six attacks were renamed late in writing, and the database stores whichever
# label was in use when each run was scored. Normalise before grouping, or
# results silently split across two names -- or vanish entirely. The
# authoritative map is LABEL_TO_METHOD in experiment_scripts/rerun_queue.py.
RENAMES = {
    "MarginalRF": "CoBP-RA",
    "Attention": "ARFFormer",
    "JointMLP": "MultiHeadMLP",
    "PartialMST": "CondMST",
    "TabDDPM": "CondDDPM",
    "ConditionedRePaint": "CondRePaint",
}

NORMALISE = "CASE attack_label " + " ".join(
    f"WHEN '{old}' THEN '{new}'" for old, new in RENAMES.items()
) + " ELSE attack_label END"


def show(con, title, note, sql):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(textwrap.dedent(note).strip())
    print()
    cursor = con.execute(sql)
    headers = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
        for i, h in enumerate(headers)
    ]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


def main():
    if not DB_PATH.exists():
        sys.exit(f"results.db not found at {DB_PATH}")

    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    total, = con.execute("SELECT COUNT(*) FROM runs").fetchone()
    features, = con.execute("SELECT COUNT(*) FROM feature_scores").fetchone()
    print(f"{DB_PATH.name}: {total:,} scored runs, {features:,} per-feature scores")

    show(
        con,
        "1. Which generator leaks the most?",
        """
        Averaged over every attack run against it, on Adult at n=10,000. This is
        Main Results 1 and 2: the spread across generators is far wider than the
        spread across attacks, and the de-identification methods sit at the top.
        """,
        """
        SELECT sdg_method,
               ROUND(AVG(ra_mean), 1) AS mean_radv,
               COUNT(*)               AS n
        FROM runs
        WHERE dataset = 'adult' AND dataset_size = 10000 AND split = 'standard'
        GROUP BY sdg_method HAVING n >= 20
        ORDER BY mean_radv DESC LIMIT 10
        """,
    )

    show(
        con,
        "2. ... and how much does the choice of attack matter?",
        """
        The same slice, now averaged over generators instead of over attacks,
        restricted to the thirteen generator configurations that make up the
        columns of the paper's Table 1 so that every attack is averaged over the
        same cells. Renamed labels are normalised; ablation variants
        (underscore), ensembles (+) and oracles are excluded -- the oracles are
        upper bounds rather than attacks, and would otherwise top the ranking.

        Read this against query 1. Generators span roughly fifteen points of
        R_adv; attacks span about five. That gap is Main Result 1: what you
        generate with matters far more than what you are attacked with.

        Do not read the ordering *within* this narrow band as the paper's attack
        ranking. Averaging flat over thirteen generators is not the comparison
        the paper makes, and inside a five-point spread the order moves with the
        slice you choose. For the paper's own comparison, regenerate Table 1
        (Experiment 1 in ARTIFACT-APPENDIX.md).
        """,
        f"""
        SELECT {NORMALISE}            AS attack,
               ROUND(AVG(ra_mean), 1) AS mean_radv,
               COUNT(DISTINCT sdg_method) AS generators,
               COUNT(*)               AS n
        FROM runs
        WHERE dataset = 'adult' AND dataset_size = 10000 AND qi = 'QI1'
          AND split = 'standard'
          AND sdg_method IN (
                'RankSwap', 'CellSuppression', 'Synthpop', 'TVAE', 'CTGAN',
                'ARF', 'TabDDPM', 'MST_eps0.1', 'MST_eps1', 'MST_eps10',
                'MST_eps100', 'MST_eps1000', 'AIM_eps1')
          AND attack_label NOT LIKE '%!_%' ESCAPE '!'
          AND attack_label NOT LIKE '%+%'
          AND attack_label NOT LIKE '%Oracle%'
        GROUP BY attack HAVING generators >= 10
        ORDER BY mean_radv DESC LIMIT 12
        """,
    )

    show(
        con,
        "3. Which attributes of a person are recoverable?",
        """
        The feature_scores join. Reconstruction is very far from uniform across
        attributes, which is what makes the disparate-impact question in the
        paper worth asking -- and what makes this table the useful one if you are
        studying which attributes leak rather than which attacks win.
        """,
        """
        SELECT f.feature,
               ROUND(AVG(f.ra_score), 1) AS mean_radv,
               COUNT(*)                  AS n
        FROM feature_scores f JOIN runs r ON r.run_id = f.run_id
        WHERE r.dataset = 'adult' AND r.dataset_size = 10000
          AND r.qi = 'QI1' AND r.split = 'standard'
        GROUP BY f.feature ORDER BY mean_radv DESC
        """,
    )

    show(
        con,
        "4. Does more privacy budget mean more leakage?",
        """
        MST across nine values of epsilon, ordered numerically. Main Result 4:
        risk climbs with epsilon up to roughly 10 and then flattens, so buying
        privacy above that point costs utility without measurably reducing
        reconstruction.
        """,
        """
        SELECT sdg_method,
               ROUND(AVG(ra_mean), 1) AS mean_radv,
               COUNT(*)               AS n
        FROM runs
        WHERE dataset = 'adult' AND dataset_size = 10000 AND qi = 'QI1'
          AND split = 'standard' AND sdg_method LIKE 'MST_eps%'
        GROUP BY sdg_method
        ORDER BY CAST(REPLACE(sdg_method, 'MST_eps', '') AS REAL)
        """,
    )

    show(
        con,
        "5. Is the attack memorizing, or learning the distribution?",
        """
        The memorization test scores one attack twice: on records the generator
        actually trained on, and on held-out records from the same distribution.
        A large gap would mean the attack recovers memorized rows. Main Result 5
        is that the gap is small -- most reconstruction is distributional, and
        would succeed against people who were never in the dataset.
        """,
        f"""
        SELECT {NORMALISE} AS attack,
               ROUND(AVG(CASE WHEN split='train'       THEN ra_mean END), 1) AS on_train,
               ROUND(AVG(CASE WHEN split='nontraining' THEN ra_mean END), 1) AS on_holdout,
               ROUND(AVG(CASE WHEN split='train'       THEN ra_mean END)
                   - AVG(CASE WHEN split='nontraining' THEN ra_mean END), 1) AS gap
        FROM runs
        WHERE split IN ('train', 'nontraining') AND dataset = 'adult'
        GROUP BY attack
        HAVING on_train IS NOT NULL AND on_holdout IS NOT NULL
        ORDER BY gap DESC
        """,
    )

    print()
    print("Read DATABASE.md for the schema, the label conventions these queries")
    print("work around, and the caveats to check before drawing conclusions.")
    con.close()


if __name__ == "__main__":
    main()
