#!/usr/bin/env python
"""
Rebuild every paper table and figure that this artifact can regenerate.

This is the single entry point for artifact evaluation. It reads the manifest in
`experiment_scripts/paper_objects.py`, runs each generator listed there, and
writes the results into `expected_output/tables/` next to the committed reference
copies, so a run can be diffed against the numbers printed in the paper.

    python reproduce.py                 # rebuild everything (about 30 seconds)
    python reproduce.py --list          # what is covered, and what is not
    python reproduce.py --only table1   # rebuild one object
    python reproduce.py --out DIR       # write somewhere else

Provenance rule: every number this emits comes from `experiment_scripts/results.db`
or from a named, committed result CSV. No number is carried over from the .tex
sources of the paper.

The generator functions themselves live in `experiment_scripts/regen_camera_tables.py`;
this script only decides what to run and reports on it.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiment_scripts"))

import paper_objects as M  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def cmd_list() -> int:
    """Print the coverage report: which paper objects rebuild, and which do not."""
    c = M.counts()
    print(f"\n{c['done']} of {c['labelled']} labelled paper objects regenerate from "
          f"committed data.")
    print(f"{c['todo']} still need work; {c['external']} are not derived from this "
          f"repository.\n")

    print("REGENERATES TODAY")
    print(f"  {'paper label':<34}{'output':<38}{'checked':<9}source")
    for o in M.by_status("done"):
        label = o.label or "(unlabelled)"
        mark = "byte-diff" if o.check else "--"
        print(f"  {label:<34}{o.output:<38}{mark:<9}{o.source}")

    todo = M.by_status("todo")
    if todo:
        print("\nNOT YET REGENERATED")
        for tier, heading in [
            ("A", "Tier A -- data committed, generator missing"),
            ("B", "Tier B -- data committed, only formatter needs Weights & Biases"),
            ("C", "Tier C -- result file not committed yet"),
        ]:
            rows = [o for o in todo if o.tier == tier]
            if not rows:
                continue
            print(f"\n  {heading}  ({len(rows)})")
            for o in rows:
                print(f"    {o.label:<34}{o.source}")

    ext = M.by_status("external")
    if ext:
        print(f"\nNOT FROM THIS REPOSITORY  ({len(ext)})")
        for o in ext:
            print(f"    {o.label:<34}{o.source}")
    print()
    return 0


def cmd_check_list() -> int:
    """Print the outputs the smoke test byte-diffs, one per line. Consumed by test.sh."""
    for name in M.checked_outputs():
        print(name)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Building
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(gen, only_keys: set[str] | None) -> list[str]:
    """Generator names to run, honouring --only."""
    if only_keys is None:
        return M.generators()
    names, unknown = [], []
    for key in only_keys:
        obj = M.find(key)
        if obj is None:
            unknown.append(key)
        elif obj.status != "done":
            print(f"error: '{key}' ({obj.label}) does not regenerate yet "
                  f"-- see 'reproduce.py --list'", file=sys.stderr)
            raise SystemExit(2)
        elif obj.generator not in names:
            names.append(obj.generator)
    if unknown:
        print(f"error: unknown object(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"known keys: {', '.join(o.key for o in M.MANIFEST)}", file=sys.stderr)
        raise SystemExit(2)
    return names


def cmd_build(only_keys: set[str] | None, out_dir: Path | None) -> int:
    if out_dir is not None:
        # regen_camera_tables resolves its output directory at import time.
        os.environ["RECON_TABLE_OUT"] = str(out_dir)

    import regen_camera_tables as gen  # noqa: E402  (import after RECON_TABLE_OUT)

    runs = gen.load_runs()
    runs = runs[runs.confidence == "certain"]   # never build a table from flagged rows

    results = {}
    for name in _resolve(gen, only_keys):
        fn = getattr(gen, name, None)
        if fn is None:
            print(f"error: manifest names generator '{name}', which does not exist "
                  f"in regen_camera_tables.py", file=sys.stderr)
            return 2
        # Generators that need the runs frame take it; the CSV-only ones take nothing.
        results[name] = fn(runs) if inspect.signature(fn).parameters else fn()

    _report(gen, results, only_keys is None)
    return 0


def _report(gen, results, full_run: bool) -> None:
    """List what was written, then surface the two data-coverage warnings."""
    print("wrote:")
    for f in sorted(gen.OUT.iterdir()):
        print(f"  {f.name:<40}{f.stat().st_size:>8} B")

    if full_run:
        produced = {f.name for f in gen.OUT.iterdir()}
        absent = [n for n in M.expected_outputs() if n not in produced]
        if absent:
            print("\nexpected but not produced:")
            for n in absent:
                print(f"  {n}")

    if "table7" in results and results["table7"]:
        print("\nTable 7 rows with NO QI_linear memorization runs "
              "(cannot be filled without new runs):")
        for m in results["table7"]:
            print(f"  {m}")

    if "table1" in results:
        _grid, cov = results["table1"]
        if cov:
            print("\ncells with n<5 disjoint samples (caption must not claim 5):")
            for a, b, n in cov:
                print(f"  {a:<24}{b:<28}n={n}")


# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="reproduce.py",
        description="Rebuild the paper's tables and figures from committed data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with no arguments to rebuild everything.",
    )
    p.add_argument("--list", action="store_true",
                   help="show which paper objects regenerate, and which do not")
    p.add_argument("--check-list", action="store_true",
                   help="print the outputs test.sh verifies, one per line")
    p.add_argument("--docs", action="store_true",
                   help="regenerate the manifest-derived sections of the markdown docs")
    p.add_argument("--only", metavar="KEY[,KEY...]",
                   help="rebuild only these objects (keys from --list)")
    p.add_argument("--out", metavar="DIR", type=Path,
                   help="write outputs here instead of expected_output/tables/")
    a = p.parse_args(argv)

    problems = M._self_check()
    if problems:
        for e in problems:
            print(f"manifest error: {e}", file=sys.stderr)
        return 2

    if a.list:
        return cmd_list()
    if a.check_list:
        return cmd_check_list()
    if a.docs:
        import make_docs
        return make_docs.main()

    only = set(a.only.split(",")) if a.only else None
    return cmd_build(only, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
