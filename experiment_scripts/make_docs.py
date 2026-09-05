#!/usr/bin/env python
"""
Regenerate the manifest-derived sections of the markdown documentation.

Two documents describe which paper objects rebuild and which do not. Both used to
be maintained by hand, and both drifted -- the coverage counts in
TODO-TABLE-COVERAGE.md disagreed with reality twice. Those sections are now
generated from `paper_objects.py` instead, spliced between marker comments:

    <!-- BEGIN generated: <block> -->   ... generated content ...   <!-- END generated -->

Prose outside the markers is hand-written and is never touched.

    python reproduce.py --docs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_objects as M

ROOT = Path(__file__).resolve().parent.parent
TODO = ROOT / "TODO-TABLE-COVERAGE.md"
SCRIPTS_README = ROOT / "experiment_scripts" / "README.md"

TIER_HEADINGS = {
    "A": ("Tier A -- data already committed, generator missing",
          "Each of these needs only a new function in `regen_camera_tables.py` and a\n"
          "`status=\"done\"` flip in the manifest. No experiments to re-run, no files to\n"
          "add. This is the bulk of the remaining work and should be done first."),
    "B": ("Tier B -- data committed, formatter is WandB-only",
          "The data is in `results.db`, but the only existing formatter needs our\n"
          "Weights & Biases credentials, so a reviewer cannot run it. Port the\n"
          "formatting logic to read the DB, the way `wandb_to_latex.py` already does."),
    "C": ("Tier C -- needs a result file that is not committed yet", ""),
    "D": ("Tier D -- not derived from this repository",
          "These are correctly out of scope. They are listed so the count reconciles\n"
          "and nobody goes looking for a generator."),
}


def _cell(text: str) -> str:
    """Make a value safe to drop into a markdown table cell."""
    return " ".join(str(text).split()).replace("|", r"\|")


def block_coverage() -> str:
    """The whole status section of TODO-TABLE-COVERAGE.md."""
    c = M.counts()
    out = [
        f"**Status:** {c['done']} of the manuscript's {c['labelled']} labelled tables "
        f"and figures regenerate\ntoday. This file tracks the other "
        f"{c['todo'] + c['external']}: {c['todo']} that still need work, and "
        f"{c['external']}\nthat are not derived from this repository at all.",
        "",
        "Check the live figure at any time with `python reproduce.py --list`.",
        "",
        "---",
        "",
        f"## Done -- regenerates from committed data ({c['done']} labelled "
        f"+ {c['unlabelled']} unlabelled)",
        "",
        "| Paper object | Output | Source | Verified by `test.sh` |",
        "|---|---|---|---|",
    ]
    for o in M.by_status("done"):
        label = f"`{o.label}`" if o.label else "_(no label)_"
        out.append(f"| {label} | `{o.output}` | `{_cell(o.source)}` | "
                   f"{'yes' if o.check else 'no -- binary'} |")

    for tier in ("A", "B", "C", "D"):
        rows = M.by_tier(tier)
        if not rows:
            continue
        heading, blurb = TIER_HEADINGS[tier]
        out += ["", f"## {heading} ({len(rows)})", ""]
        if blurb:
            out += [blurb, ""]
        if tier == "D":
            out += ["| Paper object | Why |", "|---|---|"]
            for o in rows:
                out.append(f"| `{o.label}` | {_cell(o.note or o.source)} |")
        else:
            out += ["| Paper object | Data is in | Notes |", "|---|---|---|"]
            for o in rows:
                out.append(f"| `{o.label}` | `{_cell(o.source)}` | {_cell(o.note)} |")
    return "\n".join(out)


def block_reproduction() -> str:
    """The coverage table inside experiment_scripts/README.md."""
    c = M.counts()
    out = [
        f"`python reproduce.py` rebuilds the following into `expected_output/tables/`. "
        f"That is\n{c['done']} of the paper's {c['labelled']} labelled objects; "
        f"`python reproduce.py --list` shows the rest and\n"
        f"[`../TODO-TABLE-COVERAGE.md`](../TODO-TABLE-COVERAGE.md) tracks the work to "
        f"close the gap.",
        "",
        "| Paper object | Output | Source |",
        "|---|---|---|",
    ]
    for o in M.by_status("done"):
        label = f"`{o.label}`" if o.label else "_(no label)_"
        out.append(f"| {label} | `{o.output}` | `{_cell(o.source)}` |")
    return "\n".join(out)


BLOCKS = {
    TODO: {"coverage": block_coverage},
    SCRIPTS_README: {"reproduction": block_reproduction},
}


def splice(path: Path, name: str, body: str) -> bool:
    """Replace the content between this block's markers. Returns True if changed."""
    begin, end = f"<!-- BEGIN generated: {name} -->", "<!-- END generated -->"
    text = path.read_text()
    if begin not in text:
        raise SystemExit(
            f"error: {path.name} has no '{begin}' marker.\n"
            "Add the marker pair where the generated section belongs:\n"
            f"  {begin}\n  {end}")
    head, rest = text.split(begin, 1)
    if end not in rest:
        raise SystemExit(f"error: {path.name} has '{begin}' but no closing '{end}'.")
    _stale, tail = rest.split(end, 1)
    new = f"{head}{begin}\n<!-- Generated by 'python reproduce.py --docs'. Do not edit by hand. -->\n\n{body}\n\n{end}{tail}"
    if new == text:
        return False
    path.write_text(new)
    return True


def main() -> int:
    problems = M._self_check()
    if problems:
        for e in problems:
            print(f"manifest error: {e}", file=sys.stderr)
        return 2
    changed = False
    for path, blocks in BLOCKS.items():
        for name, fn in blocks.items():
            if splice(path, name, fn()):
                print(f"updated  {path.relative_to(ROOT)}  [{name}]")
                changed = True
            else:
                print(f"current  {path.relative_to(ROOT)}  [{name}]")
    if not changed:
        print("\nDocumentation already matches the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
