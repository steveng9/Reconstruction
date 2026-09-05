#!/usr/bin/env python
"""Run every attack and every generator once, and report what works here.

Both registries import lazily and degrade gracefully, so a method can be listed
and still be unusable in the current environment -- a missing optional
dependency, no GPU, no R, no Gurobi licence. Importing is not evidence that a
method runs, which is what this script checks: each attack is run against the
committed dummy dataset, and each generator is asked for a small synthetic
frame.

    python experiment_scripts/verify_registries.py            # both
    python experiment_scripts/verify_registries.py attacks
    python experiment_scripts/verify_registries.py sdg

Expect several failures in the light Docker image -- it deliberately omits the
GPU and R generators and Gurobi. The point is that each one fails with a clear
reason, and that everything the image does claim actually runs.
"""

import sys as _sys, pathlib as _pathlib
for _anc in _pathlib.Path(__file__).resolve().parents:
    if (_anc / "paths.py").exists():
        _sys.path.insert(0, str(_anc))
        break
from paths import DATA_ROOT, REPO_ROOT, add_external_to_sys_path

# The diffusion and LP attacks import modules from the external submodules by
# bare name. master_experiment_script.py appends those directories to sys.path
# on startup; do the same here so this script sees the same registry a real run
# would.
add_external_to_sys_path()

import argparse
import json
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

DUMMY = Path(str(DATA_ROOT)) / "dummy"
if not DUMMY.exists():                      # data/ may point elsewhere
    DUMMY = Path(str(REPO_ROOT)) / "data" / "dummy"

SAMPLE = DUMMY / "size_1000" / "sample_00"


def _load_dummy():
    import pandas as pd
    train = pd.read_csv(SAMPLE / "train.csv")
    synth = pd.read_csv(SAMPLE / "MST_eps10" / "synth.csv")
    meta = json.loads((DUMMY / "meta.json").read_text())
    return train, synth, meta


def _row(name, ok, detail, secs):
    mark = "ok  " if ok else "FAIL"
    print(f"  {mark}  {name:<28} {secs:6.1f}s  {detail}")
    return ok


def check_attacks():
    from attacks import ATTACK_REGISTRY
    train, synth, meta = _load_dummy()

    qi = ["region", "sex", "age_band"]
    hidden = [c for c in train.columns if c not in qi]
    targets = train.head(20)

    # The dummy dataset is categorical, so only the categorical and agnostic
    # registries can be exercised against it. Continuous-only attacks are listed
    # as skipped rather than counted as failures -- there is nothing here for
    # them to run on.
    cat = dict(ATTACK_REGISTRY.get("categorical", {}))
    agn = dict(ATTACK_REGISTRY.get("agnostic", {}))
    runnable = {**cat, **agn}
    cont_only = sorted(set(ATTACK_REGISTRY.get("continuous", {})) - set(runnable))

    # Attacks read their hyperparameters out of cfg["attack_params"], which the
    # real pipeline fills from attack_defaults. Use the same source here, or
    # every parameterised attack fails on a missing key rather than on anything
    # to do with the environment.
    from attack_defaults import ATTACK_PARAM_DEFAULTS

    # This script asks "does it run", not "how well does it do". The diffusion
    # and neural attacks would otherwise train from scratch for many minutes
    # each on CPU, so cut the training budget to the smallest value that still
    # exercises every code path.
    QUICK = {"num_epochs": 2, "epochs": 2, "num_timesteps": 10, "resamples": 1,
             "n_estimators": 10, "patience": 1}

    def _cfg_for(name):
        params = dict(ATTACK_PARAM_DEFAULTS.get(name, {}))
        for key, value in QUICK.items():
            if key in params:
                params[key] = value
        return {
            "attack_params": params,
            "dataset": {"name": "dummy", "dir": scratch, "size": 1000,
                        "type": "categorical"},
            "QI": "QI1",
            "sdg_method": "MST",
            "attack_method": name,
        }

    # Several attacks cache trained models next to the data they were given.
    # Run against a throwaway copy so the committed dummy dataset stays clean.
    tmp = tempfile.mkdtemp(prefix="verify_registries_")
    scratch = str(Path(tmp) / "sample_00")
    shutil.copytree(str(SAMPLE), scratch)

    print(f"\n=== attacks ({len(runnable)} runnable on categorical data) ===")
    passed = failed = 0
    for name in sorted(runnable):
        t0 = time.time()
        try:
            out = runnable[name](_cfg_for(name), synth, targets, qi, hidden)
            recon = out[0] if isinstance(out, tuple) else out
            ok = recon is not None and len(recon) == len(targets)
            detail = f"reconstructed {len(recon)}x{len(hidden)}" if ok else "empty result"
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {str(e)[:90]}"
        if _row(name, ok, detail, time.time() - t0):
            passed += 1
        else:
            failed += 1

    shutil.rmtree(tmp, ignore_errors=True)

    if cont_only:
        print(f"  -- {len(cont_only)} continuous-only attacks not exercised by the "
              f"categorical dummy dataset: {', '.join(cont_only)}")
    return passed, failed


# Training budgets cut to the smallest value that still runs every step. These
# generators are being checked for "does it run here", not for output quality.
GENERATOR_QUICK = {
    "TabDDPM": {"iterations": 20, "num_timesteps": 10},
    "TVAE": {"epochs": 2},
    "CTGAN": {"epochs": 2},
}


def check_sdg():
    from sdg import SDG_REGISTRY, get_sdg
    train, _, meta = _load_dummy()

    # Private-GSD cannot consume a schema with ordinal columns: the transformer
    # it inherits from SmartNoise asks for bounds it is given no budget to
    # estimate, and fails before generation starts. The paper's datasets declare
    # none, so this never arises there -- but the dummy dataset does, so fold
    # them into categorical for this one generator rather than report a failure
    # that says nothing about the environment.
    folded = {"categorical": meta["categorical"] + meta["ordinal"],
              "continuous": meta["continuous"], "ordinal": []}

    print(f"\n=== generators ({len(SDG_REGISTRY)}) ===")
    passed = failed = 0
    for name in sorted(SDG_REGISTRY):
        t0 = time.time()
        cfg = {"epsilon": 10.0} if name in (
            "MST", "AIM", "PrivBayes", "MWEMPGM", "PrivSyn", "PrivateGSD") else {}
        # As with the attacks: exercise the code path, do not train to quality.
        # TabDDPM defaults to 10,000 iterations, which is many minutes on CPU.
        cfg.update(GENERATOR_QUICK.get(name, {}))
        schema = folded if name == "PrivateGSD" else meta
        try:
            out = get_sdg(name)(train, schema, **cfg)
            ok = out is not None and len(out) > 0
            detail = f"generated {out.shape[0]}x{out.shape[1]}" if ok else "empty frame"
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {str(e)[:90]}"
        if _row(name, ok, detail, time.time() - t0):
            passed += 1
        else:
            failed += 1
    return passed, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default="both",
                    choices=["both", "attacks", "sdg"])
    args = ap.parse_args(argv)

    if not SAMPLE.exists():
        sys.exit(f"dummy dataset not found at {SAMPLE}")

    passed = failed = 0
    if args.which in ("both", "attacks"):
        p, f = check_attacks(); passed += p; failed += f
    if args.which in ("both", "sdg"):
        p, f = check_sdg(); passed += p; failed += f

    print(f"\n{passed} ran, {failed} did not")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
