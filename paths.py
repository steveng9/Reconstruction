"""Centralized filesystem paths for the Reconstruction artifact.

Nothing in this repository should hard-code an absolute path. Every root below is
derived from the location of this file and can be overridden with an environment
variable, so the same checkout runs unchanged on a reviewer's laptop, inside the
Docker image, or on a cluster.

Environment variables
---------------------
RECON_DATA_ROOT      Root of the dataset tree (default: <repo>/data).
                     Expected layout:
                         $RECON_DATA_ROOT/{dataset}/meta.json
                         $RECON_DATA_ROOT/{dataset}/size_{N}/sample_{XX}/train.csv
                         $RECON_DATA_ROOT/{dataset}/size_{N}/sample_{XX}/{SDG}/synth.csv
RECON_EXTERNAL_ROOT  Root holding the two vendored sibling repos, normally the
                     git submodules at <repo>/external.
RECON_NIST_CRC_ROOT  NIST CRC competition data (default: $RECON_DATA_ROOT/NIST_CRC).
RECON_RESULTS_DB     SQLite results database (default:
                     <repo>/experiment_scripts/results.db).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _env_path(var: str, default: Path) -> Path:
    """Return $var as a Path if set, else `default`."""
    value = os.environ.get(var)
    return Path(value).expanduser() if value else default


DATA_ROOT = _env_path("RECON_DATA_ROOT", REPO_ROOT / "data")
EXTERNAL_ROOT = _env_path("RECON_EXTERNAL_ROOT", REPO_ROOT / "external")
NIST_CRC_ROOT = _env_path("RECON_NIST_CRC_ROOT", DATA_ROOT / "NIST_CRC")
RESULTS_DB = _env_path("RECON_RESULTS_DB", REPO_ROOT / "experiment_scripts" / "results.db")

# The two external attack repositories, shipped as git submodules under external/.
MIA_ON_DIFFUSION = EXTERNAL_ROOT / "MIA_on_diffusion"
RECON_SYNTH = EXTERNAL_ROOT / "recon-synth"

# recon-synth is imported by direct module path rather than as a package, to avoid
# colliding with this repo's own `attacks/` package (see CLAUDE.md).
EXTERNAL_SYS_PATH = [
    MIA_ON_DIFFUSION,
    MIA_ON_DIFFUSION / "midst_models" / "single_table_TabDDPM",
    RECON_SYNTH,
    RECON_SYNTH / "attacks",
    RECON_SYNTH / "attacks" / "solvers",
]


def add_external_to_sys_path() -> None:
    """Put the external attack repos on sys.path (idempotent).

    Required by the CondDDPM / CondRePaint / RePaint attacks (MIA_on_diffusion)
    and by LinearReconstruction (recon-synth).
    """
    for entry in EXTERNAL_SYS_PATH:
        text = str(entry)
        if text not in sys.path:
            sys.path.append(text)


def add_repo_to_sys_path() -> None:
    """Put the repository root on sys.path (idempotent).

    Used by scripts under experiment_scripts/ that import top-level modules such
    as `get_data` or `scoring`, and by worker processes spawned with `spawn`.
    """
    text = str(REPO_ROOT)
    if text not in sys.path:
        sys.path.insert(0, text)


def dataset_dir(dataset: str, size: int | str, sample: int | str) -> Path:
    """Path to one training sample, e.g. data/adult/size_10000/sample_00."""
    sample_name = sample if isinstance(sample, str) else f"sample_{sample:02d}"
    return DATA_ROOT / dataset / f"size_{size}" / sample_name


def describe() -> str:
    """Human-readable summary of the resolved roots (used by test.sh)."""
    rows = [
        ("REPO_ROOT", REPO_ROOT),
        ("DATA_ROOT", DATA_ROOT),
        ("EXTERNAL_ROOT", EXTERNAL_ROOT),
        ("NIST_CRC_ROOT", NIST_CRC_ROOT),
        ("RESULTS_DB", RESULTS_DB),
    ]
    width = max(len(name) for name, _ in rows)
    return "\n".join(
        f"{name:<{width}}  {path}  {'' if path.exists() else '(missing)'}".rstrip()
        for name, path in rows
    )


if __name__ == "__main__":
    print(describe())
