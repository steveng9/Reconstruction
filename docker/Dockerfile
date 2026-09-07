# check=skip=FromPlatformFlagConstDisallowed
# ^ BuildKit lints a hardcoded `FROM --platform`, on the reasoning that it
# usually means someone forgot about cross-compilation. Here it is deliberate
# and explained at the FROM line below, so the lint is silenced rather than
# left to print a warning a reviewer would have to wonder about.
# =============================================================================
# Light image: reconstruction attacks, scoring, and paper-table regeneration.
#
# This is the image artifact reviewers should use. It covers every stage of the
# pipeline that runs on CPU in reasonable time:
#   * generating synthetic data with the CPU DP generators (MST, AIM, PrivBayes,
#     PrivSyn, MWEM-PGM, Private-GSD) and with TabDDPM, which shares the
#     diffusion attacks' torch stack and so runs here too, slowly. Each was
#     verified to produce a synthetic frame in this image; run
#     experiment_scripts/verify_registries.py to check for yourself.
#   * running every attack in the registry, including the diffusion-based ones
#     (CondDDPM, CondRePaint, RePaint), on that synthetic data
#   * regenerating the paper's tables and figures from the shipped results.db
#
# Two things are deliberately NOT here:
#   * LinearReconstruction, which needs a licensed Gurobi (gurobipy);
#   * the GPU generators TVAE, CTGAN and ARF, and the R-based ones (Synthpop,
#     RankSwap, CellSuppression) -- see docker/Dockerfile.full.
# All degrade gracefully: an attack or generator whose dependencies are absent
# raises a clear ImportError naming the missing package, rather than breaking
# the registry it belongs to.
#
# Build (from the repository root):
#     docker build -f docker/Dockerfile -t recon-artifact:latest .
#
# Run:
#     docker run --rm -it -v "${PWD}":/workspace -w /workspace recon-artifact:latest bash
#     ./test.sh
# =============================================================================

# Python 3.9 to match the environment the paper's results were produced in.
# Pinned by digest so the base image cannot drift.
# The platform is pinned, not left to the host. Two of the pins below have no
# arm64 build at all -- pac-synth 0.0.8 ships no aarch64 wheel and falls back to
# a Rust source build that this image has no toolchain for, and torch 1.13.1 has
# no arm64 wheel whatsoever. On an Apple Silicon Mac an unpinned build therefore
# fails outright, part-way through pip, with an error about a missing Rust
# compiler that says nothing about architecture. Pinning here means the build
# works from any host: Docker Desktop emulates amd64 on Apple Silicon, so it is
# slower there but it completes.
FROM --platform=linux/amd64 python:3.9.23-slim-bookworm@sha256:7bffea15bcc3d7fb87cf10a027986203e4281e078fa2f5b234c30fca291f0834 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    # Thread oversubscription is a real problem for this codebase's worker
    # pools; one thread per process is what the paper's sweeps used.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    # Reviewers have no W&B account; keep all logging local by default.
    WANDB_MODE=offline

# git is needed for the two pinned VCS dependencies in requirements-attacks.txt.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY docker/requirements-attacks.txt /tmp/requirements-attacks.txt
RUN pip3 install --no-cache-dir --upgrade pip==24.0

# torch first, from PyPI's CPU-only index. The plain `torch==1.13.1` pin in
# requirements-attacks.txt resolves to the cu117 build, which drags in ~900 MB
# of CUDA libraries (cuBLAS, cuDNN, nvrtc) that this image can never use -- it
# is CPU only by design. `1.13.1+cpu` satisfies that pin, so the requirements
# file needs no change and the full image, which shares it, is unaffected. The
# CPU kernels are the same ones the cu117 build runs on a machine without a GPU,
# so results are unchanged.
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==1.13.1+cpu

RUN pip3 install --no-cache-dir -r /tmp/requirements-attacks.txt

# smartnoise-synth 1.0.5 declares torch>=2.2.0, but the attack stack is pinned to
# torch 1.13.1 and the MST/AIM code paths never touch torch. --no-deps keeps the
# pinned torch and reproduces the environment the paper's results came from; the
# real runtime dependencies (opendp, pac-synth, smartnoise-sql) are pinned in
# requirements-attacks.txt above.
RUN pip3 install --no-cache-dir --no-deps smartnoise-synth==1.0.5

# Same treatment for Private-GSD's jax stack. chex 0.1.90 declares numpy>=1.24.1
# while this environment is pinned to numpy 1.23.5 -- the combination the paper's
# results were produced with, and one the resolver will not accept. optax and
# flax are here too because each depends on chex and would drag it back in.
RUN pip3 install --no-cache-dir --no-deps chex==0.1.90 optax==0.2.4 flax==0.8.5

# TabPFN ships without its weights and downloads a ~100 MB checkpoint into its
# own site-packages directory on first use. The image runs as a non-root user,
# who cannot write there, so the attack fails with a PermissionError the first
# time a reviewer runs it. Fetch the checkpoint at build time instead; this also
# means the image needs no network access at run time.
RUN python -c "\
import numpy as np; \
from tabpfn import TabPFNClassifier; \
c = TabPFNClassifier(device='cpu', N_ensemble_configurations=1); \
c.fit(np.random.rand(20, 3), np.random.randint(0, 2, 20)); \
c.predict(np.random.rand(5, 3))" \
 && chmod -R a+rX "$VIRTUAL_ENV/lib/python3.9/site-packages/tabpfn"

# ENV PATH is not seen by a login shell, which re-derives PATH from /etc/profile.
# `docker run ... bash` (the documented command) is not a login shell and works
# either way, but `bash -l`, `bash --login` and some `docker exec` invocations
# are, and there the venv would silently fall back to the system python -- which
# has none of the dependencies. Put the venv on the login PATH too.
RUN printf 'export PATH="%s/bin:$PATH"\n' "$VIRTUAL_ENV" > /etc/profile.d/10-venv.sh

# Match the default host UID/GID so files written to the mounted volume stay
# editable outside the container. Override with --build-arg if `id` differs.
ARG UID=1000
ARG GID=1000
RUN groupadd --gid $GID recon 2>/dev/null || true \
 && useradd --uid $UID --gid $GID --create-home --shell /bin/bash recon 2>/dev/null || true
USER $UID:$GID

WORKDIR /workspace
CMD ["bash"]
