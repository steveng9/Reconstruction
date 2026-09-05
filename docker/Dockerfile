# =============================================================================
# Light image: reconstruction attacks, scoring, and paper-table regeneration.
#
# This is the image artifact reviewers should use. It covers every stage of the
# pipeline that runs on CPU in reasonable time:
#   * generating synthetic data with the CPU DP generators (MST, AIM, PrivBayes,
#     PrivSyn, MWEM-PGM, PrivateGSD)
#   * running every attack in the registry, including the diffusion-based ones
#     (CondDDPM, CondRePaint, RePaint), on that synthetic data
#   * regenerating the paper's tables and figures from the shipped results.db
#
# Two things are deliberately NOT here:
#   * LinearReconstruction, which needs a licensed Gurobi (gurobipy);
#   * the GPU generators (TVAE, CTGAN, ARF, TabDDPM) and the R-based ones
#     (Synthpop, RankSwap, CellSuppression) -- see docker/Dockerfile.full.
# Both degrade gracefully: attacks whose dependencies are absent are disabled
# with a clear message rather than breaking the attack registry.
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
FROM python:3.9.23-slim-bookworm@sha256:7bffea15bcc3d7fb87cf10a027986203e4281e078fa2f5b234c30fca291f0834 AS base

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
RUN pip3 install --no-cache-dir --upgrade pip==24.0 \
 && pip3 install --no-cache-dir -r /tmp/requirements-attacks.txt

# smartnoise-synth 1.0.5 declares torch>=2.2.0, but the attack stack is pinned to
# torch 1.13.1 and the MST/AIM code paths never touch torch. --no-deps keeps the
# pinned torch and reproduces the environment the paper's results came from; the
# real runtime dependencies (opendp, pac-synth, smartnoise-sql) are pinned in
# requirements-attacks.txt above.
RUN pip3 install --no-cache-dir --no-deps smartnoise-synth==1.0.5

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
