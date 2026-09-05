#!/usr/bin/env bash
# =============================================================================
# Artifact smoke test.
#
# Checks, with no downloads and no access-restricted data, that:
#   1. the environment is installed and the repository's paths resolve;
#   2. the attack pipeline runs end to end on the shipped dummy dataset, and a
#      real attack beats the mode baseline (i.e. it is genuinely reconstructing,
#      not just predicting the most common value);
#   3. the paper's tables regenerate from the shipped results database and match
#      the committed reference copies byte for byte. Which tables those are is
#      read from experiment_scripts/paper_objects.py, so this list never goes
#      stale as coverage grows.
#
# Usage:  ./test.sh
# Runtime: about two minutes on a laptop, CPU only.
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export WANDB_MODE=offline
export PYTHONWARNINGS=ignore

PASS=0; FAIL=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
hdr()  { echo; echo "=== $1 ==="; }

PY="${RECON_PYTHON:-python}"

# --- 1. environment ----------------------------------------------------------
hdr "1/4  Environment and path resolution"
"$PY" paths.py || { echo "could not run paths.py with '$PY'"; exit 1; }
"$PY" - <<'EOF' && ok "core modules import" || bad "core modules import"
import sys; sys.path.insert(0, ".")
import paths, get_data, scoring, util, attack_defaults  # noqa: F401
EOF

# The manifest lists every paper object and the generator that builds it. If a
# row names a generator that does not exist, the table would simply never be
# written -- so check the manifest before trusting section 4's file list.
"$PY" experiment_scripts/paper_objects.py >/dev/null 2>&1 \
  && ok "paper-object manifest is self-consistent" \
  || bad "paper-object manifest is self-consistent"

# --- 2. dummy data present ---------------------------------------------------
hdr "2/4  Dummy dataset"
for f in data/dummy/meta.json \
         data/dummy/size_1000/sample_00/train.csv \
         data/dummy/size_1000/sample_00/MST_eps10/synth.csv; do
  [ -f "$f" ] && ok "$f" || bad "$f missing"
done

# --- 3. attack beats the mode baseline ---------------------------------------
hdr "3/4  Attack pipeline (dummy dataset, MST epsilon=10)"
run_cfg() {  # $1 = attack name -> prints mean RA
  sed "s/^attack_method: .*/attack_method: \"$1\"/" configs/demo_dummy.yaml > /tmp/_recon_test_$$.yaml
  CONFIG_PATH_default=/tmp/_recon_test_$$.yaml "$PY" master_experiment_script.py --n_runs 1 2>/dev/null \
    | awk '/^ave: /{v=$2} END{print v}'
  rm -f /tmp/_recon_test_$$.yaml
}
BASE=$(run_cfg Mode)
ATK=$(run_cfg "CoBP-RA")
echo "  mode baseline : ${BASE:-<none>}"
echo "  CoBP-RA       : ${ATK:-<none>}"
if [ -n "${BASE:-}" ] && [ -n "${ATK:-}" ] && \
   awk -v a="$ATK" -v b="$BASE" 'BEGIN{exit !(a > b + 2.0)}'; then
  ok "CoBP-RA beats the mode baseline by >2 points"
else
  bad "CoBP-RA did not clear the mode baseline (expected roughly 40 vs 34)"
fi

# --- 4. paper tables reproduce ----------------------------------------------
hdr "4/4  Paper tables regenerate from results.db"
if [ ! -f experiment_scripts/results.db ]; then
  bad "experiment_scripts/results.db missing"
else
  TMPOUT=$(mktemp -d)
  "$PY" reproduce.py --out "$TMPOUT" >/dev/null 2>&1
  # The list of files to verify comes from the manifest, not from this script,
  # so adding a paper object to paper_objects.py automatically extends the test.
  # Figures are excluded there: PDF/PNG are not byte-stable across matplotlib
  # versions, but the numbers behind them are checked via STATS_eps_curve.md.
  CHECKED=$("$PY" reproduce.py --check-list)
  if [ -z "$CHECKED" ]; then
    bad "could not read the manifest's list of verified outputs"
  fi
  for t in $CHECKED; do
    if diff -q "expected_output/tables/$t" "$TMPOUT/$t" >/dev/null 2>&1; then
      ok "$t matches the committed reference"
    else
      bad "$t differs from the committed reference"
    fi
  done
  rm -rf "$TMPOUT"
fi

# --- summary -----------------------------------------------------------------
echo
echo "============================================"
echo "  $PASS passed, $FAIL failed"
echo "============================================"
[ "$FAIL" -eq 0 ] || exit 1
