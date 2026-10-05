#!/usr/bin/env bash
# =============================================================================
# Rebuild the paper's tables from a clean export of a commit and compare them
# with that commit's expected_output/tables/.
#
# The table code, results.db and expected_output/ have to move together. A
# commit that changes one without the others still passes ./test.sh in the
# working tree it was made in, where the newer files are sitting on disk, and
# fails for everyone who clones. This script tests what a clone would get.
#
# Usage:  maintenance_scripts/check_tables_clean.sh [commit]     (default: HEAD)
#
# To run it before every push:
#   ln -s ../../maintenance_scripts/check_tables_clean.sh .git/hooks/pre-push
# =============================================================================
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
COMMIT="${1:-HEAD}"
# As a pre-push hook git passes the remote's name and URL; test HEAD then.
git -C "$REPO_ROOT" rev-parse --verify --quiet "${COMMIT}^{commit}" >/dev/null || COMMIT=HEAD
PY="${RECON_PYTHON:-python}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git -C "$REPO_ROOT" archive "$COMMIT" | tar -x -C "$TMP"

cd "$TMP"
"$PY" reproduce.py --out "$TMP/_out" >/dev/null
status=0
for t in $("$PY" reproduce.py --check-list); do
  if ! diff -q "expected_output/tables/$t" "$TMP/_out/$t" >/dev/null; then
    echo "DIFFERS  $t"; status=1
  fi
done
if [ "$status" -eq 0 ]; then
  echo "tables rebuild cleanly from $(git -C "$REPO_ROOT" rev-parse --short "$COMMIT")"
else
  echo "A clone of this commit would fail ./test.sh. Commit the missing data or references."
fi
exit $status
