"""
Smoke test for all SDG methods.

Usage:
    python -m sdg.test_sdg [method_name ...]

If method_name is provided, only tests those methods. Otherwise tests all.
Uses a tiny subset (50 rows, 8 columns) for fast end-to-end verification.
"""

import sys
import time
import signal
import pandas as pd
from sdg import get_sdg, list_sdg


TEST_DATA = "/home/golobs/data/NIST_CRC/25_PracticeProblem/25_Demo_25f_OriginalData.csv"

# Minimal test parameters
N_ROWS = 50
N_COLS = 8       # use only first N columns to keep AIM/Synthpop fast
TIMEOUT = 120    # seconds per method before we kill it


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError(f"Timed out after {TIMEOUT}s")


def load_test_data():
    df = pd.read_csv(TEST_DATA)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Subsample rows and columns for speed
    df = df.iloc[:N_ROWS, :N_COLS].copy()

    # Auto-detect: columns with <= 20 unique values are categorical, rest continuous
    cat_cols = [c for c in df.columns if df[c].nunique() <= 20]
    cont_cols = [c for c in df.columns if df[c].nunique() > 20]
    meta = {
        "categorical": cat_cols,
        "continuous": cont_cols,
        "ordinal": [],
    }
    return df, meta


def test_method(name, train_df, meta):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    generate = get_sdg(name)

    # Method-specific config — keep everything minimal
    config = {}
    if name in ("MST", "AIM"):
        config["epsilon"] = 10.0
    elif name in ("TVAE", "CTGAN"):
        config["epochs"] = 5       # minimal training
    elif name == "TabDDPM":
        config["iterations"] = 100     # minimal training
        config["num_timesteps"] = 100
        config["batch_size"] = 64
    elif name == "RankSwap":
        swap_cols = meta.get("continuous", [])
        if not swap_cols:
            swap_cols = [train_df.columns[0]]
        config["swap_features"] = swap_cols
    elif name == "CellSuppression":
        config["key_vars"] = list(train_df.columns[:3])
        config["k"] = 3

    # Set a timeout so no single method can hang the process
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIMEOUT)

    start = time.time()
    try:
        syn_df = generate(train_df, meta, **config)
        signal.alarm(0)  # cancel alarm
        elapsed = time.time() - start

        print(f"  OK - Generated {len(syn_df)} rows x {len(syn_df.columns)} cols in {elapsed:.1f}s")
        print(f"  Input shape:  {train_df.shape}")
        print(f"  Output shape: {syn_df.shape}")

        # Check columns match
        missing = set(train_df.columns) - set(syn_df.columns)
        extra = set(syn_df.columns) - set(train_df.columns)
        if missing:
            print(f"  WARNING: Missing columns: {missing}")
        if extra:
            print(f"  WARNING: Extra columns: {extra}")

        # Show sample
        print(f"  Sample (first 3 rows):")
        print(syn_df.head(3).to_string(index=False))
        return True

    except TimeoutError:
        elapsed = time.time() - start
        print(f"  TIMEOUT after {elapsed:.0f}s — skipping {name}")
        return False

    except Exception as e:
        signal.alarm(0)
        elapsed = time.time() - start
        print(f"  FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        signal.signal(signal.SIGALRM, old_handler)


def main():
    train_df, meta = load_test_data()
    print(f"Test data: {train_df.shape[0]} rows x {train_df.shape[1]} cols")
    print(f"Categorical: {meta['categorical']}")
    print(f"Continuous:  {meta['continuous']}")

    # Determine which methods to test
    if len(sys.argv) > 1:
        methods = sys.argv[1:]
    else:
        methods = list_sdg()

    results = {}
    for name in methods:
        results[name] = test_method(name, train_df, meta)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} {status}")

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} methods passed")


if __name__ == "__main__":
    main()
