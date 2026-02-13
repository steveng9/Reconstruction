"""
Test Linear Reconstruction attack with different feature sets to identify the bug.

Compares performance with:
1. ALL 24 features (current, broken)
2. Excluding high-cardinality F21, F22
3. Only low-cardinality features (<10 unique values)
"""
import sys
import os
import pandas as pd
import numpy as np

# Add paths
sys.path.insert(0, os.path.expanduser('~/recon-synth'))
recon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(recon_path, 'SOTA_attacks'))

from linear_reconstruction import linear_reconstruction_attack


def test_with_feature_set(train, synth, qi_features, secret, test_name):
    """Test attack with a specific set of QI features."""
    print(f"\n{'='*70}")
    print(f"{test_name}")
    print(f"{'='*70}")
    print(f"QI features: {len(qi_features)}")
    print(f"Features: {qi_features}")

    # Show cardinalities
    cardinalities = {f: train[f].nunique() for f in qi_features}
    print(f"\nCardinalities:")
    high_card = {f: c for f, c in cardinalities.items() if c > 50}
    if high_card:
        print(f"  High (>50): {high_card}")
    print(f"  Average: {np.mean(list(cardinalities.values())):.1f}")
    print(f"  Max: {max(cardinalities.values())}")

    # Prepare data
    targets = train.copy()

    cfg = {
        'attack_params': {
            'k': 3,
            'n_procs': 4
        }
    }

    # Run attack
    try:
        reconstructed, _, _ = linear_reconstruction_attack(
            cfg, synth, targets, qi_features, [secret]
        )

        # Compute accuracy
        accuracy = np.mean(reconstructed[secret] == train[secret])
        baseline = max(
            np.mean(train[secret] == 1),
            np.mean(train[secret] == 2)
        )
        improvement = accuracy - baseline

        print(f"\n{'='*70}")
        print(f"RESULTS:")
        print(f"  Accuracy:    {accuracy:.2%}")
        print(f"  Baseline:    {baseline:.2%}")
        print(f"  Improvement: {improvement:+.2%}")
        if improvement > 0.01:
            print(f"  ✓ BEATS BASELINE!")
        else:
            print(f"  ✗ Does not beat baseline")
        print(f"{'='*70}")

        return accuracy, baseline, improvement

    except Exception as e:
        print(f"\n✗ Attack failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def main():
    print("="*70)
    print("TESTING LINEAR RECONSTRUCTION WITH DIFFERENT FEATURE SETS")
    print("="*70)

    # Load data
    print("\nLoading data...")
    train = pd.read_csv('/home/golobs/data/NIST_CRC/dev_data/train.csv')
    synth = pd.read_csv('/home/golobs/data/NIST_CRC/dev_data/synth.csv')

    if 'ID' in train.columns:
        train = train.drop('ID', axis=1)
    if 'ID' in synth.columns:
        synth = synth.drop('ID', axis=1)

    # Sample
    train = train.sample(n=1000, random_state=42)
    synth = synth.sample(n=1000, random_state=43)

    print(f"Train shape: {train.shape}")
    print(f"Synth shape: {synth.shape}")

    secret = 'F13'
    all_features = [col for col in train.columns if col != secret]

    # Test 1: ALL features (current approach)
    results = []

    qi_all = all_features
    acc, base, imp = test_with_feature_set(
        train, synth, qi_all, secret,
        "TEST 1: ALL 24 QI Features (including F21=620 values, F22=108 values)"
    )
    results.append(("All 24 features", len(qi_all), acc, base, imp))

    # Test 2: Exclude F21, F22
    qi_no_high = [f for f in all_features if f not in ['F21', 'F22']]
    acc, base, imp = test_with_feature_set(
        train, synth, qi_no_high, secret,
        "TEST 2: Exclude F21, F22 (22 features)"
    )
    results.append(("Exclude F21, F22", len(qi_no_high), acc, base, imp))

    # Test 3: Only low-cardinality (<10)
    cardinalities = {f: train[f].nunique() for f in all_features}
    qi_low = [f for f, c in cardinalities.items() if c < 10]
    acc, base, imp = test_with_feature_set(
        train, synth, qi_low, secret,
        f"TEST 3: Only features with <10 unique values (like ACS paper)"
    )
    results.append(("Low cardinality only", len(qi_low), acc, base, imp))

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Test':<25} {'#QI':>4} {'Accuracy':>10} {'Baseline':>10} {'Δ':>8} {'Status':>8}")
    print("-"*70)
    for name, n_qi, acc, base, imp in results:
        if acc is not None:
            status = "✓ GOOD" if imp > 0.01 else "✗ BAD"
            print(f"{name:<25} {n_qi:>4} {acc:>10.2%} {base:>10.2%} {imp:>+8.2%} {status:>8}")
        else:
            print(f"{name:<25} {n_qi:>4} {'FAILED':>10}")
    print(f"{'='*70}")

    print("\nCONCLUSION:")
    print("If Test 2 or 3 beats baseline, the bug is confirmed:")
    print("High-cardinality features (F21, F22) make queries too sparse!")


if __name__ == "__main__":
    main()
