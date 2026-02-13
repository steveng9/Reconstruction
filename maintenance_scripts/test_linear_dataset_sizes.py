"""
Test Linear Reconstruction attack with different dataset sizes.

The paper tests with 1K, 10K, 100K rows. Maybe 1K is too small for the attack to work?
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.expanduser('~/recon-synth'))
recon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(recon_path, 'SOTA_attacks'))

from linear_reconstruction import linear_reconstruction_attack


def test_with_size(train_full, synth_full, n_samples, secret, qi_features):
    """Test with specific dataset size."""
    print(f"\n{'='*70}")
    print(f"TESTING WITH {n_samples} ROWS")
    print(f"{'='*70}")

    # Sample
    train = train_full.sample(n=min(n_samples, len(train_full)), random_state=42)
    synth = synth_full.sample(n=min(n_samples, len(synth_full)), random_state=43)

    print(f"Train shape: {train.shape}")
    print(f"Synth shape: {synth.shape}")

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

        print(f"\nRESULTS:")
        print(f"  Accuracy:    {accuracy:.2%}")
        print(f"  Baseline:    {baseline:.2%}")
        print(f"  Improvement: {improvement:+.2%}")
        status = "✓ BEATS BASELINE!" if improvement > 0.01 else "✗ Does not beat baseline"
        print(f"  {status}")

        return accuracy, baseline, improvement

    except Exception as e:
        print(f"\n✗ Attack failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def main():
    print("="*70)
    print("TESTING LINEAR RECONSTRUCTION WITH DIFFERENT DATASET SIZES")
    print("="*70)

    # Load FULL datasets
    print("\nLoading full datasets...")
    train_full = pd.read_csv('/home/golobs/data/NIST_CRC/dev_data/train.csv')
    synth_full = pd.read_csv('/home/golobs/data/NIST_CRC/dev_data/synth.csv')

    if 'ID' in train_full.columns:
        train_full = train_full.drop('ID', axis=1)
    if 'ID' in synth_full.columns:
        synth_full = synth_full.drop('ID', axis=1)

    print(f"Full train shape: {train_full.shape}")
    print(f"Full synth shape: {synth_full.shape}")

    secret = 'F13'

    # Use only low-cardinality features to avoid sparsity issues
    cardinalities = {f: train_full[f].nunique() for f in train_full.columns if f != secret}
    qi_low = [f for f, c in cardinalities.items() if c < 10]

    print(f"\nUsing low-cardinality features (<10 unique values): {len(qi_low)}")
    print(f"Features: {qi_low}")

    # Test different sizes
    sizes = [500, 1000, 2000, 5000]
    results = []

    for size in sizes:
        if size > len(train_full):
            print(f"\nSkipping {size} (exceeds dataset size)")
            continue

        acc, base, imp = test_with_size(
            train_full, synth_full, size, secret, qi_low
        )
        results.append((size, acc, base, imp))

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset Size':>15} {'Accuracy':>10} {'Baseline':>10} {'Δ':>8} {'Status':>10}")
    print("-"*70)
    for size, acc, base, imp in results:
        if acc is not None:
            status = "✓ GOOD" if imp > 0.01 else "✗ BAD"
            print(f"{size:>15} {acc:>10.2%} {base:>10.2%} {imp:>+8.2%} {status:>10}")
        else:
            print(f"{size:>15} {'FAILED':>10}")
    print(f"{'='*70}")

    print("\nCONCLUSION:")
    print("If larger datasets beat baseline, then 1000 rows is too small.")
    print("The paper tests with up to 100K rows for better performance.")


if __name__ == "__main__":
    main()
