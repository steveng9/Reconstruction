"""
Test Linear Reconstruction attack integration.

Tests the wrapper with proper configuration (ALL non-secret features as QI).
"""
import sys
import os
import pandas as pd
import numpy as np

# Add recon-synth to path
sys.path.insert(0, os.path.expanduser('~/recon-synth'))

# Add SOTA_attacks to path
recon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(recon_path, 'SOTA_attacks'))

from linear_reconstruction import linear_reconstruction_attack


def test_linear_attack():
    """Test with ALL features as QI (proper configuration for this attack)."""
    print("="*70)
    print("Testing Linear Reconstruction Attack Integration")
    print("="*70)

    # Load NIST CRC data
    print("\n[1/4] Loading NIST CRC data...")
    train = pd.read_csv('/home/golobs/data/NIST_CRC/dev_data/train.csv')
    synth = pd.read_csv('/home/golobs/data/NIST_CRC/dev_data/synth.csv')

    # Drop ID column
    if 'ID' in train.columns:
        train = train.drop('ID', axis=1)

    # Sample for speed
    n_samples = 1000
    train = train.sample(n=n_samples, random_state=42)
    synth = synth.sample(n=n_samples, random_state=43)

    print(f"  Train shape: {train.shape}")
    print(f"  Synth shape: {synth.shape}")

    # IMPORTANT: QI must include ALL non-secret features for this attack to work well
    print("\n[2/4] Setting up attack parameters...")
    hidden_features = ['F13']  # Feature to reconstruct
    qi = [col for col in train.columns if col not in hidden_features]

    print(f"  QI features: {len(qi)} (ALL non-secret features)")
    print(f"  Hidden feature: {hidden_features}")

    # Prepare targets with ALL features
    targets = train.copy()

    # Config
    cfg = {
        'attack_params': {
            'k': 3,
            'n_procs': 4
        }
    }

    # Run attack
    print("\n[3/4] Running linear reconstruction attack...")
    try:
        reconstructed, _, _ = linear_reconstruction_attack(
            cfg, synth, targets, qi, hidden_features
        )
    except Exception as e:
        print(f"\n✗ Attack failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Evaluate
    print("\n[4/4] Evaluating results...")
    accuracy = np.mean(reconstructed[hidden_features[0]] == targets[hidden_features[0]])
    print(f"  Reconstruction accuracy: {accuracy:.2%}")

    # Compute baseline
    train_unique = np.unique(targets[hidden_features[0]])
    baseline = max(
        np.mean(targets[hidden_features[0]] == train_unique[0]),
        np.mean(targets[hidden_features[0]] == train_unique[1])
    )
    print(f"  Baseline (majority class): {baseline:.2%}")

    # Check output format
    print(f"\n  Reconstructed shape: {reconstructed.shape}")
    print(f"  Reconstructed columns: {list(reconstructed.columns)}")

    try:
        assert set(reconstructed.columns) == set(qi + hidden_features), "Output columns mismatch!"
        assert len(reconstructed) == len(targets), "Output length mismatch!"
    except AssertionError as e:
        print(f"\n✗ Validation failed: {e}")
        return False

    print("\n" + "="*70)
    print("✓ Integration test PASSED!")
    print("="*70)
    print("\nConfiguration notes:")
    print(f"  - Used {len(qi)} features as QI (all non-secret features)")
    print(f"  - Attack accuracy: {accuracy:.2%} vs baseline {baseline:.2%}")
    print("  - For best results, always include ALL non-secret features in QI")
    print("\nNext steps:")
    print("  1. The attack is registered as 'LinearReconstruction' (categorical)")
    print("  2. In your config, set QI to include all non-secret features")
    print("  3. Set hidden_features to a single binary feature")
    print("  4. Do NOT use with chaining or ensembling")
    print("="*70)
    return True


if __name__ == "__main__":
    success = test_linear_attack()
    sys.exit(0 if success else 1)
