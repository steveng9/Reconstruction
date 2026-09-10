"""
A complete, runnable example of adding your own reconstruction attack.

Run it from the repository root, in the `recon_` environment or inside the
Docker image:

    python examples/add_your_own_attack.py

It needs no downloads, no GPU and no Weights & Biases account: it uses the
dummy dataset shipped in `data/dummy/`, and finishes in well under a minute.

What it demonstrates
--------------------
An attack in this framework is one function with one signature. Write it,
register it, and it is immediately usable by every sweep script in the
repository and composable with the chaining and ensembling wrappers. This
script does exactly that, then scores the new attack against the mode baseline
and against CoBP-RA (the paper's strongest attack) on identical data, with the
same rarity-weighted metric used for every number in the paper.

To make the attack permanent rather than registered at run time, put the
function in `attacks/`, add it to `ATTACK_REGISTRY` in `attacks/__init__.py`
under the right `data_type`, and give it defaults in `attack_defaults.py`.
Nothing else in the repository needs to change.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from attacks import ATTACK_REGISTRY, get_attack       # noqa: E402
from get_data import load_data                        # noqa: E402
from scoring import calculate_reconstruction_score    # noqa: E402


# ---------------------------------------------------------------------------
# 1. Write the attack.
# ---------------------------------------------------------------------------
# The signature is fixed:
#
#     attack(cfg, synth, targets, qi, hidden_features)
#         -> (reconstructed_df, probas | None, classes | None)
#
#   cfg              the experiment config; per-attack parameters have already
#                    been merged into cfg["attack_params"]
#   synth            the synthetic data released by the generator - this, plus
#                    the quasi-identifiers, is everything the attacker sees
#   targets          the target records. ONLY the `qi` columns may be read; the
#                    `hidden_features` columns are the ground truth being
#                    predicted, and reading them would be cheating
#   qi               list of known (quasi-identifier) column names
#   hidden_features  list of column names to reconstruct
#
# Return a copy of `targets` with the hidden columns filled in. The second and
# third return values are class probabilities and class labels, used only by
# the soft-voting ensembler; returning None opts out of ensembling.
#
# The example below is deliberately simple: for each target record, find the
# synthetic records that agree on the most quasi-identifiers, and take the
# majority vote within that group. It is a "conditional mode" attack - a real,
# if basic, member of the paper's taxonomy.

def nearest_qi_match_reconstruction(cfg, synth, targets, qi, hidden_features):
    params = cfg.get("attack_params", {})
    min_group = params.get("min_group_size", 5)

    reconstructed = targets.copy()

    # Fall back to the global mode wherever no group is large enough to vote.
    for feature in hidden_features:
        reconstructed[feature] = synth[feature].mode()[0]

    for idx, row in targets.iterrows():
        # Score every synthetic record by how many quasi-identifiers it matches.
        agreement = pd.Series(0, index=synth.index)
        for col in qi:
            agreement += (synth[col] == row[col]).astype(int)

        # Take the best-matching group, relaxing the match until it is big
        # enough to vote with.
        for threshold in range(len(qi), -1, -1):
            group = synth[agreement >= threshold]
            if len(group) >= min_group:
                break

        for feature in hidden_features:
            modes = group[feature].mode()
            if len(modes):
                reconstructed.at[idx, feature] = modes[0]

    return reconstructed, None, None


# ---------------------------------------------------------------------------
# 2. Register it.
# ---------------------------------------------------------------------------
# In the repository proper this is a one-line edit to attacks/__init__.py. Here
# it is done at run time so that the example stays self-contained.

ATTACK_REGISTRY["categorical"]["NearestQIMatch"] = nearest_qi_match_reconstruction


# ---------------------------------------------------------------------------
# 3. Run it, and compare it against attacks already in the taxonomy.
# ---------------------------------------------------------------------------

def prepare(config, attack_name):
    """Merge the per-attack parameter block into attack_params.

    This mirrors `_prepare_config` in master_experiment_script.py, which does
    the same thing for real runs.
    """
    config = dict(config)
    merged = {
        key: value for key, value in config["attack_params"].items()
        if isinstance(value, dict) and "enabled" in value
    }
    merged.update(config["attack_params"].get(attack_name, {}))
    config["attack_params"] = merged
    return config


def main():
    with open(REPO_ROOT / "configs" / "demo_dummy.yaml") as handle:
        base_config = yaml.safe_load(handle)

    train, synth, qi, hidden_features, _ = load_data(base_config)

    print(f"Dataset        : {base_config['dataset']['name']} (n={len(train)}), "
          f"synthetic data from {base_config['sdg_method']} "
          f"eps={base_config['sdg_params']['epsilon']}")
    print(f"Known (QI)     : {', '.join(qi)}")
    print(f"To reconstruct : {', '.join(hidden_features)}")
    print()

    contenders = [
        ("NearestQIMatch (this example)", "NearestQIMatch"),
        ("Mode baseline", "Mode"),
        ("CoBP-RA (paper's strongest)", "CoBP-RA"),
    ]

    print(f"{'Attack':<32}{'mean R_adv':>12}")
    print("-" * 44)
    for label, name in contenders:
        config = prepare(base_config, name)
        attack = get_attack(name, "categorical")
        reconstructed, _, _ = attack(config, synth, train, qi, hidden_features)
        scores = calculate_reconstruction_score(train, reconstructed, hidden_features)
        print(f"{label:<32}{sum(scores) / len(scores):>12.1f}")

    print()
    print("R_adv is the rarity-weighted reconstruction advantage used for every")
    print("number in the paper (see scoring.py): 0 is a useless attack, 100 is")
    print("perfect reconstruction, and rare attribute values count for more than")
    print("common ones. Any attack scoring above the mode baseline is recovering")
    print("real information rather than guessing the most frequent value.")
    print()
    print("NearestQIMatch and the mode baseline are deterministic; CoBP-RA is not")
    print("(its random forests are unseeded), so it moves by a few tenths of a")
    print("point between runs.")


if __name__ == "__main__":
    main()
