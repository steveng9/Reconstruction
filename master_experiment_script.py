"""
Master experiment script for tabular RePaint experiments.
Usage: python run_experiment.py --config configs/experiment1.yaml --data_dir /path/to/data
"""
import argparse
import sys

N_RUNS_default = 1
parser = argparse.ArgumentParser()
argparse.ArgumentParser(description="Run tabular Reconstruction experiments")
parser.add_argument("--n_runs", type=int, default=N_RUNS_default, help="Number of runs to average over")
parser.add_argument("--on_server", type=bool, default=False, help="changes directories depending on which machine running on")
args = parser.parse_args()

# Set path BEFORE importing other modules
if args.on_server:
    sys.path.append('/home/golobs/MIA_on_diffusion/')
    sys.path.append('/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM')
    sys.path.append('/home/golobs/recon-synth')
    sys.path.append('/home/golobs/recon-synth/attacks')
    sys.path.append('/home/golobs/recon-synth/attacks/solvers')
else:
    sys.path.append('/Users/stevengolob/PycharmProjects/MIA_on_diffusion/')
    sys.path.append('/Users/stevengolob/PycharmProjects/MIA_on_diffusion/midst_models/single_table_TabDDPM')
    sys.path.append('/Users/stevengolob/PycharmProjects/recon-synth')
    sys.path.append('/Users/stevengolob/PycharmProjects/recon-synth/attacks')
    sys.path.append('/Users/stevengolob/PycharmProjects/recon-synth/attacks/solvers')

CONFIG_PATH_default = "/home/golobs/data/NIST_CRC/dev_data/dev_config.yaml" if args.on_server else "/Users/stevengolob/Documents/school/PhD/reconstruction_project/configs/dev_config.yaml"

import yaml
import numpy as np
import wandb

from get_data import load_data
from scoring import calculate_reconstruction_score, calculate_continuous_vals_reconstruction_score

# Attack registry (single source of truth for attack names)
from attacks import get_attack

# Enhancement wrappers
from enhancements import apply_chaining, apply_ensembling





# on_server = len(sys.argv) > 1 and sys.argv[1] == 'T'

def main():
    config = load_config(CONFIG_PATH_default)

    print(f"Running {args.n_runs} experiments with config: {CONFIG_PATH_default}")
    wandb.init(
        project=config['wandb']['project'],
        name=f"{config['wandb']['name']}",
        config=config,
        tags=config['wandb'].get('tags', []),
        group=config['wandb'].get('group', None)
    )

    for run_id in range(args.n_runs):
        print(f"\n\nStarting run {run_id + 1}/{args.n_runs}\n{'=' * 50}")
        run_single_experiment(config, run_id)

    wandb.finish()
    print(f"\n{'=' * 50}\nAll runs complete!\n{'=' * 50}")


#
# def parse_args():
#     parser = argparse.ArgumentParser(description="Run tabular Reconstruction experiments")
#     # parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Path to data directory")
#     # parser.add_argument("--config", type=str, default=CONFIG_PATH_default, help="Path to config YAML file")
#     parser.add_argument("--n_runs", type=int, default=N_RUNS_default, help="Number of runs to average over")
#     parser.add_argument("--on_server", type=bool, default=False, help="changes directories depending on which machine running on")
#     return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _prepare_config(config):
    """
    Prepare config by merging method-specific parameters into attack_params.

    This allows the config file to be organized by method (e.g., RandomForest: {params})
    while attack functions can access params directly from attack_params.

    Args:
        config: Original config dict

    Returns:
        Modified config with method-specific params merged into attack_params
    """
    config = config.copy()  # Don't modify original
    attack_method = config.get("attack_method")

    # Get method-specific params
    method_params = config["attack_params"].get(attack_method, {})

    # Create new attack_params that merges method-specific params with enhancements
    new_attack_params = {}

    # First, copy over enhancement configs (chaining, etc.)
    for key, value in config["attack_params"].items():
        if isinstance(value, dict) and "enabled" in value:
            # This is an enhancement config (has 'enabled' flag)
            new_attack_params[key] = value

    # Then merge in method-specific params
    new_attack_params.update(method_params)

    config["attack_params"] = new_attack_params

    return config


def _print_experiment_config(config):
    """Print key configuration parameters in a clear, organized format."""
    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)

    # Dataset info
    dataset = config.get("dataset", {})
    print(f"\n📊 Dataset:")
    print(f"   Name: {dataset.get('name', 'N/A')}")
    print(f"   Size: {dataset.get('size', 'N/A'):,}")
    print(f"   QI: {config.get('QI', 'N/A')}")

    # SDG method
    sdg_method = config.get("sdg_method", "N/A")
    print(f"\n🔒 Synthetic Data Generation:")
    print(f"   Method: {sdg_method}")
    sdg_params = config.get("sdg_params", {})
    if sdg_params:
        for key, val in sdg_params.items():
            print(f"   {key}: {val}")

    # Attack configuration
    attack_method = config.get("attack_method", "N/A")
    data_type = config.get("data_type", "agnostic")
    print(f"\n🎯 Reconstruction Attack:")
    print(f"   Method: {attack_method}")
    print(f"   Data Type: {data_type}")

    # Ensembling status
    ensembling = config.get("attack_params", {}).get("ensembling", {})
    ensembling_enabled = ensembling.get("enabled", False)
    print(f"\n🤝 Ensembling: {'✓ ENABLED' if ensembling_enabled else '✗ Disabled'}")
    if ensembling_enabled:
        methods = ensembling.get("methods", [])
        print(f"   Methods: {', '.join(methods)}")
        print(f"   Aggregation: {ensembling.get('aggregation', 'voting')}")
        print(f"   Include Primary: {ensembling.get('include_primary', True)}")
        if ensembling.get("weights"):
            print(f"   Weights: {ensembling.get('weights')}")

    # Chaining status
    chaining = config.get("attack_params", {}).get("chaining", {})
    chaining_enabled = chaining.get("enabled", False)
    print(f"\n🔗 Chaining: {'✓ ENABLED' if chaining_enabled else '✗ Disabled'}")
    if chaining_enabled:
        print(f"   Order Strategy: {chaining.get('order_strategy', 'default')}")
        if chaining.get("order_strategy") == "manual":
            print(f"   Order: {chaining.get('order', [])}")
        print(f"   Log Intermediate: {chaining.get('log_intermediate', True)}")

    # Attack method parameters
    attack_params = config.get("attack_params", {})
    method_params = {k: v for k, v in attack_params.items()
                     if k not in ["chaining"] and not (isinstance(v, dict) and "enabled" in v)}

    if method_params:
        print(f"\n⚙️  {attack_method} Parameters:")
        for key, val in sorted(method_params.items()):
            if isinstance(val, list):
                print(f"   {key}: [{', '.join(map(str, val))}]")
            else:
                print(f"   {key}: {val}")

    print("\n" + "=" * 70 + "\n")


def run_single_experiment(config, run_id):
    train, synth, qi, hidden_features = load_data(config)

    # Merge method-specific params into attack_params for easier access
    config = _prepare_config(config)

    # Print experiment configuration
    if run_id == 0:  # Only print on first run to avoid repetition
        _print_experiment_config(config)

    # Get base attack method from registry
    data_type = config.get("data_type", "agnostic")  # Default to agnostic (TabDDPM, RePaint)
    attack_method = get_attack(config["attack_method"], data_type)

    # Apply enhancements (composable wrappers)
    # Each enhancement wrapper checks its own config and bypasses if not enabled
    # Order matters: ensembling first (combines methods), then chaining (sequential prediction)
    attack_method = apply_ensembling(attack_method, config)
    reconstructed, _, _ = apply_chaining(attack_method, config, synth, train, qi, hidden_features)

    # Score based on dataset type (continuous vs categorical)
    dataset_type = config.get("dataset", {}).get("type", "categorical")
    if dataset_type == "continuous":
        scores = calculate_continuous_vals_reconstruction_score(train, reconstructed, hidden_features)
        scores = scores["normalized_rmse"].values
    else:  # categorical
        scores = calculate_reconstruction_score(train, reconstructed, hidden_features)

    results = {f"RA_{k}": v for k, v in zip(hidden_features, scores)}
    results["RA_mean"] = np.mean(scores)

    scores = list(results.values())
    print(f"\n{np.array(scores[:-1])}")
    print(f"ave: {scores[-1]}\n{'=' * 50}")
    wandb.log(results)


if __name__ == "__main__":
    main()
