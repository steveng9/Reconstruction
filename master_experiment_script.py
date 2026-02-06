"""
Master experiment script for tabular RePaint experiments.
Usage: python run_experiment.py --config configs/experiment1.yaml --data_dir /path/to/data
"""
import sys
import os
import yaml
import numpy as np
import argparse
import wandb

from get_data import load_data
from scoring import calculate_reconstruction_score


from attacks.NN_classifier import mlp_classification_reconstruction
from attacks.ML_classifiers import KNN_reconstruction, lgboost_reconstruction, SVM_classification_reconstruction
from attacks.partialDiffusion import repaint_reconstruction, partial_tabddpm_reconstruction
from attacks.attention_classifier import attention_reconstruction




N_RUNS_default = 1

CONFIG_PATH_default = "/Users/stevengolob/Documents/school/PhD/reconstruction_project/configs/dev_config.yaml"

def main():
    args = parse_args()
    config = load_config(args.config)

    print(f"Running {args.n_runs} experiments with config: {args.config}")
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



def parse_args():
    parser = argparse.ArgumentParser(description="Run tabular Reconstruction experiments")
    # parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Path to data directory")
    parser.add_argument("--config", type=str, default=CONFIG_PATH_default, help="Path to config YAML file")
    parser.add_argument("--n_runs", type=int, default=N_RUNS_default, help="Number of runs to average over")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_single_experiment(config, run_id):
    train, synth, qi, hidden_features = load_data(config)

    attack_method = globals()[config["attack_params"]["method_ref"]]
    reconstructed, _, _ = attack_method(config, synth, train[qi], qi, hidden_features)
    scores = calculate_reconstruction_score(train, reconstructed, hidden_features)

    results = {f"RA_{k}": v for k, v in zip(hidden_features, scores)}
    results["RA_mean"] = np.mean(scores)
    wandb.log(results)

    scores = list(results.values())
    print(f"\n{np.array(scores[:-1])}")
    print(f"ave: {scores[-1]}\n{'=' * 50}")


if __name__ == "__main__":
    main()
