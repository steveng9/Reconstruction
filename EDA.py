import sys
from pathlib import Path

import os
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from util import features_25, features_50, QIs, minus_QIs, calculate_reconstruction_score

base_path = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/"

def main():
    QI = "QI1"

    reconstructed = pd.read_csv(f"/Users/golobs/PycharmProjects/smartnoise-sdk/reconstructed_data_{QI}.csv").iloc[:, 1:]
    # reconstructed = pd.read_csv("25_Demo_25f_Reconstructed.csv").iloc[:, 1:]
    original = pd.read_csv("/Users/golobs/Documents/GradSchool/NIST-CRC-25/25_PracticeProblem/25_Demo_25f_OriginalData.csv")
    hidden_features = minus_QIs[QI]

    reconstruction_scores = pd.DataFrame(index=features_25)
    attack_name = "partial_MST_attack"
    reconstruction_scores.loc[hidden_features, attack_name] = calculate_reconstruction_score(original, reconstructed, hidden_features)

    for x in reconstruction_scores.loc[sorted(hidden_features), attack_name].T.to_numpy():
        print(x, end=",")
    print(round(reconstruction_scores.loc[sorted(hidden_features), attack_name].T.to_numpy().mean(), 2))


def examine_domains():

    front_cols = features_25
    remaining_cols = [col for col in features_50 if col not in front_cols]
    # remaining_cols = []
    new_column_order = front_cols + remaining_cols
    nunique = pd.DataFrame(columns=["deid_method"] + new_column_order)

    i = 0
    # for problem_num in range(1, 25):
    for problem_num in [25]:
        # Find all deidentified files for this problem number
        pattern = f"{problem_num}_*.csv"
        deid_files = list((Path(base_path) / "25_PracticeProblem").glob(pattern))

        if not deid_files:
            print(f"Warning: No deidentified files found for problem {problem_num}")
            continue

        for deid_file in deid_files:
            if "QI" not in deid_file.name and "50" in deid_file.name:
                try:
                    df = pd.read_csv(deid_file)[features_50]
                except:
                    print()
                if 'target' in df.columns:
                    df.drop(columns=['target'], inplace=True)
                file_name_parts = os.path.basename(deid_file).split("_")
                deid_method = file_name_parts[2]
                counts = df.nunique()
                nunique.loc[i] = [deid_method] + [counts[col] for col in new_column_order]
                i += 1

    print()


def examine_other_dataset_domains():
    data_path = "/Users/golobs/Downloads/national2018_b.csv"
    df = pd.read_csv(data_path)
    nunique = pd.DataFrame(columns=df.columns)

    counts = df.nunique()
    nunique.loc[0] = [counts[col] for col in df.columns]
    print()


def compare_domains():
    domain_other = [20, 96, 2, 7, 5, 9, 11, 12, 3, 3, 20, 256, 20, 13, 2556, 11, 503, 7, 3, 3, 2, 2, 602, 471]
    domain = [96, 20, 94, 5, 100, 13, 3, 20, 2, 5, 7, 4, 620, 108, 3, 5, 5, 8, 6, 3, 6, 2, 2, 4, 53]

    domain_other = sorted(domain_other)
    domain = sorted(domain)

    print("NIST CRC: other:")
    for n, o in zip(domain,domain_other):
        print(f"\t{n}, \t{o}")




if __name__ == "__main__":
    # main()
    # examine_domains()
    # examine_other_dataset_domains()
    compare_domains()
