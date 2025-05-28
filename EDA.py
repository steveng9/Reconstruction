import sys

import pandas as pd
from os import listdir
from os.path import isfile, join
from util import features_25, QIs, minus_QIs, calculate_reconstruction_score

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



main()