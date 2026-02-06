import sys

import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from util import *


def _development():
    qis = [
        "QI1",
        "QI2",
    ]
    sdg_practice_problems = [
        "25_Demo_AIM_e1_25f_Deid.csv",
        "25_Demo_TVAE_25f_Deid.csv",
        "25_Demo_CellSupression_25f_Deid.csv",
        "25_Demo_Synthpop_25f_Deid.csv",
        "25_Demo_ARF_25f_Deid.csv",
        "25_Demo_RANKSWAP_25f_Deid.csv",
        "25_Demo_MST_e10_25f_Deid.csv",
    ]
    mypath = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/25_PracticeProblem/"
    target_filename = "25_Demo_25f_OriginalData.csv"
    targets_original = pd.read_csv(join(mypath, target_filename))

    for qi_name in qis:

        reconstruction_scores = pd.DataFrame(index=features_25)

        qi = QIs[qi_name]
        hidden_features = list(set(features_25).difference(set(qi)))
        targets = targets_original[qi]

        for deid_filename in sdg_practice_problems:
            sdg_method_name = "_".join(deid_filename.split("_")[2:-2])
            # print("\n", qi_name, sdg_method_name, ml_name)
            deid = pd.read_csv(join(mypath, deid_filename))

            recon_method_name = f"{qi_name}_{sdg_method_name}"

            method_name = "_".join(deid_filename.split("_")[2:-2])
            print(deid_filename)
            deid = pd.read_csv(join(mypath, deid_filename))
            #
            # baseline_name = f"{qi_name}_{method_name}"
            # reconstruction_scores[baseline_name] = np.NAN
            # recon = mode_baseline(deid, targets, hidden_features)
            # reconstruction_scores.loc[hidden_features, baseline_name] = simple_accuracy_score(targets_original, recon, hidden_features)

            reconstruction_scores[recon_method_name] = np.NAN
            recon = simply_measure_deid_itself_baseline(deid, targets, qi, hidden_features)
            reconstruction_scores.loc[hidden_features, recon_method_name] = calculate_reconstruction_score(targets_original, recon, hidden_features)

        print()

        print(qi_name)
        for l in reconstruction_scores.loc[sorted(hidden_features)].T.to_numpy():
            for x in l:
                print(x, end=",")
            print()
            # print(round(l.mean(), 2))
        print("ave: ", round(reconstruction_scores.loc[sorted(hidden_features)].T.mean().mean(), 2))


    print()






def simply_measure_deid_itself_baseline(cfg, deid, targets, qi, hidden_features):
    reconstructed_targets = targets.copy()
    full_cycles = len(deid) // len(targets)
    remainder = len(deid) % len(targets)

    result_parts = [deid] * full_cycles
    if remainder > 0:
        result_parts.append(deid.iloc[:remainder])

    reconstructed_targets[hidden_features] = pd.concat(result_parts, ignore_index=True)[hidden_features]

    return reconstructed_targets


def random_baseline(cfg, deid, targets, qi, hidden_features):
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        reconstructed_targets[hidden_feature] = deid[hidden_feature].sample().values[0]
    return reconstructed_targets


def mean_baseline(cfg, deid, targets, qi, hidden_features):
    # TODO: mean won't work for discrete values. Fix this somehow.
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        reconstructed_targets[hidden_feature] = deid[hidden_feature].mean()
    return reconstructed_targets


def slightly_better_mean_baseline(cfg, deid, targets, qi, hidden_features):
    reconstructed_targets = targets.copy()
    lookup = deid.groupby(qi)[hidden_features].mean().round().reset_index()

    recon = pd.merge(
        reconstructed_targets,
        lookup,
        on=qi,
        how='left'
    )

    for feature in hidden_features:
        if recon[feature].isna().any():
            global_mode = round(deid[feature].mode()[0])
            recon[feature] = recon[feature].fillna(global_mode)

    for feature in hidden_features:
        recon[feature] = recon[feature].astype(int) #ro match with types

    return recon


def mode_baseline(cfg, deid, targets, qi, hidden_features):
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        reconstructed_targets[hidden_feature] = deid[hidden_feature].mode()[0]
    return reconstructed_targets




if __name__ == "__main__":
    _development()