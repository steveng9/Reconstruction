import math
import sys

import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

from get_aux_data import get_rankswap_data_as_auxiliary, create_auxiliary_combined
from baselines import KNN_baseline
from recon_ML_classifiers import random_forest_25_25_reconstruction, NB_reconstruction
from util import *



# aux_fn = get_rankswap_data_as_auxiliary
aux_fn = create_auxiliary_combined


def main():
    qis = [
        # "QI1",
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
    ml_methods = {
        "NB": NB_reconstruction,
        # "RF": random_forest_25_25_reconstruction,
        # "LR": logistic_regression_reconstruction,
        # "lgboost": lgboost_reconstruction,
        # "MLP": MLP_reconstruction,
        # "SVM": SVM_reconstruction,
        # "KNN": KNN_baseline,
        # "Ensemble": ensemble1_reconstruction,
        # "chained_RF": chained_rf_reconstruction,
        # "chained_NB": chained_nb_reconstruction,
    }

    mypath = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/25_PracticeProblem/"
    target_filename = "25_Demo_25f_OriginalData.csv"
    targets_original = pd.read_csv(join(mypath, target_filename))
    aux = aux_fn()[features_25].astype(int).sample(n=30_000)
    # aux = aux_fn()[features_25].astype(int).sample(n=100_000)
    # aux = aux_fn()[features_25].astype(int)



    for ml_name, ml_method in ml_methods.items():

        for qi_name in qis:
            reconstruction_scores = pd.DataFrame(index=features_25)
            qi = QIs[qi_name]
            hidden_features = minus_QIs[qi_name]
            targets = targets_original[qi]

            for deid_filename in sdg_practice_problems:

                sdg_method_name = "_".join(deid_filename.split("_")[2:-2])
                # print("\n", qi_name, sdg_method_name, ml_name)
                deid = pd.read_csv(join(mypath, deid_filename))


                aux, deid = match_aux_and_synth_classes(aux, deid)

                recon_method_name = f"{qi_name}_{ml_name}_{sdg_method_name}"
                # recon = logistic_regression_reconstruction(deid, targets, qi, hidden_features)
                recon, _, _ = probability_ratio_reconstruction(ml_method, deid, aux, targets, qi, hidden_features, keep_proportions=True)
                reconstruction_scores.loc[hidden_features, recon_method_name] = calculate_reconstruction_score(targets_original, recon, hidden_features)
                #
                print(qi_name, recon_method_name)
                # for x in reconstruction_scores.loc[sorted(hidden_features), recon_method_name].T.to_numpy():
                #     print(x, end=",")
                print(reconstruction_scores[recon_method_name].mean())


            # Set display width very large to avoid wrapping and truncating
            pd.set_option('display.width', 1000)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            print(qi_name)
            for l in reconstruction_scores.loc[sorted(hidden_features)].T.to_numpy():
                for x in l:
                    print(x, end=",")
                print()
                # print(round(l.mean(), 2))
            print("ave: ", round(reconstruction_scores.loc[sorted(hidden_features)].T.mean().mean(), 2))





def probability_ratio_reconstruction(base_reconstruction_method, deid, auxiliary_dataset, targets, qi, hidden_features, keep_proportions=False):
    targets_copy = targets.copy()
    classes = {col: list(set(auxiliary_dataset[col].unique()).union(set(deid[col].unique()))) for col in auxiliary_dataset.columns}
    _, base_probas, _ = base_reconstruction_method(deid, targets, qi, hidden_features, classes=classes)
    _, aux_probas, _ = base_reconstruction_method(auxiliary_dataset, targets, qi, hidden_features, classes=classes)

    for i, hidden_feature in enumerate(hidden_features):
        base_proba = base_probas[i]
        aux_proba = aux_probas[i]
        featuere_classes = classes[hidden_feature]
        if base_proba.shape[1] == aux_proba.shape[1]:
            epsilon = 1e-10
            ratio = base_proba / (aux_proba + epsilon)

            # Normalize ratios to sum to 1 for each sample
            ratio_normalized = ratio / (ratio.sum(axis=1, keepdims=True) + epsilon)

            if keep_proportions:
                # Get the row-wise max value
                row_max = ratio_normalized.max(axis=1)

                # Get sorted order (row indices) based on row-wise max value
                sorted_indices = np.argsort(-row_max)

                class_counts = {cl: 0 for cl in featuere_classes}
                deid_class_counts = {cl: 0 for cl in featuere_classes}
                deid_class_counts.update(dict(deid[hidden_feature].value_counts()))
                deid_class_counts = {cl: math.ceil(ct*len(targets)/len(deid)) for cl, ct in deid_class_counts.items()}
                feature_recon = np.zeros(len(targets))
                classes_copy = np.array(featuere_classes.copy())
                for k, idx in enumerate(sorted_indices):
                    try:
                        preferred_class_idx = ratio_normalized[idx].argmax()
                        preferred_class = classes_copy[preferred_class_idx]
                        feature_recon[idx] = preferred_class
                        class_counts[preferred_class] += 1

                        if class_counts[preferred_class] >= deid_class_counts[preferred_class]:
                            ratio_normalized = np.delete(ratio_normalized, preferred_class_idx, axis=1)
                            classes_copy = np.delete(classes_copy, preferred_class_idx)
                    except Exception as e:
                        print("Error! ", k, idx, hidden_feature, e)
                targets_copy[hidden_feature] = feature_recon

            else:
                targets_copy[hidden_feature] = np.array(featuere_classes)[ratio_normalized.argmax(axis=1)]


        else:
            # Handle case where class counts differ
            print(f"Warning: Class count mismatch for {hidden_feature}")


    return targets_copy, None, None


if __name__ == "__main__":
    main()

