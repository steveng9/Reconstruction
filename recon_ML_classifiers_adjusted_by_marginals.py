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

from baselines import KNN_baseline
from util import *



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


    for ml_name, ml_method in ml_methods.items():

        for qi_name in qis:
            reconstruction_scores = pd.DataFrame(index=features_25)

            qi = QIs[qi_name]
            hidden_features = minus_QIs[qi_name]
            targets = targets_original[qi]

            for deid_filename in sdg_practice_problems:

                sdg_method_name = "_".join(deid_filename.split("_")[2:-2])
                deid = pd.read_csv(join(mypath, deid_filename))

                recon_method_name = f"{qi_name}_{ml_name}_{sdg_method_name}"
                recon, _, _ = ml_method(deid, targets, qi, hidden_features)
                reconstruction_scores.loc[hidden_features, recon_method_name] = calculate_reconstruction_score(targets_original, recon, hidden_features)

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
            print("ave: ", round(reconstruction_scores.loc[sorted(hidden_features)].T.mean().mean(), 2))





def random_forest_25_25_reconstruction(deid, targets, qi, hidden_features, num_estimators=25, max_depth=25, classes=None, problem_name=None, qi_name=None):
    targets_copy = targets.copy()
    # aux, deid = match_aux_and_synth_classes(aux, deid)

    probas = []
    classes_ = []
    for hidden_feature in hidden_features:
        model = RandomForestClassifier(n_estimators=num_estimators, max_depth=max_depth)
        y_train = deid[hidden_feature]
        model.fit(deid[qi], y_train)
        predictions = model.predict(targets)
        y_pred_t = model.predict_proba(targets)
        y_pred_t_marg = y_pred_t.sum(axis=0)
        y_pred_d = model.predict_proba(deid[qi])
        y_pred_d_marg = y_pred_d.sum(axis=0)
        y_true_d = np.zeros((y_train.size, y_train.nunique()))
        y_true_d[np.arange(y_train.size), [model.classes_.tolist().index(label) for label in y_train]] = 1
        y_true_d_marg = y_true_d.sum(axis=0)

        # adjust the probabilities of target predictions
        y_pred_t_adjusted = y_pred_t * (y_true_d_marg / y_pred_d_marg)
        adjusted_predictions = model.classes_[y_pred_t_adjusted.argmax(axis=1)]

        targets_copy[hidden_feature] = adjusted_predictions

        if not (adjusted_predictions == predictions).all():
            print(f"adjusted {hidden_feature} predictions")

    return targets_copy, None, None



def NB_reconstruction(deid, targets, qi, hidden_features, classes=None):
    targets_copy = targets.copy()
    probas = []
    for hidden_feature in hidden_features:
        type_ = deid[hidden_feature].dtypes
        model = GaussianNB()
        y_train = deid[hidden_feature]
        model.fit(deid[qi].astype(str), y_train.astype(str))

        predictions = model.predict(targets)
        y_pred_t = model.predict_proba(targets)
        y_pred_t_marg = y_pred_t.sum(axis=0)
        y_pred_d = model.predict_proba(deid[qi])
        y_pred_d_marg = y_pred_d.sum(axis=0)
        y_true_d = np.zeros((y_train.size, y_train.nunique()))
        y_true_d[np.arange(y_train.size), [model.classes_.tolist().index(str(label)) for label in y_train]] = 1
        y_true_d_marg = y_true_d.sum(axis=0)

        # adjust the probabilities of target predictions
        y_pred_t_adjusted = y_pred_t * (y_true_d_marg / y_pred_d_marg)
        adjusted_predictions = model.classes_[y_pred_t_adjusted.argmax(axis=1)]

        targets_copy[hidden_feature] = adjusted_predictions

        if not (adjusted_predictions == predictions).all():
            print(f"adjusted {hidden_feature} predictions")

        if type_ == "float64":
            targets_copy[hidden_feature] = targets_copy[hidden_feature].astype(float)
        else:
            targets_copy[hidden_feature] = targets_copy[hidden_feature].astype(int)


    return targets_copy, None, None




if __name__ == "__main__":
    main()