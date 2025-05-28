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
    ml_methods = {
        # "NB": NB_reconstruction,
        # "RF": random_forest_25_25_reconstruction,
        # "LR": logistic_regression_reconstruction,
        # "lgboost": lgboost_reconstruction,
        # "MLP": MLP_reconstruction,
        # "SVM": SVM_reconstruction,
        # "KNN": KNN_baseline,
        # "Ensemble": ensemble1_reconstruction,
        "chained_RF": chained_rf_reconstruction,
        # "chained_NB": chained_nb_reconstruction,
    }

    mypath = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/25_PracticeProblem/"
    target_filename = "25_Demo_25f_OriginalData.csv"
    targets_original = pd.read_csv(join(mypath, target_filename))


    for ml_name, ml_method in ml_methods.items():

        for qi_name in qis:
            reconstruction_scores = pd.DataFrame(index=features_25)
            reconstruction_scores["nunique"] = targets_original.nunique()

            qi = QIs[qi_name]
            hidden_features = minus_QIs[qi_name]
            # hidden_features = list(set(features_25).difference(set(qi)))
            reconstruction_scores[f"{qi_name}_random_guess"] = np.NAN
            reconstruction_scores.loc[hidden_features, f"{qi_name}_random_guess"] = pd.Series(round(1 / targets_original.nunique() * 100, 1), index=features_25)
            targets = targets_original[qi]

            for deid_filename in sdg_practice_problems:

                sdg_method_name = "_".join(deid_filename.split("_")[2:-2])
                # print("\n", qi_name, sdg_method_name, ml_name)
                deid = pd.read_csv(join(mypath, deid_filename))

                recon_method_name = f"{qi_name}_{ml_name}_{sdg_method_name}"
                # recon = logistic_regression_reconstruction(deid, targets, qi, hidden_features)
                recon, _, _ = ml_method(deid, targets, qi, hidden_features)
                reconstruction_scores.loc[hidden_features, recon_method_name] = calculate_reconstruction_score(targets_original, recon, hidden_features)

                # print(qi_name, recon_method_name)
                # for x in reconstruction_scores.loc[sorted(hidden_features), recon_method_name].T.to_numpy():
                #     print(x, end=",")


            # Set display width very large to avoid wrapping and truncating
            pd.set_option('display.width', 1000)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            print(qi_name)
            for l in reconstruction_scores.loc[sorted(hidden_features)].T.to_numpy()[2:]:
                for x in l:
                    print(x, end=",")
                print()
                # print(round(l.mean(), 2))
            print("ave: ", round(reconstruction_scores.loc[sorted(hidden_features)].T.iloc[2:].mean().mean(), 2))








def logistic_regression_reconstruction(deid, targets, qi, hidden_features):
    targets_copy = targets.copy()
    for hidden_feature in hidden_features:
        model = LogisticRegression(max_iter=100)
        y = deid[hidden_feature].astype(str)
        if y.nunique() < 2:
            y[0] = 1

        model.fit(deid[qi].astype(str), y)
        targets_copy[hidden_feature] = model.predict(targets.astype(str))
    return targets_copy.astype(int), None


def random_forest_reconstruction(deid, targets, qi, hidden_features, num_estimators=100, max_depth=10):
    targets_copy = targets.copy()
    probas = []
    classes_ = []
    for hidden_feature in hidden_features:
        model = RandomForestClassifier(n_estimators=num_estimators, max_depth=max_depth)
        model.fit(deid[qi], deid[hidden_feature])
        targets_copy[hidden_feature] = model.predict(targets)
        probas.append(model.predict_proba(targets))
        classes_.append(model.classes_)
    return targets_copy, probas, classes_


def random_forest_25_25_reconstruction(deid, targets, qi, hidden_features, num_estimators=25, max_depth=25, classes=None):
    targets_copy = targets.copy()
    probas = []
    classes_ = []
    for hidden_feature in hidden_features:
        model = RandomForestClassifier(n_estimators=num_estimators, max_depth=max_depth)
        y_train = deid[hidden_feature]
        model.fit(deid[qi], y_train)
        all_classes = model.classes_
        if classes is not None:
            all_classes = np.array(classes[hidden_feature])
            model.classes_ = all_classes

        # Fill in missing class columns with 0s if needed
        def full_proba_matrix(probas, trained_classes, all_classes):
            full_probas = np.zeros((probas.shape[0], len(all_classes)))
            for i, cls in enumerate(trained_classes):
                full_index = np.where(all_classes == cls)[0][0]
                full_probas[:, full_index] = probas[:, i]
            return full_probas

        try:
            targets_copy[hidden_feature] = model.predict(targets)
        except Exception:
            print()
        real_probas = model.predict_proba(targets)
        trained_classes = model.classes_[np.isin(model.classes_, np.unique(y_train))]
        full_probas = full_proba_matrix(real_probas, trained_classes, all_classes)
        probas.append(full_probas)
        classes_.append(model.classes_)
    return targets_copy, probas, classes_


def chained_rf_reconstruction(deid, targets, qi, hidden_features):
    targets_copy = targets.copy()
    qi_copy = qi.copy()
    probas = []
    classes_ = []
    for hidden_feature in hidden_features:
        model = RandomForestClassifier(n_estimators=25, max_depth=25)
        type_ = deid[hidden_feature].dtypes
        model.fit(deid[qi_copy].astype(str), deid[hidden_feature].astype(str))
        targets_copy[hidden_feature] = model.predict(targets_copy[qi_copy])
        if type_ == "float64":
            targets_copy[hidden_feature] = targets_copy[hidden_feature].astype(float)
        else:
            targets_copy[hidden_feature] = targets_copy[hidden_feature].astype(int)
        probas.append(model.predict_proba(targets_copy[qi_copy]))
        classes_.append(model.classes_)
        qi_copy.append(hidden_feature)
    return targets_copy, probas, classes_


def lgboost_reconstruction(deid, targets, qi, hidden_features):
    targets_copy = targets.copy()
    probas = []
    classes_ = []

    for hidden_feature in hidden_features:
        params = {
            "num_class": max(2, deid[hidden_feature].nunique()),
            # 'objective': 'cross_entropy',
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            # 'metric': 'multi_error',
            # 'metric': 'auc_mu',
            'n_estimators': 100,
            'verbosity': -1,
        }

        # Create the LGBM classifier
        model = lgb.LGBMClassifier(**params)
        y = deid[hidden_feature]
        if y.nunique() < 2:
            y[0] = 99
        # try:
        model.fit(deid[qi], y)
        # except lgb.basic.LightGBMError:
        #     print()
        targets_copy[hidden_feature] = model.predict(targets)
        probas.append(model.predict_proba(targets))
        classes_.append(model.classes_)
    return targets_copy, probas, classes_



def SVM_reconstruction(deid, targets, qi, hidden_features):
    return None, None

def NB_reconstruction(deid, targets, qi, hidden_features):
    targets_copy = targets.copy()
    for hidden_feature in hidden_features:
        model = GaussianNB()
        type_ = deid[hidden_feature].dtypes
        model.fit(deid[qi].astype(str), deid[hidden_feature].astype(str))
        targets_copy[hidden_feature] = model.predict(targets[qi].astype(str))
        if type_ == "float64":
            targets_copy[hidden_feature] = targets_copy[hidden_feature].astype(float)
        else:
            targets_copy[hidden_feature] = targets_copy[hidden_feature].astype(int)
    return targets_copy, None, None

def chained_nb_reconstruction(deid, targets, qi, hidden_features):
    targets_copy = targets.copy()
    qi_copy = qi.copy()
    for hidden_feature in hidden_features:
        model = GaussianNB()
        type_ = deid[hidden_feature].dtypes
        model.fit(deid[qi_copy].astype(str), deid[hidden_feature].astype(str))
        targets_copy[hidden_feature] = model.predict(targets_copy[qi_copy].astype(str))
        qi_copy.append(hidden_feature)
        if type_ == "float64":
            targets_copy[hidden_feature] = targets_copy[hidden_feature].astype(float)
        else:
            targets_copy[hidden_feature] = targets_copy[hidden_feature].astype(int)
    return targets_copy, None, None

def ensemble1_reconstruction(deid, targets, qi, hidden_features):
    targets_copy = targets.copy()
    for hidden_feature in hidden_features:
        rf_recon, rf_probas, rf_classes_ = random_forest_25_25_reconstruction(deid, targets, qi, [hidden_feature])
        lgb_recon, lgb_probas, lgb_classes_ = lgboost_reconstruction(deid, targets, qi, [hidden_feature])
        knn_recon, _, _ = KNN_baseline(deid, targets, qi, [hidden_feature])

        rf_recon, rf_probas = rf_recon[hidden_feature], rf_probas[0]
        lgb_recon, lgb_probas = lgb_recon[hidden_feature], lgb_probas[0] if lgb_classes_[0].shape == rf_classes_[0].shape else np.delete(lgb_probas[0], 1, axis=1)
        knn_recon = knn_recon[hidden_feature]

        ohe = OneHotEncoder(categories=[rf_classes_[0]], sparse_output=False)
        reshaped_for_enc = knn_recon.values.reshape(-1, 1)
        knn_probas = ohe.fit_transform(reshaped_for_enc)

        ensemble = np.mean([rf_probas, lgb_probas, knn_probas], axis=0)
        predicted_indices = np.argmax(ensemble, axis=1)
        idx_to_class = dict(enumerate(rf_classes_[0]))
        predicted_classes = np.array([idx_to_class[idx] for idx in predicted_indices])
        targets_copy[hidden_feature] = predicted_classes

    return targets_copy, ensemble, None





if __name__ == "__main__":
    main()