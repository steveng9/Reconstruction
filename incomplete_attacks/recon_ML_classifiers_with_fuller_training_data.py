import os
from os.path import join

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB

from attacks.baselines_classifiers import KNN_baseline
from attacks.NN_classifier import mlp_300_reconstruction, chained_mlp_reconstruction
from util import *


mypath = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/25_PracticeProblem"

def main():
    qis = [
        "QI1",
        # "QI2",
    ]
    sdg_practice_problems = [
        "25_Demo_AIM_e1_25f_Deid.csv",
        "25_Demo_TVAE_25f_Deid.csv",
        # "25_Demo_CellSupression_25f_Deid.csv",
        # "25_Demo_Synthpop_25f_Deid.csv",
        # "25_Demo_ARF_25f_Deid.csv",
        # "25_Demo_RANKSWAP_25f_Deid.csv",
        "25_Demo_MST_e10_25f_Deid.csv",
    ]
    ml_methods = {
        # "NB": NB_reconstruction,
        # "RF": random_forest_25_25_reconstruction,
        # "LR": logistic_regression_reconstruction,
        # "lgboost": lgboost_reconstruction,
        # "MLP": mlp_300_reconstruction,
        "chained_MLP": chained_mlp_reconstruction,
        # "SVM": SVM_reconstruction,
        # "KNN": KNN_baseline,
        # "Ensemble": ensemble1_reconstruction,
        # "chained_RF": chained_rf_reconstruction,
        # "chained_NB": chained_nb_reconstruction,
    }



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
                deid = pd.read_csv(join(mypath, deid_filename))[features_25]
                add_flag = False
                # deid = get_additional_training_data(deid_filename, deid, use_only_from_deid=True, add_flag=add_flag, refine=False)
                deid = get_additional_training_data(deid_filename, deid, use_only_from_deid=False, add_flag=add_flag, refine=True)
                # deid = deid.sample(n=100_000)
                if add_flag:
                    targets.loc[:, "flag"] = 1

                recon_method_name = f"{qi_name}_{ml_name}_{sdg_method_name}"
                # recon, _, _ = ml_method(deid, targets, qi, hidden_features, problem_name=sdg_method_name, qi_name=qi_name)
                recon, _, _ = ml_method(deid, targets, qi, hidden_features)
                reconstruction_scores.loc[hidden_features, recon_method_name] = calculate_reconstruction_score(targets_original, recon, hidden_features)

                print(qi_name, recon_method_name)
                for x in reconstruction_scores.loc[sorted(hidden_features), recon_method_name].T.to_numpy():
                    print(x, end=",")
                print(reconstruction_scores[recon_method_name].mean())


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



def get_additional_training_data(deid_filename, deid, use_only_from_deid=False, add_flag=True, refine=True):
    other_data_path = Path(os.path.split(mypath)[0] + "/NIST_Red-Team_Problems1-24_v2/")

    # Find all deidentified files for this problem number
    deid_method = deid_filename.split("_")[2][:3].upper()
    pattern = f"*_{deid_method}*_Deid.csv" if use_only_from_deid else "*_Deid.csv"
    deid_files = list(other_data_path.glob(pattern))

    deid.loc[:, "flag"] = 1

    additional_training_data = [deid]
    for deid_file in deid_files:
        if refine and not \
                ("MST" in deid_file.name or \
                 "RANKSWAP" in deid_file.name or \
                 "CELL" in deid_file.name or \
                 "SYNTHPOP" in deid_file.name):
            continue
        additional_df = pd.read_csv(deid_file)[features_25]
        additional_df.loc[:, "flag"] = 0
        additional_training_data.append(additional_df)

    return pd.concat(additional_training_data, ignore_index=True)






def random_forest_25_25_reconstruction(deid, targets, qi, hidden_features, num_estimators=25, max_depth=25, classes=None, problem_name=None, qi_name=None):
    reconstructed_targets = targets.copy()
    if "flag" in deid.columns:
        qi = qi.copy()
        qi.append("flag")

    probas = []
    classes_ = []
    for hidden_feature in hidden_features:
        model = RandomForestClassifier(n_estimators=num_estimators, max_depth=max_depth)
        type_ = deid[hidden_feature].dtypes
        y_train = deid[hidden_feature]
        model.fit(deid[qi], y_train.astype(int).astype(str))
        reconstructed_targets[hidden_feature] = model.predict(targets)

        if type_ == "float64":
            reconstructed_targets[hidden_feature] = reconstructed_targets[hidden_feature].astype(float)
        else:
            reconstructed_targets[hidden_feature] = reconstructed_targets[hidden_feature].astype(int)


        real_probas = model.predict_proba(targets[qi])
        probas.append(real_probas)
        classes_.append(model.classes_)
    return reconstructed_targets, probas, classes_






def NB_reconstruction(deid, targets, qi, hidden_features, classes=None):
    reconstructed_targets = targets.copy()
    probas = []
    for hidden_feature in hidden_features:
        type_ = deid[hidden_feature].dtypes
        model = GaussianNB()
        y_train = deid[hidden_feature]
        model.fit(deid[qi].astype(str), y_train.astype(str))

        reconstructed_targets[hidden_feature] = model.predict(reconstructed_targets[qi].astype(str))
        if type_ == "float64":
            reconstructed_targets[hidden_feature] = reconstructed_targets[hidden_feature].astype(float)
        else:
            reconstructed_targets[hidden_feature] = reconstructed_targets[hidden_feature].astype(int)

        real_probas = model.predict_proba(reconstructed_targets[qi].astype(str))
        probas.append(real_probas)

    return reconstructed_targets, probas, None





if __name__ == "__main__":
    main()