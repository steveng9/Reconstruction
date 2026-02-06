import pandas as pd

from attacks.baselines_classifiers import KNN_baseline
from attacks.ML_classifiers import chained_rf_reconstruction, chained_nb_reconstruction, NB_reconstruction, \
    random_forest_25_25_reconstruction, ensemble1_reconstruction, random_forest_reconstruction
from attacks.NN_classifier import mlp_300_reconstruction, mlp_128_96_64_reconstruction
from util import minus_QIs, QIs, features_50

problems = {
    # ("11_AIM_e1_25f_QID1", "QI1", 25, {'F23': MLP_128_96_64, 'F13': , 'F11': , 'F43': , 'F36': , 'F15': , 'F33': , 'F25': , 'F18': , 'F5': , 'F30': , 'F10': , 'F12': , 'F50': , 'F3': , 'F1': , 'F9': , 'F21': }),
    # ("13_AIM_e10_25f_QID1", "QI1", 25, {'F23': MLP_128_96_64, 'F13': , 'F11': , 'F43': , 'F36': , 'F15': , 'F33': , 'F25': , 'F18': , 'F5': , 'F30': , 'F10': , 'F12': , 'F50': , 'F3': , 'F1': , 'F9': , 'F21': }),
    # ("17_TVAE_25f_QID1", "QI1", 25, {'F23': , 'F13': gnn, 'F11': , 'F43': , 'F36': , 'F15': , 'F33': , 'F25': , 'F18': , 'F5': , 'F30': , 'F10': , 'F12': , 'F50': , 'F3': , 'F1': , 'F9': , 'F21': }),
    # ("19_CELL_SUPPRESSION_25f_QID1", "QI1", 25, {'F23': lgboost_reconstruction, 'F13': , 'F11': , 'F43': , 'F36': , 'F15': , 'F33': , 'F25': , 'F18': , 'F5': , 'F30': , 'F10': , 'F12': , 'F50': , 'F3': , 'F1': , 'F9': , 'F21': }),
    # ("1_SYNTHPOP_25f_QID1", "QI1", 25, {'F23': rf_100_10, 'F13': , 'F11': , 'F43': , 'F36': , 'F15': , 'F33': , 'F25': , 'F18': , 'F5': , 'F30': , 'F10': , 'F12': , 'F50': , 'F3': , 'F1': , 'F9': , 'F21': }),
    # ("15_ARF_25f_QID1", "QI1", 25, {'F23': rf_100_10, 'F13': , 'F11': , 'F43': , 'F36': , 'F15': , 'F33': , 'F25': , 'F18': , 'F5': , 'F30': , 'F10': , 'F12': , 'F50': , 'F3': , 'F1': , 'F9': , 'F21': }),
    # ("23_RANKSWAP_25f_QID1", "QI1", 25, {'F23': chained_rf_100_10_reconstruction, 'F13': , 'F11': , 'F43': , 'F36': , 'F15': , 'F33': , 'F25': , 'F18': , 'F5': , 'F30': , 'F10': , 'F12': , 'F50': , 'F3': , 'F1': , 'F9': , 'F21': }),
    # ("5_MST_e1_25f_QID1", "QI1", 25, {'F23': rf_100_10, 'F13': , 'F11': , 'F43': , 'F36': , 'F15': , 'F33': , 'F25': , 'F18': , 'F5': , 'F30': , 'F10': , 'F12': , 'F50': , 'F3': , 'F1': , 'F9': , 'F21': }),
    # ("7_MST_e10_25f_QID1", "QI1", 25, {'F23': rf_100_10, 'F13': , 'F11': , 'F43': , 'F36': , 'F15': , 'F33': , 'F25': , 'F18': , 'F5': , 'F30': , 'F10': , 'F12': , 'F50': , 'F3': , 'F1': , 'F9': , 'F21': }),
    "QI1": [
        "11_AIM_e1_25f_QID1",
        "13_AIM_e10_25f_QID1",
        "17_TVAE_25f_QID1",
        "19_CELL_SUPPRESSION_25f_QID1",
        "1_SYNTHPOP_25f_QID1",
        "15_ARF_25f_QID1",
        "23_RANKSWAP_25f_QID1",
        "5_MST_e1_25f_QID1",
        "7_MST_e10_25f_QID1",
        # "21_CELL_SUPPRESSION_50f_QID1",
        # "3_SYNTHPOP_50f_QID1",
        # "9_MST_e10_50f_QID1",
    ],
    "QI2": [
        "12_AIM_e1_25f_QID2",
        "14_AIM_e10_25f_QID2",
        "16_ARF_25f_QID2",
        "18_TVAE_25f_QID2",
        "20_CELL_SUPPRESSION_25f_QID2",
        "24_RANKSWAP_25f_QID2",
        "2_SYNTHPOP_25f_QID2",
        "6_MST_e1_25f_QID2",
        "8_MST_e10_25f_QID2",
        # "22_CELL_SUPPRESSION_50f_QID2",
        # "4_SYNTHPOP_50f_QID2",
        # "10_MST_e10_50f_QID2",
    ]
}

# NOTE: For the 50-feature problems, simply use the chained RF reconstruction method.
problems50f = {
    "QI1": [
        "21_CELL_SUPPRESSION_50f_QID1",
        "3_SYNTHPOP_50f_QID1",
        "9_MST_e10_50f_QID1",
    ],
    "QI2": [
        "22_CELL_SUPPRESSION_50f_QID2",
        "4_SYNTHPOP_50f_QID2",
        "10_MST_e10_50f_QID2",
    ]
}

methods = {
    "QI1": [
        ('F12', random_forest_25_25_reconstruction, False),
        ('F15', random_forest_reconstruction, False),
        ('F23', mlp_128_96_64_reconstruction, False),
        ('F13', mlp_300_reconstruction, False),
        ('F11', mlp_300_reconstruction, False),
        ('F36', ensemble1_reconstruction, False),
        ('F33', KNN_baseline, False),
        ('F18', NB_reconstruction, False),
        ('F25', mlp_300_reconstruction, False),
        ('F5', NB_reconstruction, False),
        ('F50', KNN_baseline, False),
        ('F3', NB_reconstruction, False),
        ('F1', random_forest_25_25_reconstruction, False),
        ('F9', NB_reconstruction, False),
        ('F21', NB_reconstruction, False),
        ('F43', chained_rf_reconstruction, True),
        ('F30', chained_rf_reconstruction, True),
        ('F10', chained_rf_reconstruction, True),
    ],
    "QI2": [
        ('F1', random_forest_25_25_reconstruction, False),
        ('F12', random_forest_25_25_reconstruction, False),
        ('F15', NB_reconstruction, False),
        ('F17', NB_reconstruction, False),
        ('F2', NB_reconstruction, False),
        ('F21', NB_reconstruction, False),
        ('F32', NB_reconstruction, False),
        ('F36', mlp_300_reconstruction, False),
        ('F43', NB_reconstruction, False),
        ('F47', NB_reconstruction, False),
        ('F5', random_forest_25_25_reconstruction, False),
        ('F50', KNN_baseline, False),
        ('F9', NB_reconstruction, False),
        ('F10', chained_rf_reconstruction, True),
        ('F11', chained_rf_reconstruction, True),
        ('F22', chained_rf_reconstruction, True),
        ('F25', chained_rf_reconstruction, True),
        ('F33', chained_nb_reconstruction, True),
    ]
}

data_path = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/NIST_Red-Team_Problems1-24_v2/"

def main():
    # reconstruction25f()
    reconstruction50f()


def reconstruction25f():
    for qi_name, qi_methods in methods.items():
        print("Reconstruction for " + qi_name)
        qi = QIs[qi_name]
        for problem_name in problems[qi_name]:
            deid = pd.read_csv(data_path + problem_name + "_Deid.csv")
            partial_targets = pd.read_csv(data_path + problem_name + "_AttackTargets.csv")
            qi_reconstructed = qi.copy()
            print(".     for " + problem_name)
            for feature, recon_method, is_chain in qi_methods:
                print(".          for " + feature)
                qi_used = qi_reconstructed if is_chain else qi
                partial_targets[feature] = recon_method(deid, partial_targets[qi_used], qi_used, [feature])[0][feature]
                qi_reconstructed.append(feature)
            partial_targets.to_csv(data_path + "leaderboard1_submission2/" + problem_name + "_Reconstruction.csv")


def reconstruction50f():
    for qi_name in QIs.keys():
        print("Reconstruction for " + qi_name)
        qi = QIs[qi_name]
        hidden_features = list(set(features_50).difference(set(qi)))
        for problem_name in problems50f[qi_name]:
            deid = pd.read_csv(data_path + problem_name + "_Deid.csv")
            partial_targets = pd.read_csv(data_path + problem_name + "_AttackTargets.csv")
            print(".     for " + problem_name)
            partial_targets[hidden_features] = chained_rf_reconstruction(deid, partial_targets[qi], qi, hidden_features)[0][hidden_features]
            partial_targets.to_csv(data_path + "leaderboard1_submission2/" + problem_name + "_Reconstruction.csv")



if __name__ == "__main__":
    main()
