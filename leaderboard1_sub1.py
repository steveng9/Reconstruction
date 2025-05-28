import pandas as pd

from baselines import KNN_baseline
from recon_ML_classifiers import chained_rf_reconstruction, chained_nb_reconstruction, NB_reconstruction
from recon_NN_classifier import mlp_300_reconstruction
from util import minus_QIs, QIs, features_50

problems = [
    ("15_ARF_25f_QID1", "QI1", 25, chained_rf_reconstruction),
    ("11_AIM_e1_25f_QID1", "QI1", 25, mlp_300_reconstruction),
    ("13_AIM_e10_25f_QID1", "QI1", 25, mlp_300_reconstruction),
    ("17_TVAE_25f_QID1", "QI1", 25, mlp_300_reconstruction),
    ("19_CELL_SUPPRESSION_25f_QID1", "QI1", 25, chained_rf_reconstruction),
    ("1_SYNTHPOP_25f_QID1", "QI1", 25, chained_rf_reconstruction),
    ("23_RANKSWAP_25f_QID1", "QI1", 25, chained_rf_reconstruction),
    ("5_MST_e1_25f_QID1", "QI1", 25, mlp_300_reconstruction),
    ("7_MST_e10_25f_QID1", "QI1", 25, mlp_300_reconstruction),
    ("21_CELL_SUPPRESSION_50f_QID1", "QI1", 50, chained_rf_reconstruction),
    ("3_SYNTHPOP_50f_QID1", "QI1", 50, chained_rf_reconstruction),
    ("9_MST_e10_50f_QID1", "QI1", 50, mlp_300_reconstruction),

    ("12_AIM_e1_25f_QID2", "QI2", 25, NB_reconstruction),
    ("14_AIM_e10_25f_QID2", "QI2", 25, NB_reconstruction),
    ("16_ARF_25f_QID2", "QI2", 25, NB_reconstruction),
    ("18_TVAE_25f_QID2", "QI2", 25, NB_reconstruction),
    ("20_CELL_SUPPRESSION_25f_QID2", "QI2", 25, NB_reconstruction),
    ("24_RANKSWAP_25f_QID2", "QI2", 25, KNN_baseline),
    ("2_SYNTHPOP_25f_QID2", "QI2", 25, NB_reconstruction),
    ("6_MST_e1_25f_QID2", "QI2", 25, chained_nb_reconstruction),
    ("8_MST_e10_25f_QID2", "QI2", 25, chained_nb_reconstruction),
    ("22_CELL_SUPPRESSION_50f_QID2", "QI2", 50, NB_reconstruction),
    ("4_SYNTHPOP_50f_QID2", "QI2", 50, NB_reconstruction),
    ("10_MST_e10_50f_QID2", "QI2", 50, chained_nb_reconstruction),
]

data_path = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/NIST_Red-Team_Problems1-24_v2/"

def main():
    # reconstruction()
    save_as_parquet()

def reconstruction():
    for problem, qi_name, num_features, recon_method in problems:
        print("Processing", problem)
        qi = QIs[qi_name]
        # TODO: sort 50 hidden_features
        hidden_features = minus_QIs[qi_name] if num_features == 25 else list(set(features_50).difference(set(qi)))

        deid = pd.read_csv(data_path + problem + "_Deid.csv")
        partial_targets = pd.read_csv(data_path + problem + "_AttackTargets.csv")
        target_index = partial_targets["TargetID"].values
        partial_targets = partial_targets[qi]

        reconstruction, _, _ = recon_method(deid, partial_targets, qi, hidden_features)

        reconstruction["TargetID"] = target_index
        reconstruction.to_csv(data_path + "leaderboard1_submission1/" + problem + "_Reconstruction.csv")



        print()

def save_as_parquet():
    for problem, _, _, _ in problems:
        recon_csv = pd.read_csv(data_path + "leaderboard1_submission1/" + problem + "_Reconstruction.csv")
        recon_csv.to_parquet(data_path + "leaderboard1_submission1/" + problem + "_Reconstruction.parquet")


if __name__ == "__main__":
    main()
