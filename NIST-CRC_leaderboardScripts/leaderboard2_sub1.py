import pandas as pd

from attacks.baselines_classifiers import KNN_baseline, simply_measure_deid_itself_baseline
from attacks.ML_classifiers import NB_reconstruction
from attacks.NN_classifier import mlp_300_reconstruction
from util import minus_QIs, QIs, features_50

problems = [
    ("15_ARF_25f_QID1", "QI1", 25, ),
    ("11_AIM_e1_25f_QID1", "QI1", 25, ),
    ("13_AIM_e10_25f_QID1", "QI1", 25, ),
    ("17_TVAE_25f_QID1", "QI1", 25, ),
    ("19_CELL_SUPPRESSION_25f_QID1", "QI1", 25, ),
    ("1_SYNTHPOP_25f_QID1", "QI1", 25, ),
    ("23_RANKSWAP_25f_QID1", "QI1", 25, simply_measure_deid_itself_baseline),
    ("5_MST_e1_25f_QID1", "QI1", 25, ),
    ("7_MST_e10_25f_QID1", "QI1", 25, ),
    ("21_CELL_SUPPRESSION_50f_QID1", "QI1", 50, ),
    ("3_SYNTHPOP_50f_QID1", "QI1", 50, ),
    ("9_MST_e10_50f_QID1", "QI1", 50, ),

    ("12_AIM_e1_25f_QID2", "QI2", 25, ),
    ("14_AIM_e10_25f_QID2", "QI2", 25, ),
    ("16_ARF_25f_QID2", "QI2", 25, ),
    ("18_TVAE_25f_QID2", "QI2", 25, ),
    ("20_CELL_SUPPRESSION_25f_QID2", "QI2", 25, ),
    ("24_RANKSWAP_25f_QID2", "QI2", 25, simply_measure_deid_itself_baseline),
    ("2_SYNTHPOP_25f_QID2", "QI2", 25, ),
    ("6_MST_e1_25f_QID2", "QI2", 25, ),
    ("8_MST_e10_25f_QID2", "QI2", 25, ),
    ("22_CELL_SUPPRESSION_50f_QID2", "QI2", 50, ),
    ("4_SYNTHPOP_50f_QID2", "QI2", 50, ),
    ("10_MST_e10_50f_QID2", "QI2", 50, ),
]

data_path = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/"

def main():
    reconstruction()

def reconstruction():
    for problem, qi_name, num_features, recon_method in problems:
        print("Processing", problem)
        qi = QIs[qi_name]
        # TODO: sort 50 hidden_features
        hidden_features = minus_QIs[qi_name] if num_features == 25 else list(set(features_50).difference(set(qi)))

        deid = pd.read_csv(data_path + "NIST_Red-Team_Problems1-24_v2/" + problem + "_Deid.csv")
        partial_targets = pd.read_csv(data_path + "NIST_Red-Team_Problems1-24_v2/" + problem + "_AttackTargets.csv")
        target_index = partial_targets["TargetID"].values
        partial_targets = partial_targets[qi]

        reconstruction, _, _ = recon_method(deid, partial_targets, qi, hidden_features)

        reconstruction["TargetID"] = target_index
        reconstruction.to_csv(data_path + "leaderboard2_submission1/" + problem + "_Reconstruction.csv")
        reconstruction.to_parquet(data_path + "leaderboard2_submission1/" + problem + "_Reconstruction.parquet")



        print()


if __name__ == "__main__":
    main()
