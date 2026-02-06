
from pathlib import Path

import numpy as np
import pickle
import pandas as pd




QIs = {"nist_arizona_data": {"QI1": ['F37', 'F41', 'F2', 'F17', 'F22', 'F32', 'F47'],
       "QI2": ['F37', 'F41', 'F3', 'F13', 'F18', 'F23', 'F30']}}
minus_QIs = {"nist_arizona_data": {"QI1": ['F23', 'F13', 'F11', 'F43', 'F36', 'F15', 'F33', 'F25', 'F18', 'F5', 'F30', 'F10', 'F12', 'F50', 'F3', 'F1', 'F9', 'F21'],
             "QI2": ['F11', 'F43', 'F5', 'F36', 'F25', 'F47', 'F32', 'F15', 'F33', 'F17', 'F10', 'F12', 'F2', 'F1', 'F50', 'F22', 'F9', 'F21']}}
features_25 = ['F1', 'F2', 'F3', 'F5', 'F9', 'F10', 'F11', 'F12', 'F13', 'F15', 'F17', 'F18', 'F21', 'F22', 'F23', 'F25', 'F30', 'F32', 'F33', 'F36', 'F37', 'F41', 'F43', 'F47', 'F50']
features_50 = [f'F{i+1}' for i in range(50)]




# CDC Diabetes dataset
# from ucimlrepo import fetch_ucirepo
# cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
# X = cdc_diabetes_health_indicators.data.features
# y = cdc_diabetes_health_indicators.data.targets
# print(cdc_diabetes_health_indicators.metadata)
# print(cdc_diabetes_health_indicators.variables)




def load_data(config):
    data_dir = Path(config["dataset"]["dir"])
    synth = pd.read_csv(data_dir / 'synth.csv')
    train = pd.read_csv(data_dir / 'train.csv')
    qi = QIs[config["dataset"]["name"]][config["QI"]]
    hidden_features = minus_QIs[config["dataset"]["name"]][config["QI"]]

    return train, synth, qi, hidden_features



def get_meta_data_for_diffusion(cfg):
    meta = {"relation_order": [[None, "crc_data"]], "tables": {"crc_data": {"children": [], "parents": []}}}
    domain = {"F1": {"size": 101, "type": "discrete"}, "F2": {"size": 36, "type": "discrete"}, "F3": {"size": 114, "type": "discrete"}, "F5": {"size": 5, "type": "discrete"}, "F9": {"size": 924, "type": "discrete"}, "F10": {"size": 13, "type": "discrete"}, "F11": {"size": 3, "type": "discrete"}, "F12": {"size": 24, "type": "discrete"}, "F13": {"size": 2, "type": "discrete"}, "F15": {"size": 5, "type": "discrete"}, "F17": {"size": 7, "type": "discrete"}, "F18": {"size": 5, "type": "discrete"}, "F21": {"size": 7511, "type": "continuous"}, "F22": {"size": 128, "type": "discrete"}, "F23": {"size": 3, "type": "discrete"}, "F25": {"size": 5, "type": "discrete"}, "F30": {"size": 5, "type": "discrete"}, "F32": {"size": 30, "type": "discrete"}, "F33": {"size": 6, "type": "discrete"}, "F36": {"size": 3, "type": "discrete"}, "F37": {"size": 6, "type": "discrete"}, "F41": {"size": 2, "type": "discrete"}, "F43": {"size": 2, "type": "discrete"}, "F47": {"size": 4, "type": "discrete"}, "F50": {"size": 53, "type": "discrete"}, "F4": {"size": 19, "type": "discrete"}, "F6": {"size": 3, "type": "discrete"}, "F7": {"size": 4, "type": "discrete"}, "F8": {"size": 14, "type": "discrete"}, "F14": {"size": 102, "type": "continuous"}, "F16": {"size": 8, "type": "discrete"}, "F19": {"size": 94, "type": "continuous"}, "F20": {"size": 3, "type": "discrete"}, "F24": {"size": 4, "type": "discrete"}, "F26": {"size": 103, "type": "continuous"}, "F27": {"size": 2, "type": "discrete"}, "F28": {"size": 3, "type": "discrete"}, "F29": {"size": 313, "type": "continuous"}, "F31": {"size": 5, "type": "discrete"}, "F34": {"size": 6, "type": "discrete"}, "F35": {"size": 211, "type": "continuous"}, "F38": {"size": 161, "type": "continuous"}, "F39": {"size": 3, "type": "discrete"}, "F40": {"size": 2, "type": "discrete"}, "F42": {"size": 3, "type": "discrete"}, "F44": {"size": 338, "type": "continuous"}, "F45": {"size": 5, "type": "discrete"}, "F46": {"size": 8, "type": "discrete"}, "F48": {"size": 2, "type": "discrete"}, "F49": {"size": 7, "type": "discrete"}}

    return meta, domain



# for California Housing
def map_to_bin_edges(df, thresholds_dict):
    result = df.copy()
    for col in df.columns:
        thresholds = np.array(thresholds_dict[col])
        # Floor to get bin index (1.0->0, 1.9->0, 2.0->1, etc.)
        bin_indices = np.floor(df[col] - 1).astype(int)
        bin_indices = np.clip(bin_indices, 0, 19)
        result[col] = thresholds[bin_indices]
    return result


def map_with_interpolation(df, thresholds_dict):
    result = df.copy()
    for col in df.columns:
        thresholds = np.array(thresholds_dict[col])
        values = df[col].values
        lower_idx = np.floor(values - 1).astype(int)
        lower_idx = np.clip(lower_idx, 0, 18)  # Max 18 so upper_idx <= 19
        upper_idx = lower_idx + 1
        # Get fractional part (0.0 to 1.0 within the bin)
        frac = (values - 1) - np.floor(values - 1)
        # Linear interpolation
        lower_thresh = thresholds[lower_idx]
        upper_thresh = thresholds[upper_idx]
        result[col] = lower_thresh + frac * (upper_thresh - lower_thresh)
    return result


