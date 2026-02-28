
from pathlib import Path

import numpy as np
import pickle
import pandas as pd


def _sdg_dirname(method, params=None):
    """Local copy of sdg.sdg_dirname — avoids importing the full sdg package in recon_ env."""
    params = params or {}
    eps = params.get("epsilon") or params.get("eps")
    if eps is not None:
        return f"{method}_eps{eps:g}"
    return method




QIs = {
    # Adult (census income): demographic background knowledge as QI
    # Story: given public demographic info, reconstruct employment & income
    "adult": {
        "QI1": ["age", "sex", "race", "native-country", "education", "marital-status"],
        # All features except target income — for single-feature reconstruction comparison
        "QI_linear": ["age", "workclass", "fnlwgt", "education", "education-num",
                      "marital-status", "occupation", "relationship", "race", "sex",
                      "capital-gain", "capital-loss", "hours-per-week", "native-country"],
    },
    "cdc_diabetes": {
        # All features except target Diabetes_binary — for single-feature reconstruction comparison
        "QI_linear": ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
                      "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                      "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
                      "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"],
    },
    # California Housing: geographic + structural features as QI
    # Story: given publicly observable location/age/population, reconstruct economic features
    "california": {
        "QI1": ["Latitude", "Longitude", "HouseAge", "Population"],
    },
    # nist_arizona_25feat: 25-col IPUMS subset (mirrors NIST CRC competition, F-code encoding)
    # nist_arizona_50feat: 50-col subset
    # nist_arizona_data:   98-col full data (QI2 uses SUPDIST which is not in 25/50 subsets)
    # QI1 known (7) is shared across all three; hidden features differ by available columns.
    "nist_arizona_25feat": {
        "QI1": ['RACE', 'SEX', 'AGEMARR', 'GQTYPE', 'IND', 'MTONGUE', 'VETSTAT'],
    },
    "nist_arizona_50feat": {
        "QI1": ['RACE', 'SEX', 'AGEMARR', 'GQTYPE', 'IND', 'MTONGUE', 'VETSTAT'],
    },
    "nist_arizona_data": {
        "QI1": ['RACE', 'SEX', 'AGEMARR', 'GQTYPE', 'IND', 'MTONGUE', 'VETSTAT'],
        "QI2": ['SEX', 'AGE', 'RACE', 'MARST', 'NATIVITY', 'HISPAN', 'BPL',
                'COUNTY', 'SUPDIST', 'EMPSTAT', 'LABFORCE', 'EDUC'],
    },
    "california_housing_data": {
        "QI1": ['0', '1', '2', '3', '4', '5'],
    }
}
minus_QIs = {
    "adult": {
        "QI1":      ["workclass", "fnlwgt", "education-num", "occupation", "relationship",
                     "capital-gain", "capital-loss", "hours-per-week", "income"],
        "QI_linear": ["income"],
    },
    "cdc_diabetes": {
        "QI_linear": ["Diabetes_binary"],
    },
    "california": {
        "QI1": ["MedInc", "AveRooms", "AveBedrms", "AveOccup", "MedHouseVal"],
    },
    # 25-feat: all 18 non-QI features (mirrors NIST CRC competition format)
    "nist_arizona_25feat": {
        "QI1": ['AGE', 'BPL', 'CITIZEN', 'DURUNEMP', 'EDUC', 'EMPSTAT', 'FAMSIZE',
                'FARM', 'GQ', 'HISPAN', 'INCWAGE', 'LABFORCE', 'MARST', 'MIGRATE5',
                'NATIVITY', 'OWNERSHP', 'URBAN', 'WKSWORK1'],
    },
    # 50-feat: focused economic/labor/education outcomes
    "nist_arizona_50feat": {
        "QI1": ['INCWAGE', 'VALUEH', 'RENT', 'OCC', 'WKSWORK1',
                'HRSWORK1', 'CLASSWKR', 'EDUC'],
    },
    # 98-feat full data
    # QI1 hidden (8): economic + labor + education outcomes
    # QI2 hidden (10): economic + labor outcomes (EDUC already in QI2 known features)
    "nist_arizona_data": {
        "QI1": ['INCWAGE', 'VALUEH', 'RENT', 'OCC', 'WKSWORK1',
                'HRSWORK1', 'CLASSWKR', 'EDUC'],
        "QI2": ['INCWAGE', 'VALUEH', 'RENT', 'OCC', 'IND', 'WKSWORK1',
                'HRSWORK1', 'SEI', 'OCCSCORE', 'CLASSWKR'],
    },
    "california_housing_data": {
        "QI1": ['6', '7', '8'],
    }
}
# features_25 = ['F1', 'F2', 'F3', 'F5', 'F9', 'F10', 'F11', 'F12', 'F13', 'F15', 'F17', 'F18', 'F21', 'F22', 'F23', 'F25', 'F30', 'F32', 'F33', 'F36', 'F37', 'F41', 'F43', 'F47', 'F50']
# features_50 = [f'F{i+1}' for i in range(50)]




# CDC Diabetes dataset
# from ucimlrepo import fetch_ucirepo
# cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
# X = cdc_diabetes_health_indicators.data.features
# y = cdc_diabetes_health_indicators.data.targets
# print(cdc_diabetes_health_indicators.metadata)
# print(cdc_diabetes_health_indicators.variables)




def load_data(config):
    data_dir = Path(config["dataset"]["dir"])
    train = pd.read_csv(data_dir / 'train.csv')

    # synth.csv lives in a subdirectory derived from sdg_method + sdg_params
    sdg_method = config.get("sdg_method")
    if sdg_method:
        dirname = _sdg_dirname(sdg_method, config.get("sdg_params", {}))
        synth = pd.read_csv(data_dir / dirname / 'synth.csv')
    else:
        synth = pd.read_csv(data_dir / 'synth.csv')

    qi = QIs[config["dataset"]["name"]][config["QI"]]
    hidden_features = minus_QIs[config["dataset"]["name"]][config["QI"]]

    # Memorization test: load holdout from a separate directory (e.g. a different disjoint sample)
    mem_cfg = config.get("memorization_test", {})
    holdout = None
    if mem_cfg.get("enabled", False) and mem_cfg.get("holdout_dir"):
        if (data_dir / "NO_HOLDOUT").exists():
            raise ValueError(
                f"Train dir {data_dir} is marked NO_HOLDOUT — this training sample was "
                f"drawn non-disjointly and may overlap any holdout set."
            )
        holdout_dir = Path(mem_cfg["holdout_dir"])
        if (holdout_dir / "NO_HOLDOUT").exists():
            raise ValueError(
                f"Holdout dir {holdout_dir} is marked NO_HOLDOUT — this sample "
                f"overlaps with other training samples and cannot be used as holdout."
            )
        holdout = pd.read_csv(holdout_dir / 'train.csv')

    return train, synth, qi, hidden_features, holdout



def load_mia_data(config):
    """Load all data needed for MIA: train, synth, holdout, and meta.json."""
    import json
    data_dir = Path(config["dataset"]["dir"])

    train = pd.read_csv(data_dir / "train.csv")

    sdg_method = config.get("sdg_method")
    if sdg_method:
        dirname = _sdg_dirname(sdg_method, config.get("sdg_params", {}))
        synth = pd.read_csv(data_dir / dirname / "synth.csv")
    else:
        synth = pd.read_csv(data_dir / "synth.csv")

    # Holdout is required for MIA (non-member targets)
    holdout_dir = config.get("memorization_test", {}).get("holdout_dir")
    if not holdout_dir:
        raise ValueError(
            "MIA mode requires memorization_test.holdout_dir in the config "
            "(path to a disjoint sample directory to use as non-members)."
        )
    if (data_dir / "NO_HOLDOUT").exists():
        raise ValueError(
            f"Train dir {data_dir} is marked NO_HOLDOUT — this training sample was "
            f"drawn non-disjointly and may overlap any holdout set."
        )
    holdout_path = Path(holdout_dir)
    if (holdout_path / "NO_HOLDOUT").exists():
        raise ValueError(
            f"holdout_dir '{holdout_dir}' is marked NO_HOLDOUT — this sample "
            "overlaps with other training samples and cannot be used as MIA holdout."
        )
    holdout = pd.read_csv(holdout_path / "train.csv")

    # meta.json lives two levels above the sample dir: .../dataset_name/meta.json
    meta_path = data_dir.parent.parent / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found at {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)

    return train, synth, holdout, meta


def get_meta_data_for_diffusion(cfg):
    if cfg["dataset"]["name"] == "adult":
        meta = {"relation_order": [[None, "adult_data"]], "tables": {"adult_data": {"children": [], "parents": []}}}
        domain = {
            "age":             {"size": 70,   "type": "continuous"},
            "workclass":       {"size": 8,    "type": "discrete"},
            "fnlwgt":          {"size": 8477, "type": "continuous"},
            "education":       {"size": 16,   "type": "discrete"},
            "education-num":   {"size": 16,   "type": "continuous"},
            "marital-status":  {"size": 7,    "type": "discrete"},
            "occupation":      {"size": 15,   "type": "discrete"},
            "relationship":    {"size": 6,    "type": "discrete"},
            "race":            {"size": 5,    "type": "discrete"},
            "sex":             {"size": 2,    "type": "discrete"},
            "capital-gain":    {"size": 98,   "type": "continuous"},
            "capital-loss":    {"size": 69,   "type": "continuous"},
            "hours-per-week":  {"size": 86,   "type": "continuous"},
            "native-country":  {"size": 42,   "type": "discrete"},
            "income":          {"size": 2,    "type": "discrete"},
        }
    elif cfg["dataset"]["name"] == "nist_arizona_data":
        meta = {"relation_order": [[None, "crc_data"]], "tables": {"crc_data": {"children": [], "parents": []}}}
        domain = {"F1": {"size": 101, "type": "discrete"}, "F2": {"size": 36, "type": "discrete"}, "F3": {"size": 114, "type": "discrete"}, "F5": {"size": 5, "type": "discrete"}, "F9": {"size": 924, "type": "discrete"}, "F10": {"size": 13, "type": "discrete"}, "F11": {"size": 3, "type": "discrete"}, "F12": {"size": 24, "type": "discrete"}, "F13": {"size": 2, "type": "discrete"}, "F15": {"size": 5, "type": "discrete"}, "F17": {"size": 7, "type": "discrete"}, "F18": {"size": 5, "type": "discrete"}, "F21": {"size": 7511, "type": "continuous"}, "F22": {"size": 128, "type": "discrete"}, "F23": {"size": 3, "type": "discrete"}, "F25": {"size": 5, "type": "discrete"}, "F30": {"size": 5, "type": "discrete"}, "F32": {"size": 30, "type": "discrete"}, "F33": {"size": 6, "type": "discrete"}, "F36": {"size": 3, "type": "discrete"}, "F37": {"size": 6, "type": "discrete"}, "F41": {"size": 2, "type": "discrete"}, "F43": {"size": 2, "type": "discrete"}, "F47": {"size": 4, "type": "discrete"}, "F50": {"size": 53, "type": "discrete"}, "F4": {"size": 19, "type": "discrete"}, "F6": {"size": 3, "type": "discrete"}, "F7": {"size": 4, "type": "discrete"}, "F8": {"size": 14, "type": "discrete"}, "F14": {"size": 102, "type": "continuous"}, "F16": {"size": 8, "type": "discrete"}, "F19": {"size": 94, "type": "continuous"}, "F20": {"size": 3, "type": "discrete"}, "F24": {"size": 4, "type": "discrete"}, "F26": {"size": 103, "type": "continuous"}, "F27": {"size": 2, "type": "discrete"}, "F28": {"size": 3, "type": "discrete"}, "F29": {"size": 313, "type": "continuous"}, "F31": {"size": 5, "type": "discrete"}, "F34": {"size": 6, "type": "discrete"}, "F35": {"size": 211, "type": "continuous"}, "F38": {"size": 161, "type": "continuous"}, "F39": {"size": 3, "type": "discrete"}, "F40": {"size": 2, "type": "discrete"}, "F42": {"size": 3, "type": "discrete"}, "F44": {"size": 338, "type": "continuous"}, "F45": {"size": 5, "type": "discrete"}, "F46": {"size": 8, "type": "discrete"}, "F48": {"size": 2, "type": "discrete"}, "F49": {"size": 7, "type": "discrete"}}
    elif cfg["dataset"]["name"] in ("california", "california_housing_data"):
        meta = {"relation_order": [[None, "cali_data"]], "tables": {"cali_data": {"children": [], "parents": []}}}
        domain = {
            "MedInc":      {"size": 937,  "type": "continuous"},
            "HouseAge":    {"size": 51,   "type": "continuous"},
            "AveRooms":    {"size": 998,  "type": "continuous"},
            "AveBedrms":   {"size": 967,  "type": "continuous"},
            "Population":  {"size": 817,  "type": "continuous"},
            "AveOccup":    {"size": 994,  "type": "continuous"},
            "Latitude":    {"size": 370,  "type": "continuous"},
            "Longitude":   {"size": 404,  "type": "continuous"},
            "MedHouseVal": {"size": 795,  "type": "continuous"},
        }

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


