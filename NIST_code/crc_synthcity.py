"""
crc_synthcity.py

Uses Synthcity's ARF library to generate a deidentified SBO dataset.

NOTE: Before running this code, be sure to modify constant variables and ensure all paths are correct.
"""
import json
import os
import multiprocessing as mp
from pathlib import Path

import pandas as pd

from synthcity.plugins import Plugins
from synthcity.plugins.core.constraints import Constraints


# MAIN_DIRECTORY = "C:/Users/DamonStreat/Development/2025 NIST CRC Synthetic Data Gen SDNist/"
MAIN_DIRECTORY = "C:/Users/DamonStreat/Development/kmetrics"
DATA_DIR = Path(MAIN_DIRECTORY, 'data')
MATCH_3_DATA_DIR = Path(MAIN_DIRECTORY, 'Match 3/Arizona/DATA_STRATA')
ACS_PATH = Path(MATCH_3_DATA_DIR, '16_ARF_25f_QID2_GT.csv')

# Feature categories 
match_3_data_25f_cols ={
    'continuous': ['AGE', 'DURUNEMP', 'FAMSIZE', 'HRSWORK1', 'INCWAGE', 'RENT', 'VALUEH', 'WKSWORK1'],
    'categorical': ['AGEMARR','BPL', 'CHBORN', 'CITIZEN', 'CITY', 'COUNTY', 'CLASSWKR', 'EMPSTAT', 'FARM', 'FBPL', 'GQ', 'GQFUNDS', 'GQTYPE', 'HISPAN', 'INCNONWG', 'IND', 'LABFORCE', 'MARRNO', 'MARST', 'MBPL', 'METAREA', 'METRO', 'MIGCITY5', 'MIGRATE5', 'MIGTYPE5', 'MTONGUE', 'NATIVITY', 'NCHLT5', 'OCC', 'OWNERSHP', 'RACE', 'SAMEPLAC', 'SCHOOL', 'SEX', 'SSENROLL', 'URBAN', 'VETCHILD', 'VETPER', 'VETSTAT', 'VETWWI', 'WARD'],
    'ordinal': ['EDUC']
}

QID_1_FEATURES = [
    'RACE', 'SEX', 'AGEMARR', 'GQTYPE', 'IND', 'MTONGUE', 'VETSTAT'
]
QID_2_FEATURES = [
    'RACE', 'SEX', 'BPL', 'FARM', 'HISPAN', 'LABFORCE', 'MIGRATE5'
]

def create(number, out_dir, target_name, target_df, fset, fset_name, algo_name, n_iter, epsilon):
    
    print("AGEMARR", target_df.AGEMARR.value_counts())
    print("BPL", target_df.BPL.value_counts())
    if epsilon is not None and n_iter is None:
        model = Plugins().get(algo_name, epsilon=epsilon)
    elif epsilon is not None:
        model = Plugins().get(algo_name, n_iter=n_iter, epsilon=epsilon)
    elif n_iter is not None:
        model = Plugins().get(algo_name, n_iter=n_iter)
    else:
        model = Plugins().get(algo_name)

    encoder_mapping = {} # Unsued if no mixed types
    if mixed_dtypes := {c: dtype for c in target_df.columns if (dtype := pd.api.types.infer_dtype(target_df[c])).startswith("mixed")}:
        for c in mixed_dtypes.keys():
            encoder_mapping[c] = {v: i for i, v in enumerate(target_df[c].unique())}
            target_df = target_df.apply(lambda x: x.map(encoder_mapping[c]) if x.name == c else x)

    for c in target_df.columns:
        if c in match_3_data_25f_cols["categorical"]:
            target_df[c] = target_df[c].astype(str)
    target_df["BPL"] = target_df["BPL"].apply(lambda x: f"A{x}")


    print("AGEMARR", target_df.AGEMARR.value_counts())
    print("BPL", target_df.BPL.value_counts())


    model.fit(target_df)

    syn_data = model.generate(count=target_df.shape[0]).dataframe()

    print()
    print(number, 'Done')
    if epsilon is not None:
        out_file = algo_name + '_n_iter_' + str(n_iter) + '_e_' + str(epsilon) + '_' + fset_name + '.csv'
    else:
        out_file = algo_name + '_' + fset_name + '.csv'
    out_path = Path(out_dir, out_file)
    print(number, 'Created Path')
    if n_iter is not None:
        labels = {
            "labels": {
                "algorithm name": algo_name,
                "library": "synthcity",
                "feature set": "all-features" if fset is None else f'{fset_name}-focused',
                "target dataset": target_name,
                "variant label": f"n_iter={n_iter}, target column",
                "variant label detail": "added a column target to the original dataset and target column is set to dataset index"
            }
        }
    else:
        labels = {
            "labels": {
                "algorithm name": algo_name,
                "library": "synthcity",
                "feature set": "all-features" if fset is None else f'{fset_name}-focused',
                "target dataset": target_name,
                "variant label": f"target column",
                "variant label detail": "added a column target to the original dataset and target column is set to dataset index"
            }
        }
    print(number, 'Created Labels')
    if epsilon:
        labels['labels']['epsilon'] = str(epsilon)
        
    # Remove A's in BPL
    syn_data['BPL'] = syn_data['BPL'].str.replace('A', '')
    
    print("SYN AGEMARR", syn_data.AGEMARR.value_counts())
    print("SYN BPL", syn_data.BPL.value_counts())
    temp_outfile = Path(out_dir, f"16_ARF_25f_QID2_Deid.csv")
    syn_data.to_csv(temp_outfile, index=False)
    print(number, f'Finished: {out_path}')


if __name__ == '__main__':
    print(Plugins().list())
    cpu = 1
    # THIS_DIR = Path(__file__).parent

    all_match_3_cols = [col for col_type in match_3_data_25f_cols.values() for col in col_type]

    acs_all_feats = [
        "PUMA", "HISP", "NOC", "NPF", "DENSITY", "INDP", "INDP_CAT", "PINCP",
        "POVPIP", "DREM", "DPHY", "DEAR","PWGTP","WGTP", "SEX", "MSP",
        "RAC1P", "OWN_RENT", "PINCP_DECILE", "EDU", "AGEP", "HOUSING_TYPE", "DVET", "DEYE"
    ]
    
    sbo_all_feats = [
        "FIPST", "SECTOR", "EMPLOYMENT_NOISY", "PAYROLL_NOISY", "RECEIPTS_NOISY",
        "PCT1", "ETH1", "RACE1", "SEX1", "VET1","FOUNDED1", "PURCHASED1", "INHERITED1",
        "RECEIVED1", "ACQUIRENR1", "ACQYR1", "PROVIDE1", "MANAGE1", "FINANCIAL1",
        "FNCTNABV1", "FNCTNR1", "HOURS1", "PRMINC1", "SELFEMP1", "EDUC1", "AGE1",
        "BORNUS1", "DISVET1", "ESTABLISHED", "SCSAVINGS", "SCASSETS", "SCEQUITY",
        "SCCREDIT", "SCGOVTLOAN", "SCGOVTGUAR", "SCBANKLOAN", "SCFAMLOAN", "SCVENTURE",
        "SCGRANT", "SCOTHER", "SCDONTKNOW", "SCNONENEEDED", "SCNOTREPORTED", "SCAMOUNT",
        "HOMEBASED", "FRANCHISE", "FRANCHISER50", "ECSAVINGS", "ECASSETS", "ECEQUITY",
        "ECCREDIT", "ECGOVTLOAN", "ECGOVTGUAR", "ECBANKLOAN", "ECFAMLOAN", "ECVENTURE",
        "ECPROFITS", "ECGRANT", "ECOTHER", "ECDONTKNOW", "ECNOACCESS", "ECNOEXPAND", 
        "ECNOTREPORTED", "FEDERAL", "STATELOCAL", "OTHERBUS", "INDIVIDUALS", "CUSTNR",
        "EXPORTS", "OPSOUTSIDE", "OUTSOURCE", "ENGLISH", "ARABIC", "CHINESE", "FRENCH",
        "GERMAN", "GREEK", "HINDI", "ITALIAN", "JAPANESE", "KOREAN", "POLISH", "PORTUGUESE", 
        "RUSSIAN", "SPANISH", "TAGALOG", "VIETNAMESE", "LANGOTHER", "LANGNR", "FULLTIME",
        "PARTTIME", "DAYLABOR", "TEMPSTAFF", "LEASED", "CONTRACTORS", "EMPNR", "HEALTHINS",
        "RETIREMENT", "PROFITSHARE", "HOLIDAYS", "BENENABV", "BENENR", "WEBSITE",
        "ECOMMERCE", "ECOMMPCT", "ONLINEPURCH", "LT40HOURS", "LT12MONTHS", "SEASONAL",
        "OCCASIONALLY", "ACTIVITYNABV", "ACTIVITYNR", "OPERATING", "RETIRED", "DECEASED",
        "ONETIME", "LOWSALES", "NOBUSCRED", "NOPERSCRED", "STARTANOTHER", "SOLDBUS",
        "CEASEOTHER", "CEASENR", "CEASENA", "HUSBWIFE", "FAMILYBUS", "NUMOWNERS"
    ]

    acs_df = pd.read_csv(ACS_PATH)
    
    # For DATA STRATA GEN
    acs_df.set_index("ID", inplace=True) # set index equal to the id
    acs_df['target'] = acs_df.index.tolist()
    target_datasets = [('acs', acs_df.copy())]
    acs_fsets = [(None, 'all')]
    algos = [('arf', None, False)]
    epsilons = [1]

    # create runs set
    runs = []
    count = 0
    for target_name, target_df in target_datasets:
        for fset, fset_name in acs_fsets:
            features = []
            if fset:
                features = fset + ['target']
            else:
                features = target_df.columns.tolist()
            f_target_df = target_df[features].copy()

            for algo_name, n_iter, has_epsilon in algos:
                # create output directory
                # path_str = os.path.join(MAIN_DIRECTORY, "Match 3", 'Arizona', 'synthcity', 'syn_data', algo_name, target_name)
                path_str = os.path.join(MAIN_DIRECTORY, "Match 3", 'Arizona', 'DATA_STRATA', 'DEID_DATA')
                OUT_DIR = Path(path_str)
                print(path_str)
                if not OUT_DIR.exists():
                    OUT_DIR.mkdir(parents=True)
                if has_epsilon:
                    for epsilon in epsilons:
                        print(f'COUNT: {count}', target_name, fset_name, algo_name, n_iter, epsilon)

                        runs.append((count, OUT_DIR, target_name, f_target_df, fset, fset_name, algo_name, n_iter, epsilon))
                        count += 1
                else:
                    print(f'COUNT: {count}', target_name, fset_name, algo_name, n_iter)

                    runs.append((count, OUT_DIR, target_name, f_target_df, fset, fset_name, algo_name, n_iter, None))
                    count += 1

    
    for run in runs:
        create(*run)
    # pool = mp.Pool(cpu)
    # pool.starmap(create, runs)

