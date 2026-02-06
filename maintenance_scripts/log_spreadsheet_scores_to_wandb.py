
import pandas as pd
import numpy as np
import yaml
import wandb

from util import QIs, minus_QIs


notes = [
    ("MST", "tried with training MST with eps = 1000, delta = 1, the exact marginals MST chooses, as well as highly-specific 5-way, 6-way, 7-way and 8-way marginals that include all or most of the 7 existing features to reduce randomness of the. MST sampling step). "),
    ("MST", "w/RankSwap-as-AUX Normalized 5-way marginals. . only consider normalized probas from most common 10% of values "),
    ("MST", "w/RankSwap-as-AUX Normalized 4-way marginals, only consider normalized probas from most common 10% of values "),
    ("MST", "w/RankSwap-as-AUX Normalized naturally-chosen marginals, only consider normalized probas from most common 10% of values "),
    ("MST", "w/AUX Normalized naturally-chosen + given values marginals. choose value with max normalized proba"),
    ("MST", "w/AUX Normalized 4-way marginals. . only consider normalized probas from most common 10% of values "),
    ("MST", "w/AUX Normalized 4-way marginals. choose value with max normalized proba"),
    ("MST", "w/AUX Normalized 6-way marginals. choose value with max normalized proba"),
    ("MST", "w/AUX Normalized 4-way marginals. only consider normalized probas from most common 10% of values "),
    ("MST", "w/AUX Normalized naturally-chosen marginals, . choose value with max normalized proba"),
    ("MST", "w/AUX Normalized naturally-chosen marginals, only consider normalized probas from most common 10% of values "),
]
def main():

    # sdgs = [
    #     "AIM",
    #     "TVAE",
    #     "CellSupression",
    #     "Synthpop",
    #     "ARF",
    #     "RANKSWAP",
    #     "MST"
    # ]
    scores = pd.read_csv('../../../Library/Application Support/JetBrains/PyCharm2025.2/scratches/scores_temp_file.txt', sep='\t', header=None)
    with open('cfg_for_logging_spreadsheet_scores_to_wandb.yaml', 'r') as f:
        config = yaml.safe_load(f)


    for i, (sdg_method, note) in enumerate(notes):

        eps = None
        if sdg_method == "AIM":
            eps = 1
        elif sdg_method == "MST":
            eps = 10
        config["eps"] = eps

        sdg_scores = scores.iloc[i, :]
        config["sdg_method"] = sdg_method

        wandb.init(
            project=config['wandb']['project'],
            name=f"{config['wandb']['name']}",
            config=config,
            group=config['wandb'].get('group', None),
            tags=config['wandb'].get('tags', []),
            notes=note,
        )

        # # for QI1
        if config["QI"] == "QI1":
            hidden_features = [
                "F1", "F10", "F11", "F12", "F13", "F15", "F18", "F21", "F23",
                "F25", "F3", "F30", "F33", "F36", "F43", "F5", "F50", "F9"
            ]
        elif config["QI"] == "QI2":
            hidden_features = [
                "F1", "F10", "F11", "F12", "F15", "F17", "F2", "F21", "F22",
                "F25", "F32", "F33", "F36", "F43", "F47", "F5", "F50", "F9"
            ]

        results = {f"RA_{k}": v for k, v in zip(hidden_features, sdg_scores)}
        results["RA_mean"] = np.mean(sdg_scores)
        wandb.log(results)
        wandb.finish()




if __name__ == "__main__":
    main()


