import numpy as np
import pandas as pd
import pickle

QIs = {"QI1": ['F37', 'F41', 'F2', 'F17', 'F22', 'F32', 'F47'],
       "QI2": ['F37', 'F41', 'F3', 'F13', 'F18', 'F23', 'F30']}
minus_QIs = {"QI1": ['F23', 'F13', 'F11', 'F43', 'F36', 'F15', 'F33', 'F25', 'F18', 'F5', 'F30', 'F10', 'F12', 'F50', 'F3', 'F1', 'F9', 'F21'],
             "QI2": ['F11', 'F43', 'F5', 'F36', 'F25', 'F47', 'F32', 'F15', 'F33', 'F17', 'F10', 'F12', 'F2', 'F1', 'F50', 'F22', 'F9', 'F21']}
features_25 = ['F1', 'F2', 'F3', 'F5', 'F9', 'F10', 'F11', 'F12', 'F13', 'F15', 'F17', 'F18', 'F21', 'F22', 'F23', 'F25', 'F30', 'F32', 'F33', 'F36', 'F37', 'F41', 'F43', 'F47', 'F50']
features_50 = [f'F{i+1}' for i in range(50)]

def calculate_reconstruction_score(df_original, df_reconstructed, hidden_features):
    total_records = len(df_original)

    scores = []
    for col in hidden_features:
        value_counts = df_original[col].value_counts()
        rarity_scores = df_original[col].map(total_records / value_counts)
        max_score = rarity_scores.sum()

        score = ( (df_original[col].values == df_reconstructed[col].values) * rarity_scores ).sum()
        scores.append(round(score / max_score * 100, 1))
    return scores


def simple_accuracy_score(df_original, df_reconstructed, hidden_features):
    total_records = len(df_original)

    scores = []
    for col in hidden_features:
        score = (df_original[col].values == df_reconstructed[col].values).sum()
        scores.append(round(score / total_records * 100, 1))
    return scores



def run_test():

    for qi_name, qi in QIs.items():
        hidden_features = list(set(features_25).difference(set(qi)))
        print(hidden_features)


def match_aux_and_synth_classes(aux, deid):
    # fix aux data to only have values from synth data
    # mask = pd.DataFrame(index=aux.index, columns=aux.columns)
    for col in aux.columns:

        mask = aux[col].isin(deid[col])
        aux_count = (~mask).sum()

        # Replace non-matching values
        # aux[col] = aux[col].where(mask[col], deid[col].sample(n=mask[col].sum(), replace=True, ignore_index=True))

        mask_synth = deid[col].isin(aux[col])
        synth_count = (~mask_synth).sum()

        if synth_count > 0:
            # replace with value from synth NOT in aux

            # sample from deid
            # bad_indexes = aux[~mask].index
            # sampled_values = deid[~mask_synth].sample(n=len(bad_indexes), replace=True)[col]
            # aux.loc[~mask, col] = sampled_values.values

            # or go in round
            num_full_rounds = aux_count // synth_count
            remainder = aux_count % synth_count
            values_not_in_aux = deid[~mask_synth][col].values
            aux.loc[~mask, col] = np.hstack([values_not_in_aux for _ in range(num_full_rounds)] + [values_not_in_aux[:remainder]]).astype(int)
        else:
            # replace with mode of synth
            aux[col] = aux[col].where(mask, deid[col].mode()[0])

        # finally, replace rows in synth that still aren't in aux
        updated_mask_synth = deid[col].isin(aux[col])
        synth_count = (~updated_mask_synth).sum()
        if mask.sum() > 0:
            deid[col] = deid[col].where(updated_mask_synth, aux.loc[mask][col].mode()[0])
        else:
            deid[col] = deid[col].where(updated_mask_synth, aux[col].mode()[0])

        print(f"num vals replaced for {col}: aux: {aux_count}, synth: {synth_count}")

    return aux, deid

def numerical_bin_value(x: int):
    """Returns the first digit with the remaining digits set to 0"""
    if x == 0:
        return 0  # Base Case
    if x < 10:
        return x
    return int(str(x)[0]) * 10 ** (len(str(x)) - 1)





def dump_artifact(artifact, name):
    pickle_file = open(name, 'wb')
    pickle.dump(artifact, pickle_file)
    pickle_file.close()

def load_artifact(name):
    pickle_file = open(name, 'rb')
    artifact = pickle.load(pickle_file)
    pickle_file.close()
    return artifact

