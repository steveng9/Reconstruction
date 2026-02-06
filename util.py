from pathlib import Path

import numpy as np
import pickle
import pandas as pd


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

