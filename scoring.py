from pathlib import Path

import numpy as np
import pandas as pd
import pickle



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


def calculate_continuous_vals_reconstruction_score(train, reconstruction, hidden_features):
    results = {}
    for hidden_feature in hidden_features:
        real = train[hidden_feature].values
        recon = reconstruction[hidden_feature].values

        # Normalize by range of real data
        data_range = real.max() - real.min()

        if data_range == 0:
            # Constant column
            normalized_error = 0 if np.allclose(real, recon) else np.inf
        else:
            # Normalized absolute error
            normalized_error = np.abs(real - recon) / data_range

        results[hidden_feature] = {
            'mean_abs_error': np.mean(np.abs(real - recon)),
            'normalized_mae': np.mean(normalized_error),
            'mse': np.mean((real - recon) ** 2),
            'rmse': np.sqrt(np.mean((real - recon) ** 2)),
            'normalized_rmse': np.sqrt(np.mean(normalized_error ** 2)),
            'max_error': np.max(np.abs(real - recon))
        }

    return pd.DataFrame(results).T



def simple_accuracy_score(df_original, df_reconstructed, hidden_features):
    total_records = len(df_original)

    scores = []
    for col in hidden_features:
        score = (df_original[col].values == df_reconstructed[col].values).sum()
        scores.append(round(score / total_records * 100, 1))
    return scores
