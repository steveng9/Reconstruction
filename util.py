
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

run_test()