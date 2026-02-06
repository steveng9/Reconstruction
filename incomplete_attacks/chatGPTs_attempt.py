import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the files
mst_df = pd.read_csv("/Users/golobs/Documents/GradSchool/NIST-CRC-25/25_PracticeProblem/25_Demo_MST_e10_25f_Deid.csv")
qi1_df = pd.read_csv("/Users/golobs/Documents/GradSchool/NIST-CRC-25/25_PracticeProblem/25_Demo_25f_OriginalData_QI1.csv")

# Shared features
qi1_features = ['F2', 'F17', 'F22', 'F32', 'F37', 'F41', 'F47']
mst_common = mst_df[qi1_features]
qi1_common = qi1_df[qi1_features]

# Target features to reconstruct
target_features = [col for col in mst_df.columns if col not in qi1_features]

# Determine categorical vs numerical
categorical_targets = target_features
# categorical_targets = [col for col in target_features if mst_df[col].nunique() <= 10]
# numerical_targets = [col for col in target_features if col not in categorical_targets]

# Train models
models = {}
encoders = {}

for col in target_features:
    X = mst_common
    y = mst_df[col]

    if col in categorical_targets:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y_enc)
        models[col] = model
        encoders[col] = le
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        models[col] = model

# Predict with sampling
qi1_reconstructed = qi1_common.copy()

for col in target_features:
    model = models[col]
    if col in categorical_targets:
        le = encoders[col]
        probas = model.predict_proba(qi1_common)
        sampled = [np.random.choice(le.classes_, p=proba) for proba in probas]
        qi1_reconstructed[col] = sampled
    else:
        all_preds = np.stack([tree.predict(qi1_common) for tree in model.estimators_], axis=1)
        sampled = [np.random.choice(row) for row in all_preds]
        qi1_reconstructed[col] = sampled

# Merge with original dataframe
qi1_combined = qi1_df.copy()
for col in target_features:
    qi1_combined[col] = qi1_reconstructed[col]

# Reorder columns: Unnamed: 0, then F1–F50
all_features = ['Unnamed: 0'] + sorted([col for col in qi1_combined.columns if col != 'Unnamed: 0'], key=lambda x: int(x[1:]))
qi1_combined = qi1_combined[all_features]

# Save result
qi1_combined.to_csv("25_Demo_25f_Reconstructed.csv", index=False)
print("Reconstruction saved to 25_Demo_25f_Reconstructed.csv")
