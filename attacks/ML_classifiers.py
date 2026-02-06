import numpy as np
import pandas as pd
from os.path import join
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb



# ============================================================================
# DEFAULTS
# ============================================================================

# Logistic Regression
logistic_max_iter_default = 100

# Random Forest
rf_num_estimators_default = 100
rf_max_depth_default = 10

# LightGBM
lgb_num_estimators_default = 100
lgb_objective_default = 'multiclass'
lgb_metric_default = 'multi_logloss'
lgb_verbosity_default = -1

# KNN
knn_k_default = 1
knn_use_weights_default = True


# SVM Classifier
svm_clf_kernel_default = 'rbf'
svm_clf_C_default = 1.0
svm_clf_gamma_default = 'scale'



def _development():
    qis = [
        "QI1",
        "QI2",
    ]
    sdg_practice_problems = [
        # "25_Demo_AIM_e1_25f_Deid.csv",
        # "25_Demo_TVAE_25f_Deid.csv",
        # "25_Demo_CellSupression_25f_Deid.csv",
        # "25_Demo_Synthpop_25f_Deid.csv",
        "25_Demo_ARF_25f_Deid.csv",
        # "25_Demo_RANKSWAP_25f_Deid.csv",
        # "25_Demo_MST_e10_25f_Deid.csv",
    ]
    ml_methods = {
        # "NB": NB_reconstruction,
        "RF": random_forest_25_25_reconstruction,
        # "LR": logistic_regression_reconstruction,
        # "lgboost": lgboost_reconstruction,
        # "MLP": MLP_reconstruction,
        # "SVM": SVM_reconstruction,
        # "KNN": KNN_baseline,
        # "Ensemble": ensemble1_reconstruction,
        # "chained_RF": chained_rf_reconstruction,
        # "chained_NB": chained_nb_reconstruction,
    }

    mypath = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/25_PracticeProblem/"
    target_filename = "25_Demo_25f_OriginalData.csv"
    targets_original = pd.read_csv(join(mypath, target_filename))


    for ml_name, ml_method in ml_methods.items():

        for qi_name in qis:
            reconstruction_scores = pd.DataFrame(index=features_25)
            reconstruction_scores["nunique"] = targets_original.nunique()

            qi = QIs[qi_name]
            hidden_features = minus_QIs[qi_name]
            # hidden_features = list(set(features_25).difference(set(qi)))
            reconstruction_scores[f"{qi_name}_random_guess"] = np.NAN
            reconstruction_scores.loc[hidden_features, f"{qi_name}_random_guess"] = pd.Series(round(1 / targets_original.nunique() * 100, 1), index=features_25)
            targets = targets_original[qi]

            for deid_filename in sdg_practice_problems:

                sdg_method_name = "_".join(deid_filename.split("_")[2:-2])
                # print("\n", qi_name, sdg_method_name, ml_name)
                deid = pd.read_csv(join(mypath, deid_filename))

                recon_method_name = f"{qi_name}_{ml_name}_{sdg_method_name}"
                recon, _, _ = ml_method(deid, targets, qi, hidden_features, problem_name=sdg_method_name, qi_name=qi_name)
                # recon, _, _ = ml_method(deid, targets, qi, hidden_features)
                reconstruction_scores.loc[hidden_features, recon_method_name] = calculate_reconstruction_score(targets_original, recon, hidden_features)

                # print(qi_name, recon_method_name)
                # for x in reconstruction_scores.loc[sorted(hidden_features), recon_method_name].T.to_numpy():
                #     print(x, end=",")


            # Set display width very large to avoid wrapping and truncating
            pd.set_option('display.width', 1000)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            print(qi_name)
            for l in reconstruction_scores.loc[sorted(hidden_features)].T.to_numpy()[2:]:
                for x in l:
                    print(x, end=",")
                print()
                # print(round(l.mean(), 2))
            print("ave: ", round(reconstruction_scores.loc[sorted(hidden_features)].T.iloc[2:].mean().mean(), 2))



# ============================================================================
# LOGISTIC REGRESSION
# ============================================================================

def logistic_regression_reconstruction(cfg, deid, targets, qi, hidden_features):
    max_iter = cfg["attack_params"].get("max_iter", logistic_max_iter_default)
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        model = LogisticRegression(max_iter=max_iter)
        y = deid[hidden_feature].astype(str)

        # Handle edge case: ensure at least 2 classes
        if y.nunique() < 2:
            y.iloc[0] = '1'  # Use iloc for safer assignment

        model.fit(deid[qi].astype(str), y)
        reconstructed_targets[hidden_feature] = model.predict(targets[qi].astype(str))

    return reconstructed_targets.astype(int), None, None


# ============================================================================
# RANDOM FOREST
# ============================================================================

def random_forest_reconstruction(cfg, deid, targets, qi, hidden_features):
    num_estimators = cfg["attack_params"].get("num_estimators", rf_num_estimators_default)
    max_depth = cfg["attack_params"].get("max_depth", rf_max_depth_default)
    reconstructed_targets = targets.copy()
    probas = []
    classes_ = []
    for hidden_feature in hidden_features:
        model = RandomForestClassifier(n_estimators=num_estimators, max_depth=max_depth)
        model.fit(deid[qi], deid[hidden_feature])

        reconstructed_targets[hidden_feature] = model.predict(targets[qi])
        probas.append(model.predict_proba(targets[qi]))
        classes_.append(model.classes_)

    return reconstructed_targets, probas, classes_


def lgboost_reconstruction(cfg, deid, targets, qi, hidden_features):
    num_estimators = cfg["attack_params"].get("lgb_num_estimators", lgb_num_estimators_default)
    objective = cfg["attack_params"].get("lgb_objective", lgb_objective_default)
    metric = cfg["attack_params"].get("lgb_metric", lgb_metric_default)
    verbosity = cfg["attack_params"].get("lgb_verbosity", lgb_verbosity_default)

    reconstructed_targets = targets.copy()
    probas = []
    classes_ = []

    for hidden_feature in hidden_features:
        # Determine number of classes dynamically
        num_class = max(2, deid[hidden_feature].nunique())

        params = {
            "num_class": num_class,
            'objective': objective,
            'metric': metric,
            'n_estimators': num_estimators,
            'verbosity': verbosity,
        }

        model = lgb.LGBMClassifier(**params)
        y = deid[hidden_feature].copy()

        # Handle edge case: ensure at least 2 classes
        if y.nunique() < 2:
            y.iloc[0] = 99

        model.fit(deid[qi], y)
        reconstructed_targets[hidden_feature] = model.predict(targets[qi])
        probas.append(model.predict_proba(targets[qi]))
        classes_.append(model.classes_)

    return reconstructed_targets, probas, classes_


# ============================================================================
# NAIVE BAYES
# ============================================================================

def augment_df(df, classes, hidden_feature):
    """Helper function to augment dataframe with all possible classes."""
    # Create dummy rows for missing classes
    missing_classes = set(classes) - set(df[hidden_feature].unique())
    if not missing_classes:
        return df

    # Add one row per missing class (duplicating first row and changing target)
    augmented = df.copy()
    for cls in missing_classes:
        new_row = df.iloc[0:1].copy()
        new_row[hidden_feature] = cls
        augmented = pd.concat([augmented, new_row], ignore_index=True)

    return augmented


# def augment_df_OLD(deid, classes, hidden_feature):
#     deid_copy = deid.copy()
#     deid_copy = deid.iloc[:len(classes)]._append(deid_copy)
#     deid_copy.index = range(len(deid) + len(classes))
#     deid_copy.loc[range(len(classes)), hidden_feature] = classes
#
#     return deid_copy


def naive_bayes_reconstruction(cfg, deid, targets, qi, hidden_features, classes=None):
    reconstructed_targets = targets.copy()
    probas = []

    for hidden_feature in hidden_features:
        original_dtype = deid[hidden_feature].dtype
        model = GaussianNB()

        if classes is not None:
            # Augment training data to include all classes
            deid_augmented = augment_df(deid, classes[hidden_feature], hidden_feature)
            y_train = deid_augmented[hidden_feature]
            model.fit(deid_augmented[qi].astype(str), y_train.astype(str))

            # Verify all classes are present
            assert len(classes[hidden_feature]) == len(model.classes_), \
                f"Expected {len(classes[hidden_feature])} classes, got {len(model.classes_)}"
        else:
            y_train = deid[hidden_feature]
            model.fit(deid[qi].astype(str), y_train.astype(str))

        try:
            reconstructed_targets[hidden_feature] = model.predict(reconstructed_targets[qi].astype(str))

            # Restore original dtype
            if original_dtype == np.float64:
                reconstructed_targets[hidden_feature] = reconstructed_targets[hidden_feature].astype(float)
            else:
                reconstructed_targets[hidden_feature] = reconstructed_targets[hidden_feature].astype(int)
        except Exception as e:
            print(f"Exception occurred for feature {hidden_feature}: {e}")

        probas.append(model.predict_proba(reconstructed_targets[qi].astype(str)))

    return reconstructed_targets, probas, None


# ============================================================================
# K-NEAREST NEIGHBORS
# ============================================================================

# def KNN_reconstruction_OLD(cfg, deid, targets, qi, hidden_features):
#     k = cfg["attack_params"].get("k", knn_k_default)
#     use_weights = cfg["attack_params"].get("use_weights", knn_use_weights_default)
#
#     # Normalize QI features for fair distance calculation
#     scaler = StandardScaler()
#     syn_qi_scaled = scaler.fit_transform(deid[qi])
#     target_qi_scaled = scaler.transform(targets[qi])
#
#     # Fit KNN
#     nbrs = NearestNeighbors(n_neighbors=k).fit(syn_qi_scaled)
#     distances, indices = nbrs.kneighbors(target_qi_scaled)
#
#     recon = targets.copy()
#
#     for feature in hidden_features:
#         feature_values = []
#
#         for i in range(len(targets)):
#             neighbor_indices = indices[i]
#             neighbor_values = deid.iloc[neighbor_indices][feature].values
#
#             if use_weights and not np.isclose(distances[i][0], 0):
#                 # Distance-weighted voting
#                 weights = 1 / (distances[i] + 1e-6)
#                 weights = weights / np.sum(weights)
#                 weighted_value = np.sum(neighbor_values * weights)
#                 feature_values.append(round(weighted_value))
#             else:
#                 # Simple average (or mode for truly categorical)
#                 feature_values.append(round(np.mean(neighbor_values)))
#                 # TODO: For categorical features, consider using mode:
#                 # from scipy import stats
#                 # feature_values.append(stats.mode(neighbor_values, keepdims=False)[0])
#
#         recon[feature] = feature_values
#
#     return recon, None, None


# ============================================================================
# K-NEAREST NEIGHBORS (CATEGORICAL - K=1 ONLY)
# ============================================================================

def KNN_reconstruction(cfg, deid, targets, qi, hidden_features):
    # Force k=1 for categorical data
    k = 1

    # Normalize QI features for fair distance calculation
    scaler = StandardScaler()
    syn_qi_scaled = scaler.fit_transform(deid[qi])
    target_qi_scaled = scaler.transform(targets[qi])

    # Fit KNN
    nbrs = NearestNeighbors(n_neighbors=k).fit(syn_qi_scaled)
    distances, indices = nbrs.kneighbors(target_qi_scaled)

    recon = targets.copy()

    for feature in hidden_features:
        feature_values = []

        for i in range(len(targets)):
            neighbor_idx = indices[i][0]  # Only one neighbor (k=1)
            neighbor_value = deid.iloc[neighbor_idx][feature]
            feature_values.append(neighbor_value)

        recon[feature] = feature_values

    return recon.astype(int), None, None



# ============================================================================
# SVM CLASSIFIER
# ============================================================================

def SVM_classification_reconstruction(cfg, deid, targets, qi, hidden_features):
    kernel = cfg["attack_params"].get("kernel", svm_clf_kernel_default)
    C = cfg["attack_params"].get("C", svm_clf_C_default)
    gamma = cfg["attack_params"].get("gamma", svm_clf_gamma_default)
    probability = cfg["attack_params"].get("probability", True)  # Enable probability estimates

    # Normalize features for SVM (important for good performance)
    scaler = StandardScaler()

    targets_copy = targets.copy()
    probas = []
    classes_ = []

    for hidden_feature in hidden_features:
        # Scale input features
        X_train_scaled = scaler.fit_transform(deid[qi])
        X_target_scaled = scaler.transform(targets[qi])

        y_train = deid[hidden_feature]

        # Handle edge case: ensure at least 2 classes
        if y_train.nunique() < 2:
            y_train = y_train.copy()
            y_train.iloc[0] = 99

        # Train SVM Classifier
        model = SVC(kernel=kernel, C=C, gamma=gamma, probability=probability)
        model.fit(X_train_scaled, y_train)

        # Predict
        targets_copy[hidden_feature] = model.predict(X_target_scaled)

        # Get probabilities if enabled
        if probability:
            probas.append(model.predict_proba(X_target_scaled))
        else:
            probas.append(None)

        classes_.append(model.classes_)

    return targets_copy.astype(int), probas, classes_


def chained_rf_reconstruction(deid, targets, qi, hidden_features):
    reconstructed_targets = targets.copy()
    qi_copy = qi.copy()
    probas = []
    classes_ = []
    for hidden_feature in hidden_features:
        model = RandomForestClassifier(n_estimators=25, max_depth=25)
        type_ = deid[hidden_feature].dtypes
        model.fit(deid[qi_copy].astype(str), deid[hidden_feature].astype(str))
        reconstructed_targets[hidden_feature] = model.predict(reconstructed_targets[qi_copy])
        if type_ == "float64":
            reconstructed_targets[hidden_feature] = reconstructed_targets[hidden_feature].astype(float)
        else:
            reconstructed_targets[hidden_feature] = reconstructed_targets[hidden_feature].astype(int)
        probas.append(model.predict_proba(reconstructed_targets[qi_copy]))
        classes_.append(model.classes_)
        qi_copy.append(hidden_feature)
    return reconstructed_targets, probas, classes_


def chained_nb_reconstruction(deid, targets, qi, hidden_features):
    reconstructed_targets = targets.copy()
    qi_copy = qi.copy()
    for hidden_feature in hidden_features:
        model = GaussianNB()
        type_ = deid[hidden_feature].dtypes
        model.fit(deid[qi_copy].astype(str), deid[hidden_feature].astype(str))
        reconstructed_targets[hidden_feature] = model.predict(reconstructed_targets[qi_copy].astype(str))
        qi_copy.append(hidden_feature)
        if type_ == "float64":
            reconstructed_targets[hidden_feature] = reconstructed_targets[hidden_feature].astype(float)
        else:
            reconstructed_targets[hidden_feature] = reconstructed_targets[hidden_feature].astype(int)
    return reconstructed_targets, None, None

def ensemble1_reconstruction(deid, targets, qi, hidden_features):
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        rf_recon, rf_probas, rf_classes_ = random_forest_25_25_reconstruction(deid, targets, qi, [hidden_feature])
        lgb_recon, lgb_probas, lgb_classes_ = lgboost_reconstruction(deid, targets, qi, [hidden_feature])
        knn_recon, _, _ = KNN_baseline(deid, targets, qi, [hidden_feature])

        rf_recon, rf_probas = rf_recon[hidden_feature], rf_probas[0]
        lgb_recon, lgb_probas = lgb_recon[hidden_feature], lgb_probas[0] if lgb_classes_[0].shape == rf_classes_[0].shape else np.delete(lgb_probas[0], 1, axis=1)
        knn_recon = knn_recon[hidden_feature]

        ohe = OneHotEncoder(categories=[rf_classes_[0]], sparse_output=False)
        reshaped_for_enc = knn_recon.values.reshape(-1, 1)
        knn_probas = ohe.fit_transform(reshaped_for_enc)

        ensemble = np.mean([rf_probas, lgb_probas, knn_probas], axis=0)
        predicted_indices = np.argmax(ensemble, axis=1)
        idx_to_class = dict(enumerate(rf_classes_[0]))
        predicted_classes = np.array([idx_to_class[idx] for idx in predicted_indices])
        reconstructed_targets[hidden_feature] = predicted_classes

    return reconstructed_targets, ensemble, None





if __name__ == "__main__":
    _development()
