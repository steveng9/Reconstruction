import pickle

from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor, BayesianRidge, ElasticNet, HuberRegressor, RANSACRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import lightgbm as lgb



from NN_regression import mlp_repression_reconstruction

from attacks.baselines_continuous import *
from scoring import calculate_continuous_vals_reconstruction_score
from get_data import map_with_interpolation


# epss = ['{0:.2f}'.format(10**x) for x in np.arange(-1, 3.1, 0.5)]
# epss = ['{0:.2f}'.format(10**x) for x in np.arange(-1, 3.1)]
epss = ['10.00', '1000.00']
N_RUNS = 5
qi_split = 5


knn_k_default = 5  # Can use k>1 for continuous features
knn_use_weights_default = True
degree_default = 2
epsilon_default = 1.35
alpha_default = 1.0
alpha_sdg_default = 0.0001
l1_ratio_default = 0.5
penalty_default = 'l2'
max_iter_default = 100
max_iter_sdg_default = 1000



# Random Forest Regressor
rf_num_estimators_default = 100
rf_max_depth_default = 10
rf_min_samples_split_default = 2

# LightGBM Regressor
lgb_num_estimators_default = 100
lgb_learning_rate_default = 0.1
lgb_max_depth_default = -1  # -1 means no limit
lgb_verbosity_default = -1

# SVM Regressor
svm_kernel_default = 'rbf'
svm_C_default = 1.0
svm_epsilon_default = 0.1
svm_gamma_default = 'scale'


def _development():
    attacks = [
        # baselines
        simply_measure_deid_itself_baseline_cont,
        random_baseline_cont,
        mean_baseline_cont,
        median_baseline_cont,
        # conditional_mean_baseline_cont,
        # conditional_median_baseline_cont,
        random_normal_baseline_cont,
        nearest_neighbor_baseline_cont,
        # linear models
        linear_regression_reconstruction,
        # ridge_regression_reconstruction,
        # lasso_regression_reconstruction,
        sdgregressor_reconstruction,
        # bayesian_ridge_reconstruction,
        # elastic_net_reconstruction,
        # huber_regressor_reconstruction,
        # ransac_regressor_reconstruction,
        # polynomial_regressor_reconstruction,
        # MLPs
        mlp_repression_reconstruction,
        # new models
        # KNN_reconstruction_continuous,
        # SVM_regression_reconstruction,
        # random_forest_regression_reconstruction,
        # lgboost_regression_reconstruction,
    ]

    comparison_df = pd.DataFrame()
    dummy_cfg = {"attack_params": dict()}

    for attack in attacks:
        all_scores = pd.DataFrame(columns=['mean_abs_error','normalized_mae','mse','rmse','normalized_rmse','max_error','eps'])
        for eps in epss:
            for synth_id in range(N_RUNS):
                data_path = "/Users/stevengolob/Documents/school/Thesis/experiment_artifacts/shadowsets_cali/"
                sdg_method = f"expD/e{eps}/gsd/"

                # load data
                columns = [str(x) for x in range(9)]
                full_data = pd.DataFrame(StandardScaler().fit_transform(fetch_california_housing(as_frame=True).frame.sample(frac=1)), columns=columns)
                # full_data = fetch_california_housing(as_frame=True).frame.sample(frac=1)
                pickle_file = open(data_path + sdg_method + f"s{synth_id}_train_ids", 'rb')
                ids = pickle.load(pickle_file)
                train = full_data.loc[ids]
                pickle_file.close()
                synth_encoded = pd.read_parquet(data_path + sdg_method + f"s{synth_id}.parquet")
                decoding_thresholds = pickle.load(open(data_path + "cali_thresholds_for_continuous_features_1000.00_20", "rb"))
                # synth = map_to_bin_edges(synth_encoded, decoding_thresholds)
                synth = map_with_interpolation(synth_encoded, decoding_thresholds)


                qi = columns[:qi_split]
                hidden_features = columns[qi_split:]
                targets = train[qi]

                print(f"\n\n{attack.__name__}:\n")
                recon, _, _ = attack(dummy_cfg, synth, targets, qi, hidden_features)
                scores = calculate_continuous_vals_reconstruction_score(train, recon, hidden_features)
                score_means = scores.mean()
                score_means['eps'] = float(eps)
                all_scores = pd.concat([all_scores, score_means.to_frame().T], ignore_index=True)
        all_scores = all_scores.groupby(['eps']).mean()

        comparison_df[attack.__name__] = all_scores["normalized_rmse"]

        # all_scores.set_index('eps', inplace=True)
    # Show all rows and columns for this one print
    with pd.option_context('display.width', 1000, 'display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', 12):
        print(comparison_df)
    print()




# ============================================================================
# K-NEAREST NEIGHBORS (CONTINUOUS)
# ============================================================================

def KNN_reconstruction_continuous(cfg, deid, targets, qi, hidden_features):
    k = cfg["attack_params"].get("knn_k", knn_k_default)
    use_weights = cfg["attack_params"].get("knn_use_weights", knn_use_weights_default)

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
            neighbor_indices = indices[i]
            neighbor_values = deid.iloc[neighbor_indices][feature].values

            if use_weights and k > 1:
                # Distance-weighted average
                # Check if closest neighbor is exact match (distance ~= 0)
                if np.isclose(distances[i][0], 0):
                    # Exact match - use that value
                    feature_values.append(neighbor_values[0])
                else:
                    # Weight by inverse distance
                    weights = 1 / (distances[i] + 1e-6)
                    weights = weights / np.sum(weights)
                    weighted_value = np.sum(neighbor_values * weights)
                    feature_values.append(weighted_value)
            else:
                # Simple average (unweighted)
                feature_values.append(np.mean(neighbor_values))

        recon[feature] = feature_values

    return recon, None, None


def linear_regression_reconstruction(cfg, synth, targets, qi, hidden_features):
    # StandardScaler().fit(targets)
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        model = LinearRegression()
        y = synth[hidden_feature]
        model.fit(synth[qi], y)
        reconstructed_targets[hidden_feature] = model.predict(targets)
    return reconstructed_targets, None, None



def ridge_regression_reconstruction(cfg, synth, targets, qi, hidden_features):
    # StandardScaler().fit(targets)
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        model = Ridge(alpha=cfg["attack_params"].get("alpha", alpha_default))
        y = synth[hidden_feature]
        model.fit(synth[qi], y)
        reconstructed_targets[hidden_feature] = model.predict(targets)
    return reconstructed_targets, None, None


def lasso_regression_reconstruction(cfg, synth, targets, qi, hidden_features):
    # StandardScaler().fit(targets)
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        model = Lasso(alpha=cfg["attack_params"].get("alpha", alpha_default))
        y = synth[hidden_feature]
        model.fit(synth[qi], y)
        reconstructed_targets[hidden_feature] = model.predict(targets)
    return reconstructed_targets, None, None


def sdgregressor_reconstruction(cfg, synth, targets, qi, hidden_features):
    # StandardScaler().fit(targets)
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        model = SGDRegressor(
            max_iter=cfg["attack_params"].get("max_iter_sdg", max_iter_sdg_default),
            penalty=cfg["attack_params"].get("penalty", penalty_default),
            alpha=cfg["attack_params"].get("alpha_sdg", alpha_sdg_default)
        )
        y = synth[hidden_feature]
        model.fit(synth[qi], y)
        reconstructed_targets[hidden_feature] = model.predict(targets)
    return reconstructed_targets, None, None




def bayesian_ridge_reconstruction(cfg, synth, targets, qi, hidden_features):
    # StandardScaler().fit(targets)
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        model = BayesianRidge()
        y = synth[hidden_feature]
        model.fit(synth[qi], y)
        reconstructed_targets[hidden_feature] = model.predict(targets)
    return reconstructed_targets, None, None


def elastic_net_reconstruction(cfg, synth, targets, qi, hidden_features):
    # StandardScaler().fit(targets)
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        model = ElasticNet(
            alpha=cfg["attack_params"].get("alpha",alpha_default),
            l1_ratio=cfg["attack_params"].get("l1_ratio",l1_ratio_default)
        )
        y = synth[hidden_feature]
        model.fit(synth[qi], y)
        reconstructed_targets[hidden_feature] = model.predict(targets)
    return reconstructed_targets, None, None


def huber_regressor_reconstruction(cfg, synth, targets, qi, hidden_features):
    # StandardScaler().fit(targets)
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        model = HuberRegressor(epsilon=cfg["attack_params"].get("epsilon", epsilon_default))
        y = synth[hidden_feature]
        model.fit(synth[qi], y)
        reconstructed_targets[hidden_feature] = model.predict(targets)
    return reconstructed_targets, None, None


def ransac_regressor_reconstruction(cfg, synth, targets, qi, hidden_features):
    # StandardScaler().fit(targets)
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        model = RANSACRegressor()
        y = synth[hidden_feature]
        model.fit(synth[qi], y)
        reconstructed_targets[hidden_feature] = model.predict(targets)
    return reconstructed_targets, None, None


def polynomial_regressor_reconstruction(cfg, synth, targets, qi, hidden_features):
    # StandardScaler().fit(targets)
    reconstructed_targets = targets.copy()
    for hidden_feature in hidden_features:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=cfg["attack_params"].get("degree",degree_default))),
            ('linear', LinearRegression())
        ])
        y = synth[hidden_feature]
        model.fit(synth[qi], y)
        reconstructed_targets[hidden_feature] = model.predict(targets)
    return reconstructed_targets, None, None


# ============================================================================
# RANDOM FOREST REGRESSOR
# ============================================================================

def random_forest_regression_reconstruction(cfg, deid, targets, qi, hidden_features):
    num_estimators = cfg["attack_params"].get("num_estimators", rf_num_estimators_default)
    max_depth = cfg["attack_params"].get("max_depth", rf_max_depth_default)
    min_samples_split = cfg["attack_params"].get("min_samples_split", rf_min_samples_split_default)

    targets_copy = targets.copy()

    for hidden_feature in hidden_features:
        model = RandomForestRegressor(
            n_estimators=num_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )

        model.fit(deid[qi], deid[hidden_feature])
        targets_copy[hidden_feature] = model.predict(targets[qi])

    return targets_copy, None, None


# ============================================================================
# LIGHTGBM REGRESSOR
# ============================================================================

def lgboost_regression_reconstruction(cfg, deid, targets, qi, hidden_features):
    num_estimators = cfg["attack_params"].get("num_estimators", lgb_num_estimators_default)
    learning_rate = cfg["attack_params"].get("learning_rate", lgb_learning_rate_default)
    max_depth = cfg["attack_params"].get("max_depth", lgb_max_depth_default)
    verbosity = cfg["attack_params"].get("verbosity", lgb_verbosity_default)

    targets_copy = targets.copy()

    for hidden_feature in hidden_features:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': num_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'verbosity': verbosity,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(deid[qi], deid[hidden_feature])

        targets_copy[hidden_feature] = model.predict(targets[qi])

    return targets_copy, None, None


# ============================================================================
# SVM REGRESSOR
# ============================================================================

def SVM_regression_reconstruction(cfg, deid, targets, qi, hidden_features):
    kernel = cfg["attack_params"].get("kernel", svm_kernel_default)
    C = cfg["attack_params"].get("C", svm_C_default)
    epsilon = cfg["attack_params"].get("epsilon", svm_epsilon_default)
    gamma = cfg["attack_params"].get("gamma", svm_gamma_default)

    # Normalize features for SVM (important for good performance)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    targets_copy = targets.copy()

    for hidden_feature in hidden_features:
        # Scale input features
        X_train_scaled = scaler_X.fit_transform(deid[qi])
        X_target_scaled = scaler_X.transform(targets[qi])

        # Scale target variable
        y_train = deid[[hidden_feature]].values
        y_train_scaled = scaler_y.fit_transform(y_train).ravel()

        # Train SVM
        model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        model.fit(X_train_scaled, y_train_scaled)

        # Predict and inverse transform
        y_pred_scaled = model.predict(X_target_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        targets_copy[hidden_feature] = y_pred

    return targets_copy, None, None


if __name__ == "__main__":
    _development()
