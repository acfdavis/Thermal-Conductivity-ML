from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVR
import xgboost as xgb
import numpy as np
import pandas as pd
import re
import joblib
from kneed import KneeLocator


def normalize_column_names(df):
    df.columns = [re.sub(r"[\s\[\]<>]", "_", col) for col in df.columns]
    return df


def split_data(X, y):
    X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(
        X, np.log1p(y), y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train_log, y_test_log, y_train, y_test


def scale_features(X_train, X_test):
    """Scales features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, scaler


def apply_power_transform(X_train_scaled, X_test_scaled):
    """Applies Yeo-Johnson power transformation."""
    pt = PowerTransformer(method='yeo-johnson')
    X_train_transformed = pd.DataFrame(
        pt.fit_transform(X_train_scaled),
        columns=X_train_scaled.columns,
        index=X_train_scaled.index
    )
    X_test_transformed = pd.DataFrame(
        pt.transform(X_test_scaled),
        columns=X_test_scaled.columns,
        index=X_test_scaled.index
    )
    return X_train_transformed, X_test_transformed, pt


def train_random_forest(X_train, y_train_log):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train_log)
    return rf_model


def train_baseline_random_forest(X_train, y_train):
    """Trains a baseline Random Forest model with optimized parameters."""
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train_log):
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train_log)
    return gb_model


def train_svr(X_train, y_train_log):
    svr_model = SVR()
    svr_model.fit(X_train, y_train_log)
    return svr_model


def train_xgboost(X_train, y_train_log):
    X_train_cleaned = normalize_column_names(X_train.copy())
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train_cleaned, y_train_log)
    return xgb_model


def train_baseline_xgboost(X_train, y_train):
    """Trains a baseline XGBoost model."""
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test_log, y_test):
    if isinstance(model, xgb.XGBRegressor):
        # Ensure column names are strings
        X_test.columns = X_test.columns.astype(str)

        # Normalize column names
        X_test_cleaned = normalize_column_names(X_test.copy())
        y_pred_log = model.predict(X_test_cleaned)
    else:
        y_pred_log = model.predict(X_test)

    y_pred = np.expm1(y_pred_log)

    mae_log = mean_absolute_error(y_test_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
    r2_log = r2_score(y_test_log, y_pred_log)

    mae_original = mean_absolute_error(y_test, y_pred)
    rmse_original = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_original = r2_score(y_test, y_pred)

    results = {
        "log": {"mae": mae_log, "rmse": rmse_log, "r2": r2_log, "predictions": y_pred_log},
        "original": {
            "mae": mae_original,
            "rmse": rmse_original,
            "r2": r2_original,
            "predictions": y_pred,
        },
    }

    return results


def train_and_evaluate_model(model, X_train, y_train_log, X_test, y_test_log, y_test):
    """
    Train the given model and evaluate it using the standard metrics.
    Returns the fitted model and results dict.
    """
    model.fit(X_train, y_train_log)
    results = evaluate_model(model, X_test, y_test_log, y_test)
    return model, results


def compare_models(results_dict):
    """
    Given a dict of model_name: results (from evaluate_model),
    return a DataFrame of metrics for easy comparison.
    """
    rows = []
    for name, res in results_dict.items():
        row = {
            'Model': name,
            'R2 (log)': res['log']['r2'],
            'MAE (log)': res['log']['mae'],
            'RMSE (log)': res['log']['rmse'],
            'R2 (orig)': res['original']['r2'],
            'MAE (orig)': res['original']['mae'],
            'RMSE (orig)': res['original']['rmse'],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def get_optimal_k(X_scaled, k_range=range(2, 16)):
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_k = k_range[np.argmax(silhouette_scores)]
    return optimal_k, silhouette_scores


def get_optimal_k_with_metrics(X_scaled, k_range=range(2, 16)):
    """Calculate optimal k using multiple metrics."""
    silhouette_scores = []
    inertia_scores = []
    calinski_scores = []
    davies_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)

        # Calculate metrics
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        inertia_scores.append(kmeans.inertia_)
        calinski_scores.append(calinski_harabasz_score(X_scaled, kmeans.labels_))
        davies_scores.append(davies_bouldin_score(X_scaled, kmeans.labels_))

    optimal_k = k_range[np.argmax(silhouette_scores)]

    metrics = {
        "Silhouette": silhouette_scores,
        "Inertia": inertia_scores,
        "Calinski-Harabasz": calinski_scores,
        "Davies-Bouldin": davies_scores
    }

    return optimal_k, metrics


def add_cluster_features(X_scaled, X_original, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cluster_features = encoder.fit_transform(cluster_labels.reshape(-1, 1))

    cluster_df = pd.DataFrame(cluster_features, index=X_original.index, columns=[f'cluster_{i}' for i in range(n_clusters)])
    X_final = pd.concat([X_original, cluster_df], axis=1)

    return X_final, cluster_labels, kmeans, encoder


def apply_cluster_features(X_scaled, X_original, kmeans, encoder):
    cluster_labels = kmeans.predict(X_scaled)
    cluster_features = encoder.transform(cluster_labels.reshape(-1, 1))

    n_clusters = len(encoder.categories_[0])
    cluster_df = pd.DataFrame(cluster_features, index=X_original.index, columns=[f'cluster_{i}' for i in range(n_clusters)])
    X_final = pd.concat([X_original, cluster_df], axis=1)

    return X_final, cluster_labels


def save_model(model, path="outputs/final_model.pkl"):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def select_optimal_k(metrics_dict, k_range):
    """Select optimal k based on a composite score from normalized metrics."""
    normalized_metrics = {
        metric_name: (np.array(values) / max(values) if max(values) > 0 else np.array(values))
        for metric_name, values in metrics_dict.items()
    }

    # Adjust metrics where lower values are better (e.g., Davies-Bouldin Index)
    inverted_metrics = {
        metric_name: (1 - values if metric_name == "Davies-Bouldin" else values)
        for metric_name, values in normalized_metrics.items()
    }

    # Compute composite score by averaging normalized metrics
    composite_scores = np.mean(list(inverted_metrics.values()), axis=0)

    # Select k with the highest composite score
    optimal_k = k_range[np.argmax(composite_scores)]

    return optimal_k, composite_scores


def find_elbow_point(inertia_values, k_range):
    """
    Automates the calculation of the optimal k using the elbow method.
    
    Parameters:
    - inertia_values: List of inertia values for different k.
    - k_range: Range of k values corresponding to the inertia values.
    
    Returns:
    - optimal_k: The k value at the elbow point.
    """
    # Use KneeLocator to find the elbow point
    kneedle = KneeLocator(k_range, inertia_values, curve="convex", direction="decreasing")
    optimal_k = kneedle.knee
    return optimal_k


def get_inertia_values(X_scaled, k_range):
    """
    Calculate inertia values for a range of k values.

    Parameters:
    - X_scaled: Scaled feature matrix.
    - k_range: Range of k values to evaluate.

    Returns:
    - inertia_values: List of inertia values for each k.
    """
    inertia_values = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia_values.append(kmeans.inertia_)
    return inertia_values


def select_features(X, missing_thresh=0.3, low_var_thresh=1e-5, corr_thresh=0.85):
    """
    Selects features by dropping those with high missingness, low variance, or high correlation.
    """
    features_to_drop = []

    # 1. Drop features with >30% missing values
    missing = X.isnull().mean()
    drop_missing = missing[missing > missing_thresh].index.tolist()
    features_to_drop.extend(drop_missing)

    # 2. Drop features with very low variance
    low_var = X.std()
    drop_low_var = low_var[low_var < low_var_thresh].index.tolist()
    features_to_drop.extend(drop_low_var)

    # 3. Drop one of each pair of highly correlated features
    numeric_X = X.select_dtypes(include=np.number)
    corr_matrix = numeric_X.drop(columns=features_to_drop, errors='ignore').corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_high_corr = [column for column in upper.columns if any(upper[column] > corr_thresh)]
    features_to_drop.extend(drop_high_corr)

    # Get the final unique list of features to drop
    final_features_to_drop = list(set(features_to_drop))
    
    X_selected = X.drop(columns=final_features_to_drop, errors='ignore')
    
    return X_selected, final_features_to_drop


def compare_models_with_clusters(X_train, X_test, y_train_log, y_test_log, y_train, y_test, cluster_labels):
    # Train model without cluster labels
    baseline_model, baseline_results = train_and_evaluate_model(
        train_baseline_xgboost(X_train, y_train_log),
        X_train, y_train_log, X_test, y_test_log, y_test
    )

    # Convert cluster_labels to pandas Series with the same index as X_train and X_test
    cluster_labels_series = pd.Series(cluster_labels, index=X_train.index.union(X_test.index))

    # Add cluster labels to the dataset
    X_train_with_clusters = pd.concat([X_train, cluster_labels_series.loc[X_train.index]], axis=1)
    X_test_with_clusters = pd.concat([X_test, cluster_labels_series.loc[X_test.index]], axis=1)

    # Train model with cluster labels
    model_with_clusters, results_with_clusters = train_and_evaluate_model(
        train_baseline_xgboost(X_train_with_clusters, y_train_log),
        X_train_with_clusters, y_train_log, X_test_with_clusters, y_test_log, y_test
    )

    return {
        'Baseline (No Clusters)': baseline_results,
        'With Clusters': results_with_clusters
    }