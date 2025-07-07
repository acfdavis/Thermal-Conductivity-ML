from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVR
import xgboost as xgb
import numpy as np
import pandas as pd
import re
import joblib


def normalize_column_names(df):
    df.columns = [re.sub(r"[\s\[\]<>]", "_", col) for col in df.columns]
    return df


def split_data(X, y):
    X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(
        X, np.log1p(y), y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train_log, y_test_log, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    X_train_scaled = normalize_column_names(X_train_scaled)
    X_test_scaled = normalize_column_names(X_test_scaled)
    return X_train_scaled, X_test_scaled, scaler


def train_random_forest(X_train, y_train_log):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train_log)
    return rf_model


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


def evaluate_model(model, X_test, y_test_log, y_test):
    if isinstance(model, xgb.XGBRegressor):
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


def get_optimal_k(X_scaled, k_range=range(2, 11)):
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_k = k_range[np.argmax(silhouette_scores)]
    return optimal_k, silhouette_scores


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