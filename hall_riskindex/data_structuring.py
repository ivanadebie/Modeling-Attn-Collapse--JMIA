"""
Data structuring for change point detection pipeline.
Transforms feature dictionaries into matrices, normalizes, and applies smoothing/PCA.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Tuple

def features_to_matrix(features: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """
    Convert list of feature dicts to feature matrix and feature names.
    Returns:
        X: np.ndarray of shape (T, D)
        feature_names: list of feature names
    """
    feature_names = [k for k in features[0].keys() if k != "hallucination_label"]
    X = np.array([[f[name] for name in feature_names] for f in features])
    return X, feature_names

def normalize_features(X: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize features using z-score or robust scaling.
    """
    if method == "zscore":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("Unknown normalization method")
    return scaler.fit_transform(X)

def smooth_features(X: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Apply moving average smoothing to feature matrix.
    """
    df = pd.DataFrame(X)
    return df.rolling(window, min_periods=1, center=True).mean().values

def apply_pca(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    """
    Apply PCA to reduce dimensionality of feature matrix.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def structure_data(features: List[Dict[str, Any]], normalization: str = "zscore", smoothing_window: int = 3, pca_components: int = 10) -> Dict[str, Any]:
    """
    Full data structuring pipeline.
    Returns dict with raw, normalized, smoothed, and PCA-transformed matrices.
    """
    X_raw, feature_names = features_to_matrix(features)
    X_norm = normalize_features(X_raw, method=normalization)
    X_smooth = smooth_features(X_norm, window=smoothing_window)
    X_pca = apply_pca(X_smooth, n_components=min(pca_components, X_smooth.shape[1]))
    hallucination_labels = np.array([f["hallucination_label"] for f in features])
    return {
        "raw": X_raw,
        "normalized": X_norm,
        "smoothed": X_smooth,
        "pca": X_pca,
        "feature_names": feature_names,
        "labels": hallucination_labels
    }

