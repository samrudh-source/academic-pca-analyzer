import numpy as np
from typing import Tuple, Dict

def compute_mean_vector(X: np.ndarray) -> np.ndarray:
    """
    Computes the mean vector mu for feature matrix X (n_students, m_features).
    mu_j = (1/n) * sum_i(X_ij)
    """
    return np.mean(X, axis=0)

def center_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centers the matrix X by subtracting the feature means.
    X_c = X - mu
    Returns (X_c, mu)
    """
    mu = compute_mean_vector(X)
    X_c = X - mu
    return X_c, mu

def get_centering_details(X: np.ndarray, feature_names: list) -> Dict[str, any]:
    """
    Returns structured data for visualization and mathematical display of Stage 2 (Centering).
    """
    X_c, mu = center_data(X)
    mean_dict = dict(zip(feature_names, mu))
    return {
        'mean_vector': mu,
        'mean_dict': mean_dict,
        'X_centered': X_c
    }
