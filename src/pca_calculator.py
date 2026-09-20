import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def compute_covariance_matrix(X_c: np.ndarray) -> np.ndarray:
    """
    Computes covariance matrix C manually using:
    C = (1 / (n - 1)) * X_c^T * X_c
    """
    n = X_c.shape[0]
    if n <= 1:
        raise ValueError("Number of samples (students) must be greater than 1 to compute sample covariance.")
    C = (1.0 / (n - 1)) * np.dot(X_c.T, X_c)
    return C

def compute_eigen_decomposition(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes eigenvalues and eigenvectors using np.linalg.eigh(C).
    Sorts them in descending order of eigenvalues.

    Returns:
    - eigenvalues: sorted 1D array (m,)
    - eigenvectors: sorted 2D array (m, m) where columns are eigenvectors v_i
    - variance_explained_ratio: array of lambda_i / sum(lambda)
    - cumulative_variance_ratio: array of cumulative sums
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Handle numerical inaccuracies (clip negative values to zero)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    
    total_variance = np.sum(eigenvalues)
    if total_variance == 0:
        variance_explained_ratio = np.zeros_like(eigenvalues)
    else:
        variance_explained_ratio = eigenvalues / total_variance
        
    cumulative_variance_ratio = np.cumsum(variance_explained_ratio)
    
    return eigenvalues, eigenvectors, variance_explained_ratio, cumulative_variance_ratio

def verify_orthogonality(V: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculates V^T * V to demonstrate that eigenvectors form an orthonormal basis.
    For orthonormal eigenvectors, V^T * V should equal Identity matrix I.
    Returns (V_T_V, max_absolute_error_from_identity).
    """
    V_T_V = np.dot(V.T, V)
    identity = np.eye(V.shape[1])
    max_diff = np.max(np.abs(V_T_V - identity))
    return V_T_V, max_diff

def select_principal_components(
    eigenvectors: np.ndarray, 
    k: int
) -> np.ndarray:
    """
    Selects the first k principal eigenvectors: V_k shape (m, k).
    """
    return eigenvectors[:, :k]

def project_data(X_c: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    """
    Projects centered data matrix X_c onto selected principal eigenvectors V_k.
    Z = X_c * V_k
    """
    return np.dot(X_c, V_k)

def create_loading_table(
    eigenvectors: np.ndarray, 
    feature_names: list, 
    k: int
) -> pd.DataFrame:
    """
    Creates a Pandas DataFrame for feature loadings (contributions) for top k components.
    Columns: PC1, PC2, ..., PCk
    Rows: Feature names
    """
    pc_cols = [f"PC{i+1}" for i in range(k)]
    V_k = eigenvectors[:, :k]
    loading_df = pd.DataFrame(V_k, index=feature_names, columns=pc_cols)
    return loading_df

def run_complete_pca(X: np.ndarray, feature_names: list, k: int = 2) -> Dict[str, Any]:
    """
    Runs full sequential PCA computational process.
    Returns structured results dictionary containing every intermediate state.
    """
    from src.preprocessing import center_data
    
    n_students, m_features = X.shape
    X_c, mu = center_data(X)
    C = compute_covariance_matrix(X_c)
    evals, evecs, var_ratio, cum_var_ratio = compute_eigen_decomposition(C)
    V_T_V, max_ortho_error = verify_orthogonality(evecs)
    
    # Clip k to max features
    k = max(1, min(k, m_features))
    V_k = select_principal_components(evecs, k)
    Z = project_data(X_c, V_k)
    loading_df = create_loading_table(evecs, feature_names, k)
    
    # Analyze strong feature contributions per PC
    top_features_per_pc = {}
    for i, col in enumerate(loading_df.columns):
        series = loading_df[col].abs()
        sorted_feats = series.sort_values(ascending=False)
        top_features_per_pc[col] = sorted_feats.head(3).index.tolist()
        
    return {
        'n': n_students,
        'm': m_features,
        'X': X,
        'mu': mu,
        'X_c': X_c,
        'C': C,
        'eigenvalues': evals,
        'eigenvectors': evecs,
        'variance_ratio': var_ratio,
        'cum_variance_ratio': cum_var_ratio,
        'V_T_V': V_T_V,
        'ortho_error': max_ortho_error,
        'k': k,
        'V_k': V_k,
        'Z': Z,
        'loading_df': loading_df,
        'top_features_per_pc': top_features_per_pc,
        'feature_names': feature_names
    }
