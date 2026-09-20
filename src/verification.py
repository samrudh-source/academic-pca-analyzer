import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Dict, Any

def verify_with_sklearn(X: np.ndarray, feature_names: list, k: int = 2) -> Dict[str, Any]:
    """
    Computes PCA using scikit-learn's PCA class and compares results against
    the manual step-by-step linear algebra calculations.

    Returns comparison dictionary and detailed Pandas DataFrame tables.
    """
    n_students, m_features = X.shape
    k = min(k, m_features)
    
    # Fit sklearn PCA
    sklearn_pca = PCA(n_components=k)
    Z_sklearn = sklearn_pca.fit_transform(X)
    
    sk_evals = sklearn_pca.explained_variance_
    sk_var_ratio = sklearn_pca.explained_variance_ratio_
    sk_components = sklearn_pca.components_  # shape (k, m) - rows are PCs
    
    return {
        'sk_pca_object': sklearn_pca,
        'sk_evals': sk_evals,
        'sk_var_ratio': sk_var_ratio,
        'sk_components': sk_components,
        'Z_sklearn': Z_sklearn
    }

def generate_verification_summary(
    manual_results: Dict[str, Any], 
    sklearn_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Builds comparative summary table comparing manual NumPy calculations vs Scikit-Learn.
    Handles sign orientation matching between manual and sklearn components.
    """
    k = manual_results['k']
    manual_evals = manual_results['eigenvalues'][:k]
    sk_evals = sklearn_results['sk_evals'][:k]
    
    manual_var_ratio = manual_results['variance_ratio'][:k]
    sk_var_ratio = sklearn_results['sk_var_ratio'][:k]
    
    Z_manual = manual_results['Z'][:, :k]
    Z_sklearn = sklearn_results['Z_sklearn'][:, :k]
    
    # Check max absolute differences for eigenvalues and variance ratios
    eval_max_diff = np.max(np.abs(manual_evals - sk_evals))
    var_ratio_max_diff = np.max(np.abs(manual_var_ratio - sk_var_ratio))
    
    # Adjust sign of Z_sklearn or Z_manual column-wise for visual comparison if needed
    Z_sk_aligned = Z_sklearn.copy()
    sign_flips = []
    for col in range(k):
        # dot product check
        dot = np.dot(Z_manual[:, col], Z_sklearn[:, col])
        if dot < 0:
            Z_sk_aligned[:, col] = -Z_sk_aligned[:, col]
            sign_flips.append(True)
        else:
            sign_flips.append(False)
            
    score_max_diff = np.max(np.abs(Z_manual - Z_sk_aligned))
    
    # Create metric comparison dataframe
    comp_data = []
    for i in range(k):
        comp_data.append({
            'Component': f"PC{i+1}",
            'Manual Eigenvalue': manual_evals[i],
            'Sklearn Eigenvalue': sk_evals[i],
            'Eigenvalue Diff': abs(manual_evals[i] - sk_evals[i]),
            'Manual Var Ratio (%)': manual_var_ratio[i] * 100,
            'Sklearn Var Ratio (%)': sk_var_ratio[i] * 100,
            'Var Ratio Diff (%)': abs(manual_var_ratio[i] - sk_var_ratio[i]) * 100,
            'Sign Flipped?': 'Yes (-1 multiplier)' if sign_flips[i] else 'No'
        })
        
    summary_df = pd.DataFrame(comp_data)
    
    return {
        'summary_df': summary_df,
        'eval_max_diff': eval_max_diff,
        'var_ratio_max_diff': var_ratio_max_diff,
        'score_max_diff': score_max_diff,
        'Z_sk_aligned': Z_sk_aligned,
        'sign_flips': sign_flips
    }
