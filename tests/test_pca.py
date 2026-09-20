import pytest
import numpy as np
import pandas as pd
import os
import sys

# Ensure src module is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import validate_and_prepare_data
from src.preprocessing import center_data, compute_mean_vector
from src.pca_calculator import (
    compute_covariance_matrix, 
    compute_eigen_decomposition, 
    verify_orthogonality,
    project_data,
    run_complete_pca
)
from src.verification import verify_with_sklearn, generate_verification_summary

@pytest.fixture
def dummy_data():
    """Generates synthetic student score data for testing."""
    np.random.seed(42)
    data = {
        'Student_ID': [f'S{100+i}' for i in range(10)],
        'Student_Name': [f'Student_{i}' for i in range(10)],
        'Mathematics': np.random.uniform(50, 100, 10),
        'Physics': np.random.uniform(50, 100, 10),
        'Chemistry': np.random.uniform(50, 100, 10),
        'Programming': np.random.uniform(50, 100, 10)
    }
    return pd.DataFrame(data)

def test_data_validation(dummy_data):
    prepared = validate_and_prepare_data(dummy_data)
    assert prepared['n_students'] == 10
    assert prepared['m_features'] == 4
    assert len(prepared['feature_names']) == 4
    assert 'Student_Name' not in prepared['feature_names']
    assert prepared['X'].shape == (10, 4)

def test_centering(dummy_data):
    prepared = validate_and_prepare_data(dummy_data)
    X = prepared['X']
    X_c, mu = center_data(X)
    
    # Check that row means of X_c are approximately zero
    centered_means = np.mean(X_c, axis=0)
    np.testing.assert_allclose(centered_means, 0.0, atol=1e-12)

def test_covariance_matrix(dummy_data):
    prepared = validate_and_prepare_data(dummy_data)
    X_c, _ = center_data(prepared['X'])
    C = compute_covariance_matrix(X_c)
    
    # Check covariance matrix shape
    assert C.shape == (4, 4)
    # Check symmetry C = C^T
    np.testing.assert_allclose(C, C.T, atol=1e-12)

def test_eigen_decomposition(dummy_data):
    prepared = validate_and_prepare_data(dummy_data)
    X_c, _ = center_data(prepared['X'])
    C = compute_covariance_matrix(X_c)
    evals, evecs, var_ratio, cum_var_ratio = compute_eigen_decomposition(C)
    
    # Check descending order of eigenvalues
    assert np.all(np.diff(evals) <= 0)
    # Check sum of variance ratio equals 1.0
    np.testing.assert_allclose(np.sum(var_ratio), 1.0, atol=1e-7)

def test_orthogonality(dummy_data):
    prepared = validate_and_prepare_data(dummy_data)
    X_c, _ = center_data(prepared['X'])
    C = compute_covariance_matrix(X_c)
    _, evecs, _, _ = compute_eigen_decomposition(C)
    
    V_T_V, max_error = verify_orthogonality(evecs)
    identity = np.eye(4)
    np.testing.assert_allclose(V_T_V, identity, atol=1e-10)
    assert max_error < 1e-10

def test_projection(dummy_data):
    prepared = validate_and_prepare_data(dummy_data)
    res = run_complete_pca(prepared['X'], prepared['feature_names'], k=2)
    Z = res['Z']
    
    assert Z.shape == (10, 2)

def test_sklearn_verification(dummy_data):
    prepared = validate_and_prepare_data(dummy_data)
    manual_res = run_complete_pca(prepared['X'], prepared['feature_names'], k=2)
    sklearn_res = verify_with_sklearn(prepared['X'], prepared['feature_names'], k=2)
    summary = generate_verification_summary(manual_res, sklearn_res)
    
    # Check that eigenvalue difference is negligible (< 1e-5)
    assert summary['eval_max_diff'] < 1e-5
    # Check that variance ratio difference is negligible (< 1e-5)
    assert summary['var_ratio_max_diff'] < 1e-5
