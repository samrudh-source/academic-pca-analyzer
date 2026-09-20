import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

def load_student_data(file_source: Any) -> pd.DataFrame:
    """
    Load student dataset from a file path or file-like object (e.g. Streamlit UploadedFile).
    """
    if isinstance(file_source, str):
        df = pd.read_csv(file_source)
    else:
        df = pd.read_csv(file_source)
    return df

def validate_and_prepare_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates dataset, identifies numeric feature columns vs metadata columns,
    handles missing values, and extracts feature matrix X.

    Returns dictionary containing:
    - original_df: full original DataFrame
    - metadata_df: non-numerical columns (IDs, names, etc.)
    - feature_df: numerical feature DataFrame (cleaned)
    - X: numpy array of shape (n, m)
    - feature_names: list of feature column names
    - n_students: number of rows (n)
    - m_features: number of features (m)
    - missing_values_count: dict of missing values per column
    """
    original_df = df.copy()
    
    # Identify non-numerical columns or typical ID columns
    id_like_cols = [c for c in df.columns if any(term in c.lower() for term in ['id', 'name', 'roll', 'student', 'sno', 's_no'])]
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Feature columns are numeric columns excluding those that are pure ID numbers if specified
    # However, if user uploaded a file where all are numeric, keep non-ID ones
    feature_cols = [col for col in numeric_cols if col not in id_like_cols]
    
    # If feature_cols is empty, fall back to all numeric columns
    if len(feature_cols) == 0:
        feature_cols = numeric_cols

    metadata_cols = [col for col in df.columns if col not in feature_cols]
    
    metadata_df = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)
    feature_df = df[feature_cols].copy()
    
    # Check for missing values
    missing_counts = feature_df.isnull().sum().to_dict()
    
    # Impute missing values with mean if any exist
    if feature_df.isnull().values.any():
        feature_df = feature_df.fillna(feature_df.mean())
        
    X = feature_df.values.astype(float)
    n_students, m_features = X.shape

    return {
        'original_df': original_df,
        'metadata_df': metadata_df,
        'feature_df': feature_df,
        'X': X,
        'feature_names': feature_cols,
        'n_students': n_students,
        'm_features': m_features,
        'missing_values_count': missing_counts
    }
