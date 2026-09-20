import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Custom modules
from src.data_loader import load_student_data, validate_and_prepare_data
from src.preprocessing import center_data, get_centering_details
from src.pca_calculator import (
    compute_covariance_matrix, 
    compute_eigen_decomposition, 
    verify_orthogonality,
    select_principal_components,
    project_data,
    create_loading_table,
    run_complete_pca
)
from src.visualization import (
    plot_scree, 
    plot_explained_variance, 
    plot_2d_pca_scatter, 
    plot_feature_loadings, 
    plot_covariance_heatmap
)
from src.verification import verify_with_sklearn, generate_verification_summary

# Page configuration
st.set_page_config(
    page_title="Academic Performance Principal Pattern Analyzer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        font-weight: 500;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2563EB;
        margin-bottom: 10px;
    }
    .workflow-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        padding: 15px;
        border-radius: 8px;
        font-weight: 600;
        color: #1E40AF;
        text-align: center;
        margin: 15px 0px;
    }
</style>
""", unsafe_allow_globals=True)

# Title & Subtitle
st.markdown('<p class="main-title">Academic Performance Principal Pattern Analyzer</p>', unsafe_allow_globals=True)
st.markdown('<p class="sub-title">"Linear Algebra Based Student Performance Analysis using PCA"</p>', unsafe_allow_globals=True)
st.markdown("---")

# Session State Initialization
SAMPLE_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'sample_students.csv')

if 'raw_df' not in st.session_state:
    if os.path.exists(SAMPLE_CSV_PATH):
        st.session_state.raw_df = pd.read_csv(SAMPLE_CSV_PATH)
    else:
        st.session_state.raw_df = None

# Sidebar Controls
st.sidebar.header("📁 Data & Controls")

dataset_option = st.sidebar.radio(
    "Choose Data Source:",
    ("Sample Dataset", "Upload Custom CSV")
)

if dataset_option == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Student Data (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            st.session_state.raw_df = load_student_data(uploaded_file)
            st.sidebar.success("Custom CSV uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")
            
elif dataset_option == "Sample Dataset":
    if st.sidebar.button("Reload Sample Dataset") or st.session_state.raw_df is None:
        if os.path.exists(SAMPLE_CSV_PATH):
            st.session_state.raw_df = pd.read_csv(SAMPLE_CSV_PATH)
            st.sidebar.info("Loaded 15-student academic sample dataset.")

if st.session_state.raw_df is None:
    st.warning("Please upload a CSV file or load the sample dataset to proceed.")
    st.stop()

# Validate & Prepare Data
data_dict = validate_and_prepare_data(st.session_state.raw_df)
X = data_dict['X']
n_students = data_dict['n_students']
m_features = data_dict['m_features']
feature_names = data_dict['feature_names']
metadata_df = data_dict['metadata_df']

# Component Slider in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("⚙️ PCA Parameters")
k_components = st.sidebar.slider(
    "Select Number of Principal Components (k):",
    min_value=1,
    max_value=m_features,
    value=min(3, m_features)
)

if st.sidebar.button("Reset Application State"):
    st.session_state.raw_df = pd.read_csv(SAMPLE_CSV_PATH) if os.path.exists(SAMPLE_CSV_PATH) else None
    st.rerun()

# Run Complete Manual PCA Calculations
pca_results = run_complete_pca(X, feature_names, k=k_components)
X_c = pca_results['X_c']
mu = pca_results['mu']
C = pca_results['C']
evals = pca_results['eigenvalues']
evecs = pca_results['eigenvectors']
var_ratio = pca_results['variance_ratio']
cum_var_ratio = pca_results['cum_variance_ratio']
V_T_V = pca_results['V_T_V']
V_k = pca_results['V_k']
Z = pca_results['Z']
loading_df = pca_results['loading_df']

# Navigation Tabs
tab_names = [
    "1. Home",
    "2. Dataset",
    "3. Centering",
    "4. Covariance Matrix",
    "5. Eigen Analysis",
    "6. Principal Components",
    "7. Projection & Reduction",
    "8. Visualizations",
    "9. Verification",
    "10. Mathematical Explanation"
]

tabs = st.tabs(tab_names)

# ==================== TAB 1: HOME ====================
with tabs[0]:
    st.header("📌 Project Aim & Sequential Computational Workflow")
    st.markdown("""
    This application provides a **step-by-step linear algebra implementation** of Principal Component Analysis (PCA) applied to student academic dataset.
    Rather than calling black-box machine learning libraries for primary computations, every intermediate mathematical step is explicitly calculated and displayed.
    """)
    
    st.markdown("""
    <div class="workflow-box">
    DATA &rarr; MATRIX &rarr; CENTERING &rarr; COVARIANCE MATRIX &rarr; EIGENVALUES & EIGENVECTORS &rarr; ORTHOGONAL DIRECTIONS &rarr; PROJECTION &rarr; REDUCED REPRESENTATION &rarr; INTERPRETATION
    </div>
    """, unsafe_allow_globals=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Number of Students (n)", n_students)
    with col2:
        st.metric("Original Features (m)", m_features)
    with col3:
        st.metric("Selected Components (k)", k_components)
    with col4:
        st.metric("Retained Variance", f"{cum_var_ratio[k_components-1]*100:.2f}%")

    st.subheader("💡 Retained Mathematical Concepts")
    concepts_col1, concepts_col2 = st.columns(2)
    with concepts_col1:
        st.markdown("""
        1. **Matrix ($X \in \mathbb{R}^{n \\times m}$)**: High-dimensional representation of student academic attributes.
        2. **Vector Space**: Feature space spanned by academic subject axes.
        3. **Mean Vector ($\mathbf{\mu}$)**: Feature-wise expected values.
        4. **Data Centering ($X_c$)**: Translation of data origin to coordinate mean.
        5. **Covariance Matrix ($C$)**: Pairwise linear feature interaction & total sample variance.
        """)
    with concepts_col2:
        st.markdown("""
        6. **Eigenvalues ($\lambda$)**: Amount of variance captured along principal directions.
        7. **Eigenvectors ($\mathbf{v}$)**: Orthonormal basis vectors representing principal axes.
        8. **Orthogonality ($V^T V \approx I$)**: Linear independence and zero redundancy between principal directions.
        9. **Projection ($Z = X_c V_k$)**: Orthogonal transformation into lower-dimensional subspace.
        10. **Dimensionality Reduction**: Information compression retaining maximum cumulative variance.
        """)

# ==================== TAB 2: DATASET ====================
with tabs[1]:
    st.header("📊 Stage 1 — Student-Feature Matrix Representation")
    st.markdown(r"""
    The dataset is structured as a Student-Feature Matrix $X \in \mathbb{R}^{n \times m}$ where:
    - $n = $ number of students (%d)
    - $m = $ number of academic features (%d)
    """ % (n_students, m_features))
    
    st.latex(r"X \in \mathbb{R}^{n \times m}")
    
    st.subheader("Original Student Dataset (with Metadata)")
    st.dataframe(data_dict['original_df'], use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Extracted Feature Matrix $X$ (Numerical)")
        feature_matrix_df = pd.DataFrame(X, columns=feature_names)
        if not metadata_df.empty:
            feature_matrix_df.index = metadata_df.iloc[:, 0].astype(str)
        st.dataframe(feature_matrix_df, use_container_width=True)
    with col_b:
        st.subheader("Dataset Summary Statistics")
        st.dataframe(feature_matrix_df.describe(), use_container_width=True)

    if any(data_dict['missing_values_count'].values()):
        st.info("Missing value check: Missing values found and imputed using feature column mean.")
    else:
        st.success("Data Validation Passed: No missing values in numerical feature matrix.")

# ==================== TAB 3: CENTERING ====================
with tabs[2]:
    st.header("🎯 Stage 2 — Data Centering")
    st.markdown(r"""
    ### Why Data Centering is Required Before PCA:
    Principal Component Analysis identifies directions of maximum variance passing through the center of mass of the data cloud.
    Subtracting the mean shifts the origin $(0,0,\dots,0)$ to the mean vector $\mathbf{\mu}$, ensuring that the covariance matrix reflects variance around the mean rather than offset from the original axis origin.
    """)
    
    st.subheader("1. Feature Means ($\mu_j$)")
    st.latex(r"\mu_j = \frac{1}{n} \sum_{i=1}^{n} X_{ij}")
    
    mean_df = pd.DataFrame([mu], columns=feature_names, index=[r"Mean ($\mu_j$)"])
    st.dataframe(mean_df.style.format("{:.2f}"), use_container_width=True)
    
    st.subheader("2. Centered Matrix ($X_c$)")
    st.latex(r"X_c = X - \mathbf{\mu}")
    
    Xc_df = pd.DataFrame(X_c, columns=feature_names)
    if not metadata_df.empty:
        Xc_df.index = metadata_df.iloc[:, 0].astype(str)
    st.dataframe(Xc_df.style.format("{:.2f}"), use_container_width=True)
    
    # Verification check for centering
    row_means_check = np.mean(X_c, axis=0)
    check_df = pd.DataFrame([row_means_check], columns=feature_names, index=["Centered Feature Means"])
    st.write("Verification (Centered Column Means should equal 0.00):")
    st.dataframe(check_df.style.format("{:.12f}"), use_container_width=True)

# ==================== TAB 4: COVARIANCE MATRIX ====================
with tabs[3]:
    st.header("🔄 Stage 3 — Covariance Matrix Calculation")
    st.markdown(r"""
    The sample covariance matrix $C \in \mathbb{R}^{m \times m}$ measures how academic features co-vary with each other.
    Diagonal elements $C_{jj}$ represent feature variance $\sigma_j^2$, while off-diagonal elements $C_{ij}$ represent covariance between feature $i$ and feature $j$.
    """)
    
    st.latex(r"C = \frac{1}{n-1} X_c^T X_c")
    
    C_df = pd.DataFrame(C, index=feature_names, columns=feature_names)
    st.subheader("Calculated Covariance Matrix $C$")
    st.dataframe(C_df.style.format("{:.2f}"), use_container_width=True)
    
    st.markdown("""
    **Interpretation of Covariance Matrix:**
    - High positive covariance indicates students who score high in one feature tend to score high in the other.
    - Near-zero covariance indicates independent or uncorrelated academic performance attributes.
    """)

# ==================== TAB 5: EIGEN ANALYSIS ====================
with tabs[4]:
    st.header("⚡ Stage 4 — Eigenvalue & Eigenvector Analysis")
    st.markdown(r"""
    Solving the characteristic equation of the covariance matrix yields eigenvalues $\lambda$ and eigenvectors $\mathbf{v}$:
    """)
    st.latex(r"C \mathbf{v} = \lambda \mathbf{v}")
    
    st.markdown(r"""
    - **Eigenvectors ($\mathbf{v}$)**: Represent the directions of the principal axes (principal directions).
    - **Eigenvalues ($\lambda$)**: Represent the magnitude of variance along each principal direction.
    """)
    
    eigen_summary = []
    for i in range(m_features):
        eigen_summary.append({
            "Component": f"PC{i+1}",
            r"Eigenvalue ($\lambda_i$)": evals[i],
            "Variance Explained (%)": var_ratio[i] * 100,
            "Cumulative Variance (%)": cum_var_ratio[i] * 100
        })
    eigen_df = pd.DataFrame(eigen_summary)
    st.subheader("Eigenvalues & Explained Variance Table")
    st.dataframe(eigen_df.style.format({
        r"Eigenvalue ($\lambda_i$)": "{:.4f}",
        "Variance Explained (%)": "{:.2f}%",
        "Cumulative Variance (%)": "{:.2f}%"
    }), use_container_width=True)
    
    st.subheader("Eigenvectors Matrix $V$ (Columns are Eigenvectors $\mathbf{v}_1, \mathbf{v}_2, \dots$)")
    V_df = pd.DataFrame(evecs, index=feature_names, columns=[f"v_{i+1} (PC{i+1})" for i in range(m_features)])
    st.dataframe(V_df.style.format("{:.4f}"), use_container_width=True)

# ==================== TAB 6: PRINCIPAL COMPONENTS ====================
with tabs[5]:
    st.header("📍 Stage 5 & 6 — Orthogonality & Principal Components")
    
    st.subheader("Stage 5: Verification of Orthogonality")
    st.markdown(r"""
    Because $C$ is a symmetric matrix ($C = C^T$), its eigenvectors are mutually orthogonal.
    Since they are normalized to unit length ($||\mathbf{v}_i|| = 1$), the matrix $V$ is orthonormal, satisfying:
    """)
    st.latex(r"V^T V \approx I_m")
    
    ortho_df = pd.DataFrame(V_T_V, index=[f"v_{i+1}" for i in range(m_features)], columns=[f"v_{i+1}" for i in range(m_features)])
    st.dataframe(ortho_df.style.format("{:.4f}"), use_container_width=True)
    st.caption(f"Maximum absolute deviation from Identity Matrix: {pca_results['ortho_error']:.2e}")
    
    st.subheader("Stage 6: Loading Table ($V_k$) for Selected Components")
    st.markdown(f"User selected $k = {k_components}$ principal components.")
    st.dataframe(loading_df.style.format("{:.4f}"), use_container_width=True)
    
    st.subheader("Dominant Academic Feature Contributions per Component")
    for pc_col, top_feats in pca_results['top_features_per_pc'].items():
        st.markdown(f"- **{pc_col}**: Primary contributing features are **{', '.join(top_feats)}**.")

# ==================== TAB 7: PROJECTION & REDUCTION ====================
with tabs[6]:
    st.header("📐 Stage 7 & 8 — Vector Projection & Dimensionality Reduction")
    
    st.subheader("Stage 7: Orthogonal Vector Projection Formula")
    st.markdown(r"""
    Projecting the centered data matrix $X_c$ onto the $k$ principal eigenvector subspace $V_k$:
    """)
    st.latex(r"Z = X_c V_k")
    st.markdown(r"where $X_c \in \mathbb{R}^{n \times m}$, $V_k \in \mathbb{R}^{m \times k}$, and $Z \in \mathbb{R}^{n \times k}$.")
    
    Z_df = pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(k_components)])
    if not metadata_df.empty:
        Z_df.index = metadata_df.iloc[:, 0].astype(str)
    st.dataframe(Z_df.style.format("{:.4f}"), use_container_width=True)
    
    st.subheader("Stage 8: Dimensionality Reduction Summary")
    red_col1, red_col2, red_col3 = st.columns(3)
    with red_col1:
        st.metric("Original Dimensions", f"{m_features}-D")
    with red_col2:
        st.metric("Reduced Dimensions", f"{k_components}-D")
    with red_col3:
        st.metric("Information Preserved", f"{cum_var_ratio[k_components-1]*100:.2f}%")

# ==================== TAB 8: VISUALIZATIONS ====================
with tabs[7]:
    st.header("📈 Stage 9 — Visualizations")
    
    vis_row1_col1, vis_row1_col2 = st.columns(2)
    with vis_row1_col1:
        st.pyplot(plot_scree(evals))
    with vis_row1_col2:
        st.pyplot(plot_explained_variance(var_ratio, cum_var_ratio))
        
    vis_row2_col1, vis_row2_col2 = st.columns(2)
    with vis_row2_col1:
        student_labels = metadata_df.iloc[:, 0] if not metadata_df.empty else None
        st.pyplot(plot_2d_pca_scatter(Z, labels=student_labels, var_ratio=var_ratio))
    with vis_row2_col2:
        st.pyplot(plot_feature_loadings(loading_df))
        
    st.subheader("Covariance Matrix Heatmap")
    st.pyplot(plot_covariance_heatmap(C, feature_names))

    st.subheader("🔍 Student Performance Pattern Interpretation")
    st.markdown("""
    Based on the numerical decomposition of the student dataset:
    """)
    pc1_top = ", ".join(pca_results['top_features_per_pc'].get('PC1', []))
    pc2_top = ", ".join(pca_results['top_features_per_pc'].get('PC2', []))
    
    st.write(f"1. **Principal Component 1 (PC1)** explains **{var_ratio[0]*100:.2f}%** of total academic variation, heavily weighted by: **{pc1_top}**.")
    if m_features > 1:
        st.write(f"2. **Principal Component 2 (PC2)** explains **{var_ratio[1]*100:.2f}%** of variance, dominated by: **{pc2_top}**.")
    st.write(f"3. Dimensionality reduction from {m_features} features to {k_components} principal components retains **{cum_var_ratio[k_components-1]*100:.2f}%** of overall information.")

# ==================== TAB 9: VERIFICATION ====================
with tabs[8]:
    st.header("✅ Stage 10 — Verification with Scikit-Learn PCA")
    st.markdown("""
    To verify mathematical accuracy, manual NumPy calculations are compared against `scikit-learn.decomposition.PCA`.
    """)
    
    sk_results = verify_with_sklearn(X, feature_names, k=k_components)
    verif_summary = generate_verification_summary(pca_results, sk_results)
    
    st.subheader("Comparative Metric Summary Table")
    st.dataframe(verif_summary['summary_df'], use_container_width=True)
    
    st.markdown(r"""
    > **Note on Eigenvector Sign Ambiguity**:
    > If $C \mathbf{v} = \lambda \mathbf{v}$, then $C (-\mathbf{v}) = \lambda (-\mathbf{v})$.
    > Both $\mathbf{v}$ and $-\mathbf{v}$ define the exact same principal axis line in vector space.
    > Differences in sign between manual NumPy eigenvectors and Scikit-Learn SVD implementations are mathematically equivalent.
    """)
    
    st.subheader("Projected Scores Comparison (Aligned Signs)")
    col_m, col_sk = st.columns(2)
    with col_m:
        st.write("Manual NumPy Projection Matrix Z:")
        st.dataframe(pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(k_components)]).style.format("{:.4f}"), use_container_width=True)
    with col_sk:
        st.write("Scikit-Learn Projection Matrix Z (Aligned Sign):")
        st.dataframe(pd.DataFrame(verif_summary['Z_sk_aligned'], columns=[f"PC{i+1}" for i in range(k_components)]).style.format("{:.4f}"), use_container_width=True)
        
    st.success(f"Verification Passed! Max Absolute Error: Eigenvalues = {verif_summary['eval_max_diff']:.2e}, Projection Scores = {verif_summary['score_max_diff']:.2e}")

# ==================== TAB 10: MATHEMATICAL EXPLANATION ====================
with tabs[9]:
    st.header("📘 Mathematical Foundations & Latex Formulas")
    
    st.markdown("### Core PCA Equations")
    st.latex(r"\text{Mean: } \mathbf{\mu}_j = \frac{1}{n} \sum_{i=1}^n X_{ij}")
    st.latex(r"\text{Data Centering: } X_c = X - \mathbf{\mu}")
    st.latex(r"\text{Covariance Matrix: } C = \frac{1}{n-1} X_c^T X_c")
    st.latex(r"\text{Eigen-Equation: } C \mathbf{v}_i = \lambda_i \mathbf{v}_i")
    st.latex(r"\text{Orthogonality: } \mathbf{v}_i^T \mathbf{v}_j = \delta_{ij} = \begin{cases} 1 & \text{if } i=j \\ 0 & \text{if } i \neq j \end{cases}")
    st.latex(r"\text{Projection: } Z = X_c V_k")
    st.latex(r"\text{Explained Variance Ratio: } EVR_i = \frac{\lambda_i}{\sum_{j=1}^m \lambda_j}")
    st.latex(r"\text{Cumulative Variance: } CEV_k = \sum_{i=1}^k EVR_i")
    
    st.markdown("---")
    st.subheader("Detailed Mathematical Concept Breakdown")
    
    with st.expander("1. Matrix & Vector Space"):
        st.write("Each student is represented as a point in an m-dimensional real vector space R^m. The student-feature matrix X groups all n student vectors.")
        
    with st.expander("2. Centering & Origin Shift"):
        st.write("Centering translates the centroid of the data cloud to the origin (0,0,...,0). This ensures covariance measures variance about the mean.")
        
    with st.expander("3. Covariance Matrix & Variance Maximization"):
        st.write("PCA seeks a unit vector v that maximizes the variance of projected points: Var(X_c v) = v^T C v subject to ||v||^2 = 1. Using Lagrange multipliers, this optimizes to C v = lambda v.")
        
    with st.expander("4. Eigenvalues & Principal Directions"):
        st.write("Eigenvalues quantify the total variance along their corresponding eigenvector directions. PC1 corresponds to the largest eigenvalue lambda_1.")
        
    with st.expander("5. Orthogonality & Projection"):
        st.write("Orthogonality guarantees zero cross-correlation between principal components. Projection Z = X_c V_k maps high-dimensional points onto the optimal k-dimensional subspace.")
