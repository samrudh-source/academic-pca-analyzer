# Academic Performance Principal Pattern Analyzer
### *Linear Algebra Based Student Performance Analysis using PCA*

---

## 1. Project Title
**Academic Performance Principal Pattern Analyzer**

---

## 2. Aim
To develop an interactive Python application that analyzes multidimensional student academic performance data and identifies dominant patterns of variation using Principal Component Analysis (PCA) built explicitly from fundamental Linear Algebra and Vector Calculus principles.

---

## 3. Problem Statement
Academic institutions capture student performance across multiple subjects and evaluation criteria (e.g., Mathematics, Physics, Chemistry, Programming, Attendance, Internal Assessments). Analyzing raw high-dimensional academic tables makes it difficult to detect underlying patterns, redundant evaluations, or primary skill drivers. High dimensionality also obscures student grouping and comparative performance. 

This project solves this problem by projecting $m$-dimensional student performance vectors onto a lower-dimensional orthogonal principal subspace ($k$-dimensions) while preserving maximum cumulative variance, demonstrating step-by-step matrix operations without hiding calculations behind black-box library calls.

---

## 4. Objectives
1. **Mathematical Demonstration**: Rigorously illustrate 10 core linear algebra concepts sequentially: Matrix, Vector Space, Mean/Centering, Covariance Matrix, Eigenvalues, Eigenvectors, Orthogonality, Projection, Principal Components, and Dimensionality Reduction.
2. **Explicit Computational Workflow**: Calculate each PCA stage manually using NumPy matrix algebra such that the output of each stage serves directly as the input to the next.
3. **Interactive Dashboard**: Build a Streamlit web interface allowing CSV upload, parameter adjustment, tabbed exploration, and visual plotting.
4. **Scikit-Learn Verification**: Compare manual matrix calculations against `scikit-learn.decomposition.PCA` to verify numerical precision and explain sign orientation properties.
5. **Academic Viva Preparedness**: Provide comprehensive documentation and viva voce Q&A suitable for B.Tech university evaluation.

---

## 5. Mathematical Concepts Used
- **Matrix Representation**: $X \in \mathbb{R}^{n \times m}$
- **Vector Space & Subspaces**: Span of feature basis vectors in $\mathbb{R}^m$
- **Mean & Data Centering**: Translation of origin to data centroid $\boldsymbol{\mu}$
- **Covariance Matrix**: Pairwise linear feature interaction matrix $C \in \mathbb{R}^{m \times m}$
- **Eigenvalues ($\lambda$) & Eigenvectors ($\mathbf{v}$)**: Spectral decomposition satisfying $C \mathbf{v} = \lambda \mathbf{v}$
- **Orthogonality**: Orthonormal basis verification $V^T V \approx I_m$
- **Vector Projection**: Orthogonal projection operator $Z = X_c V_k$
- **Principal Components**: Ordered orthogonal directions of maximum variance
- **Dimensionality Reduction**: Truncation from $\mathbb{R}^m \to \mathbb{R}^k$ ($k < m$)
- **Variance Ratio**: Quantifying retained information percentage ($\frac{\lambda_i}{\sum \lambda}$)

---

## 6. Mathematical Formulation

### Stage 1: Feature Matrix
Let $X$ be an $n \times m$ matrix where $n$ is the number of students and $m$ is the number of academic features:
$$X \in \mathbb{R}^{n \times m}$$

### Stage 2: Feature Mean Vector & Data Centering
For each column $j \in \{1, \dots, m\}$:
$$\mu_j = \frac{1}{n} \sum_{i=1}^{n} X_{ij}$$

The centered matrix $X_c$ is obtained by subtracting the mean vector $\boldsymbol{\mu}$:
$$X_c = X - \boldsymbol{\mu}$$

### Stage 3: Covariance Matrix
The sample covariance matrix $C$ is computed manually as:
$$C = \frac{1}{n-1} X_c^T X_c$$

### Stage 4: Eigen-Decomposition
Eigenvalues $\lambda_i$ and eigenvectors $\mathbf{v}_i$ satisfy:
$$C \mathbf{v}_i = \lambda_i \mathbf{v}_i$$
Eigenvalues are sorted in descending order: $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_m \ge 0$.

### Stage 5: Orthogonality Verification
Since $C$ is real and symmetric ($C = C^T$), eigenvectors are orthogonal:
$$\mathbf{v}_i^T \mathbf{v}_j = 0 \quad \forall i \neq j$$
For normalized eigenvectors ($||\mathbf{v}_i|| = 1$), the eigenvector matrix $V$ satisfies:
$$V^T V = I_m$$

### Stage 6 & 7: Component Selection & Projection
Selecting the top $k$ eigenvectors forms $V_k \in \mathbb{R}^{m \times k}$.
The projected scores $Z \in \mathbb{R}^{n \times k}$ are calculated via matrix multiplication:
$$Z = X_c V_k$$

### Stage 8: Variance Explained Ratio
$$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^m \lambda_j}, \quad \text{CEV}_k = \sum_{i=1}^k \text{EVR}_i$$

---

## 7. Sequential Computational Workflow
```
STUDENT DATA (CSV / Upload)
      │
      ▼
STUDENT-FEATURE MATRIX (X ∈ ℝ^(n×m))
      │
      ▼
DATA CENTERING (Xc = X - μ)
      │
      ▼
COVARIANCE MATRIX (C = 1/(n-1) Xcᵀ Xc)
      │
      ▼
EIGENVALUE / EIGENVECTOR DECOMPOSITION (C v = λ v)
      │
      ▼
ORTHOGONAL PRINCIPAL DIRECTIONS (V^T V ≈ I)
      │
      ▼
ORTHOGONAL PROJECTION (Z = Xc V_k)
      │
      ▼
REDUCED REPRESENTATION & INTERPRETATION (k-D Subspace)
```

---

## 8. Technology Used
- **Python 3.9+**
- **NumPy**: Linear algebra, matrix operations, matrix multiplication, eigen-decomposition (`np.linalg.eigh`).
- **Pandas**: CSV handling, DataFrame structure, statistics.
- **Matplotlib & Seaborn**: Scree plot, loading plots, 2D PCA scatter plot, heatmaps.
- **Streamlit**: Web dashboard UI and interactive controls.
- **Scikit-Learn**: Used strictly in the Verification tab for comparison.
- **Pytest**: Unit testing framework.

---

## 9. Project Structure
```
academic-pca-analyzer/
│
├── app.py                     # Main Streamlit Dashboard Application
├── requirements.txt           # Python dependencies
├── README.md                  # Detailed Academic Documentation & Viva Q&A
├── data/
│   └── sample_students.csv    # Default 15-student academic dataset
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_loader.py         # CSV loading & data validation
│   ├── preprocessing.py       # Data centering & mean vector calculations
│   ├── pca_calculator.py      # Manual PCA pipeline & linear algebra
│   ├── visualization.py       # Matplotlib & Seaborn plotting functions
│   └── verification.py        # Scikit-learn comparison & sign alignment
│
└── tests/
    └── test_pca.py            # Pytest suite for linear algebra steps
```

---

## 10. Installation Steps

### Step 1: Open Terminal in Project Directory
Navigate to the root directory `academic-pca-analyzer`:
```bash
cd academic-pca-analyzer
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 11. How to Run

### Run Unit Tests (Pytest)
```bash
python -m pytest
```

### Run Streamlit Web Dashboard
```bash
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 12. Sample Input Format
The CSV input file must contain numeric columns representing academic performance features. Metadata columns (such as `Student_ID` or `Student_Name`) are automatically preserved as labels.

Example (`sample_students.csv`):
```csv
Student_ID,Student_Name,Mathematics,Physics,Chemistry,Programming,Data_Structures,Attendance,Internal_Assessment
S101,Aarav Sharma,88,82,79,95,92,94,46
S102,Ananya Patel,92,90,88,84,81,96,48
S103,Rohan Verma,55,58,62,70,68,78,32
```

---

## 13. Expected Output
1. **Interactive Matrix Displays**: Mean vector $\boldsymbol{\mu}$, Centered matrix $X_c$, Covariance matrix $C$, Eigenvalues $\boldsymbol{\lambda}$, Eigenvectors $V$, Orthogonality matrix $V^T V$, Loading matrix $V_k$, Projected scores $Z$.
2. **5 Matplotlib Charts**: Scree Plot, Explained Variance Plot, 2D PCA Scatter Plot, Feature Loading Bar Chart, Covariance Heatmap.
3. **Scikit-Learn Verification Table**: Numerical diff showing exact match ($\text{diff} < 10^{-5}$) between manual and library calculations.

---

## 14. Verification Methodology
Manual NumPy calculations are verified against `sklearn.decomposition.PCA(n_components=k)`:
1. **Eigenvalues**: Verified that manual eigenvalues $\lambda_i$ match `pca.explained_variance_`.
2. **Variance Ratio**: Verified that manual ratio matches `pca.explained_variance_ratio_`.
3. **Sign Alignment**: Because eigenvectors $\mathbf{v}$ and $-\mathbf{v}$ both satisfy $C \mathbf{v} = \lambda \mathbf{v}$, dot-product sign checking aligns projected scores before computing maximum absolute error.

---

## 15. Results and Interpretation
- **PC1 (Dominant Component)**: Explains the highest percentage of variance (typically ~60-70%). It represents overall academic proficiency combining core subjects and problem-solving skills.
- **PC2 (Secondary Component)**: Explains secondary variance (typically ~15-25%), often highlighting contrasts between theoretical core sciences (Physics/Math) vs practical computing (Programming/Data Structures).
- **Dimensionality Reduction**: Retains >85% of total variance while reducing dimensions from 7 features down to 2 principal components.

---

## 16. Limitations
1. **Linearity Assumption**: PCA only captures linear feature relationships. Non-linear manifolds require non-linear techniques (e.g., Kernel PCA or t-SNE).
2. **Sensitivity to Outliers**: Extreme outlier scores can heavily skew feature means and covariance calculations.
3. **Scale Dependency**: Features with much larger variances dominate PCA unless standardized (z-score scaling).

---

## 17. Future Enhancements
1. Add Standard Scaling (Z-score normalization) option for datasets with disparate feature units.
2. Implement 3D Interactive Scatter Plot using Plotly.
3. Support Kernel PCA for non-linear student performance pattern analysis.

---

## 18. University Viva Questions and Answers

### Q1: What is Principal Component Analysis (PCA)?
**Answer**: PCA is an unsupervised linear dimensionality reduction technique that transforms a set of correlated numerical features into a smaller set of uncorrelated orthogonal variables called Principal Components, while retaining maximum variance.

### Q2: Why is Data Centering mandatory before PCA?
**Answer**: Centering shifts the origin of the coordinate system to the mean vector $\boldsymbol{\mu}$ of the dataset. Without centering, the first principal component would point towards the mean of the data cloud rather than along the direction of maximum variance.

### Q3: What is the significance of the Covariance Matrix in PCA?
**Answer**: The covariance matrix $C = \frac{1}{n-1} X_c^T X_c$ summarizes all pairwise linear relationships and variances between features. The eigenvectors of $C$ yield the principal component directions, and the eigenvalues quantify variance along those directions.

### Q4: Why are eigenvectors of the Covariance Matrix orthogonal?
**Answer**: According to the Spectral Theorem in linear algebra, any real symmetric matrix (such as the covariance matrix $C = C^T$) has real eigenvalues and mutually orthogonal eigenvectors.

### Q5: What does $C \mathbf{v} = \lambda \mathbf{v}$ mean geometrically?
**Answer**: When matrix $C$ multiplies eigenvector $\mathbf{v}$, the direction of vector $\mathbf{v}$ does not change; it is merely scaled by factor $\lambda$ (the eigenvalue). Geometrically, $\mathbf{v}$ is an axis of inertia of the data distribution.

### Q6: How do you choose the number of principal components $k$?
**Answer**: $k$ is selected based on the Scree Plot elbow point, Kaiser criterion ($\lambda \ge 1$), or by choosing $k$ such that the cumulative explained variance ratio reaches a target threshold (e.g., 85% or 90%).

### Q7: What is the difference between Eigenvectors and Feature Loadings?
**Answer**: Eigenvectors are unit vectors ($||\mathbf{v}|| = 1$) representing principal directions. Feature loadings are the coefficients of the eigenvectors, indicating how strongly each original feature contributes to a given principal component.

### Q8: What is vector projection in PCA?
**Answer**: Vector projection maps high-dimensional centered data points $X_c \in \mathbb{R}^{n \times m}$ onto the subspace spanned by the selected eigenvectors $V_k \in \mathbb{R}^{m \times k}$ via matrix multiplication $Z = X_c V_k$.

### Q9: Why might manual PCA eigenvectors differ in sign from `scikit-learn` PCA?
**Answer**: Eigenvectors are defined up to a sign flip: if $\mathbf{v}$ is an eigenvector, $-\mathbf{v}$ is also a valid eigenvector for the same eigenvalue $\lambda$. Different algorithms (e.g. SVD vs Eigh) may select opposite sign conventions.

### Q10: What is the total variance of the dataset?
**Answer**: The total variance equals the sum of the variances of individual features (trace of covariance matrix $\text{Tr}(C)$), which is also equal to the sum of all eigenvalues $\sum_{i=1}^m \lambda_i$.

### Q11: What is the Explained Variance Ratio?
**Answer**: The Explained Variance Ratio of component $i$ is $\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^m \lambda_j}$, representing the fraction of total dataset variance captured by $PC_i$.

### Q12: Can PCA be used for categorical features?
**Answer**: No. Standard PCA requires continuous numerical distance metrics to compute means and covariances. Categorical variables require techniques like Multiple Correspondence Analysis (MCA).

### Q13: What is the matrix dimension of the projected score matrix $Z$?
**Answer**: If $X_c$ has shape $n \times m$ and $V_k$ has shape $m \times k$, the projected score matrix $Z = X_c V_k$ has dimension $n \times k$.

### Q14: How does PCA handle multicollinearity between features?
**Answer**: PCA eliminates multicollinearity completely because principal components are mutually orthogonal, resulting in zero covariance between projected variables.

### Q15: What is the time complexity of PCA via Eigen-decomposition?
**Answer**: Computing the covariance matrix takes $O(n \cdot m^2)$, and eigen-decomposition of an $m \times m$ matrix takes $O(m^3)$. Overall complexity is $O(n m^2 + m^3)$.
