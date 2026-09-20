import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional

# Set consistent matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_scree(eigenvalues: np.ndarray) -> plt.Figure:
    """
    Creates Scree Plot: Eigenvalue vs Principal Component index.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    components = np.arange(1, len(eigenvalues) + 1)
    
    ax.plot(components, eigenvalues, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='Eigenvalue')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Kaiser Criterion ($\lambda=1$)')
    
    ax.set_title('Scree Plot: Eigenvalues per Principal Component', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Principal Component (PC Index)', fontsize=12, labelpad=10)
    ax.set_ylabel('Eigenvalue ($\lambda_i$)', fontsize=12, labelpad=10)
    ax.set_xticks(components)
    ax.set_xticklabels([f"PC{i}" for i in components])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    return fig

def plot_explained_variance(variance_ratio: np.ndarray, cum_variance_ratio: np.ndarray) -> plt.Figure:
    """
    Creates Individual and Cumulative Explained Variance plot.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    components = np.arange(1, len(variance_ratio) + 1)
    
    # Bar plot for individual variance
    bars = ax1.bar(components, variance_ratio * 100, alpha=0.6, color='#2ca02c', label='Individual Variance (%)')
    ax1.set_xlabel('Principal Component', fontsize=12, labelpad=10)
    ax1.set_ylabel('Individual Explained Variance (%)', fontsize=12, color='#2ca02c', labelpad=10)
    ax1.set_xticks(components)
    ax1.set_xticklabels([f"PC{i}" for i in components])
    ax1.set_ylim(0, max(variance_ratio * 100) * 1.2)
    
    # Line plot for cumulative variance
    ax2 = ax1.twinx()
    line = ax2.plot(components, cum_variance_ratio * 100, 'ro-', linewidth=2.5, markersize=7, label='Cumulative Variance (%)')
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12, color='red', labelpad=10)
    ax2.set_ylim(0, 105)
    ax2.grid(False)
    
    # Annotate values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    plt.title('Explained Variance Ratio (Individual & Cumulative)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig

def plot_2d_pca_scatter(
    Z: np.ndarray, 
    labels: Optional[pd.Series] = None, 
    var_ratio: Optional[np.ndarray] = None
) -> plt.Figure:
    """
    Creates 2D PCA Scatter plot (PC1 vs PC2).
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    pc1_var = f" ({var_ratio[0]*100:.1f}%)" if var_ratio is not None and len(var_ratio) > 0 else ""
    pc2_var = f" ({var_ratio[1]*100:.1f}%)" if var_ratio is not None and len(var_ratio) > 1 else ""
    
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c='#1f77b4', s=90, alpha=0.85, edgecolors='black', linewidths=1.2)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Annotate points with student labels if provided
    if labels is not None:
        for i, txt in enumerate(labels):
            ax.annotate(str(txt), (Z[i, 0], Z[i, 1]), xytext=(6, 6), textcoords='offset points', fontsize=9, alpha=0.9)
            
    ax.set_title('2D Projection of Students in Principal Subspace ($PC_1$ vs $PC_2$)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(f'Principal Component 1 (PC1){pc1_var}', fontsize=12, labelpad=10)
    ax.set_ylabel(f'Principal Component 2 (PC2){pc2_var}', fontsize=12, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_feature_loadings(loading_df: pd.DataFrame) -> plt.Figure:
    """
    Creates Bar Chart showing feature loadings for PC1 and PC2.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    cols_to_plot = [c for c in ['PC1', 'PC2'] if c in loading_df.columns]
    plot_df = loading_df[cols_to_plot]
    
    plot_df.plot(kind='bar', ax=ax, width=0.7, colormap='viridis', edgecolor='black')
    
    ax.set_title('Academic Feature Loadings on Principal Components', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Academic Features', fontsize=12, labelpad=10)
    ax.set_ylabel('Eigenvector Component (Loading Weight)', fontsize=12, labelpad=10)
    ax.set_xticklabels(loading_df.index, rotation=35, ha='right', fontsize=10)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Component', frameon=True)
    
    plt.tight_layout()
    return fig

def plot_covariance_heatmap(C: np.ndarray, feature_names: list) -> plt.Figure:
    """
    Creates Heatmap of the Covariance Matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        C, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        xticklabels=feature_names, 
        yticklabels=feature_names,
        ax=ax,
        cbar_kws={'label': 'Covariance'}
    )
    
    ax.set_title('Covariance Matrix Heatmap ($C = \\frac{1}{n-1} X_c^T X_c$)', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    return fig
