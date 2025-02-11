import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os


def plot_singular_values(S, file_name):
    """
    Plot singular values and their cumulative explained variance
    """
    plt.figure(figsize=(12, 5))

    # Plot singular values
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(S) + 1), S, 'bo-')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Values Distribution')
    plt.grid(True)

    # Plot cumulative explained variance
    plt.subplot(1, 2, 2)
    explained_variance_ratio = (S ** 2) / (S ** 2).sum()
    plt.plot(range(1, len(S) + 1), np.cumsum(explained_variance_ratio), 'ro-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'singular_values_{file_name}.png')
    plt.close()


def plot_feature_contributions(Vt, feature_names, file_name):
    """
    Plot feature contributions to top singular vectors
    """
    n_components = min(3, Vt.shape[0])  # Show top 3 components or less

    plt.figure(figsize=(12, 4 * n_components))
    for i in range(n_components):
        plt.subplot(n_components, 1, i + 1)
        contributions = pd.Series(Vt[i], index=feature_names)
        contributions.sort_values(ascending=True).plot(kind='barh')
        plt.title(f'Feature Contributions to Singular Vector {i + 1}')
        plt.xlabel('Contribution Magnitude')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'feature_contributions_{file_name}.png')
    plt.close()


def plot_singular_matrix_heatmap(S, n_components, file_name):
    """
    Plot heatmap of the singular value matrix (Σ)
    """
    # Create diagonal matrix of singular values
    S_matrix = np.zeros((n_components, n_components))
    np.fill_diagonal(S_matrix, S[:n_components])

    plt.figure(figsize=(10, 8))
    sns.heatmap(S_matrix, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Singular Value Matrix (Σ)')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    plt.tight_layout()
    plt.savefig(f'singular_matrix_{file_name}.png')
    plt.close()


def analyze_file(file_path):
    """
    Load CSV file and perform SVD analysis with visualizations
    """
    # Get file name without extension for titles
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    print(f"\nAnalyzing file: {file_name}")
    print("-" * 50)

    # Read data
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # Perform SVD
    print("\nPerforming SVD analysis...")
    U, S, Vt = np.linalg.svd(data_scaled, full_matrices=False)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Singular values plot
    plot_singular_values(S, file_name)

    # 2. Feature contributions plot
    plot_feature_contributions(Vt, df.columns, file_name)

    # 3. Singular matrix heatmap
    n_components = min(5, len(S))  # Show top 5 components or less
    plot_singular_matrix_heatmap(S, n_components, file_name)

    # Save component compositions
    component_df = pd.DataFrame(
        Vt[:n_components].T,
        columns=[f'Singular Vector {i + 1}' for i in range(n_components)],
        index=df.columns
    )
    component_df.to_csv(f'svd_components_{file_name}.csv')

    # Print summary
    print("\nAnalysis complete! Generated files:")
    print(f"1. singular_values_{file_name}.png - Singular values distribution")
    print(f"2. feature_contributions_{file_name}.png - Feature contributions")
    print(f"3. singular_matrix_{file_name}.png - Singular value matrix heatmap")
    print(f"4. svd_components_{file_name}.csv - Component compositions")


if __name__ == "__main__":
    # File path - replace with your file path
    file_path = "PK_NEv_fault_datasetnew20250125.csv"  # Update with your file path
    analyze_file(file_path)