import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os


def ensure_svd_directory():
    """
    Create SVD directory if it doesn't exist
    """
    svd_dir = "SVD"
    if not os.path.exists(svd_dir):
        os.makedirs(svd_dir)
    return svd_dir


def plot_singular_values(S, file_name, svd_dir):
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
    plt.savefig(os.path.join(svd_dir, f'singular_values_{file_name}.png'), bbox_inches='tight', dpi=300)
    plt.close()


def plot_feature_contributions(Vt, feature_names, file_name, svd_dir):
    """
    Plot feature contributions for all singular vectors
    """
    n_vectors = Vt.shape[0]
    n_cols = 3
    n_rows = (n_vectors + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(20, 6 * n_rows))

    for i in range(n_vectors):
        ax = plt.subplot(n_rows, n_cols, i + 1)

        # Create contributions series
        contributions = pd.Series(Vt[i], index=feature_names)
        sorted_idx = contributions.abs().sort_values(ascending=True).index
        contributions_sorted = contributions[sorted_idx]

        # Create horizontal bar plot with colors
        colors = ['red' if x < 0 else 'blue' for x in contributions_sorted.values]
        bars = ax.barh(range(len(contributions_sorted)),
                       contributions_sorted.values,
                       color=colors,
                       alpha=0.6)

        # Customize plot
        ax.set_yticks(range(len(contributions_sorted)))
        ax.set_yticklabels(contributions_sorted.index, fontsize=8)
        ax.set_title(f'Feature Contributions to Singular Vector {i + 1}')
        ax.set_xlabel('Contribution Value')
        ax.grid(True)

        # Add value labels
        for j, v in enumerate(contributions_sorted):
            label_pos = v + (0.01 if v >= 0 else -0.01)
            ax.text(label_pos, j, f'{v:.3f}',
                    va='center',
                    ha='left' if v >= 0 else 'right',
                    fontsize=8)

        # Adjust limits to prevent cutting
        xmin, xmax = ax.get_xlim()
        margin = (xmax - xmin) * 0.15
        ax.set_xlim(xmin - margin, xmax + margin)

    plt.tight_layout()
    plt.savefig(os.path.join(svd_dir, f'feature_contributions_{file_name}.png'),
                bbox_inches='tight',
                dpi=300,
                facecolor='white',
                edgecolor='none')
    plt.close()


def plot_singular_matrix_heatmap(S, file_name, svd_dir):
    """
    Plot heatmap of the complete singular value matrix (Σ)
    """
    n_components = len(S)
    S_matrix = np.zeros((n_components, n_components))
    np.fill_diagonal(S_matrix, S)

    plt.figure(figsize=(12, 10))
    sns.heatmap(S_matrix, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Complete Singular Value Matrix (Σ)')

    # Create labels that show both index and value
    labels = [f'SV {i + 1}\n(σ={S[i]:.2f})' for i in range(n_components)]
    plt.xticks(np.arange(n_components) + 0.5, labels, rotation=45)
    plt.yticks(np.arange(n_components) + 0.5, labels, rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(svd_dir, f'singular_matrix_{file_name}.png'),
                bbox_inches='tight',
                dpi=300,
                facecolor='white')
    plt.close()


def analyze_file(file_path):
    """
    Load CSV file and perform SVD analysis with visualizations
    """
    # Create SVD directory
    svd_dir = ensure_svd_directory()

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

    # Calculate variance explained
    variance_explained = (S ** 2) / (S ** 2).sum() * 100
    cumulative_variance = np.cumsum(variance_explained)

    print("\nVariance explained by each component:")
    for i, (var, cum_var) in enumerate(zip(variance_explained, cumulative_variance), 1):
        print(f"Vector {i}: {var:.2f}% ({cum_var:.2f}% cumulative)")

    # Generate visualizations
    print("\nGenerating visualizations in SVD directory...")

    # 1. Singular values plot
    plot_singular_values(S, file_name, svd_dir)

    # 2. Feature contributions plot
    plot_feature_contributions(Vt, df.columns, file_name, svd_dir)

    # 3. Singular matrix heatmap
    plot_singular_matrix_heatmap(S, file_name, svd_dir)

    # Save V^T matrix (feature patterns)
    vt_df = pd.DataFrame(
        Vt.T,
        columns=[f'Singular Vector {i + 1}' for i in range(len(S))],
        index=df.columns
    )
    vt_df.to_csv(os.path.join(svd_dir, f'svd_components_V_{file_name}.csv'))

    # Save U matrix (sample patterns)
    u_df = pd.DataFrame(
        U,
        columns=[f'Component {i + 1}' for i in range(U.shape[1])]
    )
    # Add index to track original sample numbers
    u_df.index.name = 'Sample_ID'
    u_df.to_csv(os.path.join(svd_dir, f'svd_components_U_{file_name}.csv'))

    # Save S values (singular values)
    s_df = pd.DataFrame({
        'Singular_Value': S,
        'Variance_Explained': variance_explained,
        'Cumulative_Variance': cumulative_variance
    })
    s_df.index = [f'Component_{i + 1}' for i in range(len(S))]
    s_df.to_csv(os.path.join(svd_dir, f'svd_values_S_{file_name}.csv'))

    print("\nAnalysis complete! Generated files in SVD directory:")
    print(f"1. singular_values_{file_name}.png - Singular values distribution")
    print(f"2. feature_contributions_{file_name}.png - Feature contributions")
    print(f"3. singular_matrix_{file_name}.png - Singular value matrix")
    print(f"4. svd_components_V_{file_name}.csv - V^T matrix (feature patterns)")
    print(f"5. svd_components_U_{file_name}.csv - U matrix (sample patterns)")
    print(f"6. svd_values_S_{file_name}.csv - S values and variance explained")


if __name__ == "__main__":
    file_path = r"C:\Users\Ilya Polonsky\PycharmProjects\Data\CSV_DATA_SETS\Fault_6_CSV_minor_secondary_timeseries.csv"
    # Update with your file path
    analyze_file(file_path)