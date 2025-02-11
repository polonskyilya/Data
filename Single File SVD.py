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
    Plot feature contributions for all singular vectors
    """
    n_vectors = Vt.shape[0]

    # Calculate how many subplots we need (arrange in multiple columns if too many)
    n_cols = min(3, n_vectors)
    n_rows = (n_vectors + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))

    for i in range(n_vectors):
        plt.subplot(n_rows, n_cols, i + 1)
        contributions = pd.Series(Vt[i], index=feature_names)
        contributions.sort_values(ascending=True).plot(kind='barh')
        plt.title(f'Feature Contributions to Singular Vector {i + 1}')
        plt.xlabel('Contribution Magnitude')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'feature_contributions_{file_name}.png')
    plt.close()


def plot_variance_table(S, file_name):
    """
    Create and save variance explanation table
    """
    squared_values = S ** 2
    total_variance = squared_values.sum()
    explained_variance = squared_values / total_variance * 100
    cumulative_variance = np.cumsum(explained_variance)

    # Create table data
    data = {
        'Singular Value': S,
        'Squared Value': squared_values,
        'Variance Explained (%)': explained_variance,
        'Cumulative Variance (%)': cumulative_variance
    }

    # Create a figure and axis
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=[[f"{row[0]:.2f}", f"{row[1]:.2f}", f"{row[2]:.2f}%", f"{row[3]:.2f}%"]
                  for row in zip(data['Singular Value'],
                                 data['Squared Value'],
                                 data['Variance Explained (%)'],
                                 data['Cumulative Variance (%)'])],
        colLabels=['σᵢ', 'σᵢ²', 'Variance Explained (%)', 'Cumulative Variance (%)'],
        rowLabels=[f"Vector {i + 1}" for i in range(len(S))],
        loc='center'
    )

    # Modify table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Add title with formula
    plt.title('Variance Explanation Table\nFormula: Variance Explained (%) = (σᵢ² / Σσᵢ²) × 100%',
              pad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(f'variance_table_{file_name}.png', bbox_inches='tight', dpi=300)
    plt.close()


def plot_singular_matrix_heatmap(S, file_name):
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
    plt.savefig(f'singular_matrix_{file_name}.png')
    plt.close()


def analyze_file(file_path):
    """
    Load CSV file and perform SVD analysis with visualizations for all components
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

    # Print explained variance information
    explained_variance_ratio = (S ** 2) / (S ** 2).sum()
    cumulative_variance = np.cumsum(explained_variance_ratio)
    print(f"\nNumber of singular values: {len(S)}")
    print("\nExplained variance by component:")
    for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance), 1):
        print(f"Component {i}: {var:.4f} ({cum_var:.4f} cumulative)")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Variance explanation table
    plot_variance_table(S, file_name)

    # 2. Singular values plot
    plot_singular_values(S, file_name)

    # 2. Feature contributions plot (all vectors)
    plot_feature_contributions(Vt, df.columns, file_name)

    # 3. Complete singular matrix heatmap
    plot_singular_matrix_heatmap(S, file_name)

    # Save all component compositions
    component_df = pd.DataFrame(
        Vt.T,
        columns=[f'Singular Vector {i + 1}' for i in range(len(S))],
        index=df.columns
    )
    component_df.to_csv(f'svd_components_{file_name}.csv')

    # Print summary
    print("\nAnalysis complete! Generated files:")
    print(f"1. singular_values_{file_name}.png - Complete singular values distribution")
    print(f"2. feature_contributions_{file_name}.png - All feature contributions")
    print(f"3. singular_matrix_{file_name}.png - Complete singular value matrix")
    print(f"4. svd_components_{file_name}.csv - All component compositions")


if __name__ == "__main__":
    # File path - replace with your file path
    file_path = r"C:\Users\Ilya Polonsky\Downloads\CSV_3_Fault_Label_Filtered Anomaly.csv"  # Update with your file path
    analyze_file(file_path)