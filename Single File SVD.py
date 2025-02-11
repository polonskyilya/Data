import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os


def clean_numeric_data(value):
    """
    Clean numeric strings by handling European number format and scientific notation
    """
    if isinstance(value, str):
        value = value.replace(',', '.')
        if 'E' in value.upper():
            try:
                return float(value.replace('E', 'e'))
            except ValueError:
                return np.nan
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value


def plot_feature_distributions(df, file_name):
    """
    Plot distributions for all features
    """
    n_features = len(df.columns)
    n_rows = (n_features + 2) // 3  # Ceiling division by 3

    plt.figure(figsize=(15, 4 * n_rows))
    for i, column in enumerate(df.columns, 1):
        plt.subplot(n_rows, 3, i)
        sns.histplot(data=df, x=column, kde=True)
        plt.title(f'{column} Distribution')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'distributions_{file_name}.png')
    plt.close()


def plot_pair_correlations(df, file_name, top_n=5):
    """
    Plot pairwise correlations for top correlated features
    """
    # Get top correlated features
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    top_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                 for i, j in zip(*np.where(upper > 0.5))]
    top_pairs = sorted(top_pairs, key=lambda x: x[2], reverse=True)[:top_n]

    if top_pairs:
        # Create scatter plots for top pairs
        n_pairs = len(top_pairs)
        plt.figure(figsize=(15, 5 * ((n_pairs + 2) // 3)))
        for i, (feat1, feat2, corr) in enumerate(top_pairs, 1):
            plt.subplot((n_pairs + 2) // 3, 3, i)
            sns.scatterplot(data=df, x=feat1, y=feat2, alpha=0.5)
            plt.title(f'Correlation: {corr:.2f}')
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'pair_correlations_{file_name}.png')
        plt.close()


def plot_component_analysis(df, file_name):
    """
    Plot PCA component analysis visualizations
    """
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # Plot explained variance ratio
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio')
    plt.grid(True)

    # Plot first two components
    plt.subplot(2, 2, 2)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('First Two Principal Components')
    plt.grid(True)

    # Plot component weights
    plt.subplot(2, 2, 3)
    component_weights = pd.DataFrame(
        pca.components_[:2].T,
        columns=['First Component', 'Second Component'],
        index=df.columns
    )
    sns.heatmap(component_weights, cmap='coolwarm', center=0, annot=True, fmt='.2f')
    plt.title('Component Weights')

    plt.tight_layout()
    plt.savefig(f'component_analysis_{file_name}.png')
    plt.close()


def analyze_file(file_path):
    """
    Load CSV file and perform comprehensive analysis with visualizations
    """
    # Get file name without extension for titles
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    print(f"\nAnalyzing file: {file_name}")
    print("-" * 50)

    # Read and clean data
    df = pd.read_csv(file_path)
    print(f"Original data shape: {df.shape}")

    for column in df.columns:
        df[column] = df[column].apply(clean_numeric_data)

    df = df.dropna()
    print(f"Data shape after cleaning: {df.shape}")

    # Generate all visualizations
    print("\nGenerating visualizations...")

    # 1. Correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'Correlation Matrix - {file_name}')
    plt.tight_layout()
    plt.savefig(f'correlation_matrix_{file_name}.png')
    plt.close()

    # 2. Feature distributions
    plot_feature_distributions(df, file_name)

    # 3. Pair correlations
    plot_pair_correlations(df, file_name)

    # 4. Component analysis
    plot_component_analysis(df, file_name)

    # Perform SVD
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    U, S, Vt = np.linalg.svd(data_scaled, full_matrices=False)

    # Calculate and print key findings
    explained_variance_ratio = (S ** 2) / (S ** 2).sum()
    n_components = 3

    component_df = pd.DataFrame(
        Vt[:n_components].T,
        columns=[f'Component {i + 1}' for i in range(n_components)],
        index=df.columns
    )

    # Save results
    correlation_matrix.to_csv(f'correlation_matrix_{file_name}.csv')
    component_df.to_csv(f'component_composition_{file_name}.csv')

    # Print summary
    print("\nAnalysis complete! Generated files:")
    print(f"1. correlation_matrix_{file_name}.png - Correlation heatmap")
    print(f"2. distributions_{file_name}.png - Feature distributions")
    print(f"3. pair_correlations_{file_name}.png - Top correlated pairs")
    print(f"4. component_analysis_{file_name}.png - PCA analysis")
    print(f"5. correlation_matrix_{file_name}.csv - Detailed correlations")
    print(f"6. component_composition_{file_name}.csv - Component compositions")


if __name__ == "__main__":
    # File path - replace with your file path
    file_path = r"C:\Users\Ilya Polonsky\Downloads\WO_FaultLabel_CSV_data_set_minor_secondary_Filtered Anomaly.csv"
    analyze_file(file_path)