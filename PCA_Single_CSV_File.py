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


def perform_pca_analysis(file_path):
    """
    Perform PCA analysis and create visualizations for a single CSV file
    """
    # Get file name for output files
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\nPerforming PCA analysis for {file_name}")
    print("-" * 50)

    # Read and clean data
    df = pd.read_csv(file_path)
    print(f"Original data shape: {df.shape}")

    # Clean data
    for column in df.columns:
        df[column] = df[column].apply(clean_numeric_data)
    df = df.dropna()
    print(f"Data shape after cleaning: {df.shape}")
    print("\nColumns in dataset:")
    print(df.columns.tolist())

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)

    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Print explained variance for each component
    print("\nExplained variance ratio by component:")
    for i, var in enumerate(explained_variance_ratio, 1):
        print(f"PC{i}: {var:.4f} ({cumulative_variance_ratio[i - 1]:.4f} cumulative)")

    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print(f"\nNumber of components needed for 95% variance: {n_components_95}")

    # 1. Scree plot and cumulative variance
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(explained_variance_ratio) + 1),
             explained_variance_ratio, 'bo-')
    plt.title(f'Scree Plot - {file_name}')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1),
             cumulative_variance_ratio, 'ro-')
    plt.axhline(y=0.95, color='k', linestyle='--', label='95% Threshold')
    plt.title(f'Cumulative Explained Variance - {file_name}')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'pca_variance_{file_name}.png')
    plt.close()

    # 2. Component loadings heatmap
    n_components_plot = min(5, len(df.columns))  # Show up to first 5 components
    loadings = pd.DataFrame(
        pca.components_[:n_components_plot].T,
        columns=[f'PC{i + 1}' for i in range(n_components_plot)],
        index=df.columns
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'PCA Component Loadings - {file_name}')
    plt.tight_layout()
    plt.savefig(f'pca_loadings_{file_name}.png')
    plt.close()

    # 3. Biplot of first two components
    plt.figure(figsize=(12, 8))

    # Plot scores
    scores = pca_result[:, 0:2]
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.5)

    # Plot loadings
    loadings_scaled = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
    for i, (x, y) in enumerate(loadings_scaled):
        plt.arrow(0, 0, x, y, color='r', alpha=0.5)
        plt.text(x * 1.2, y * 1.2, df.columns[i], color='r')

    plt.title(f'PCA Biplot - {file_name}')
    plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%} explained variance)')
    plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2%} explained variance)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'pca_biplot_{file_name}.png')
    plt.close()

    # Save loadings to CSV
    loadings.to_csv(f'pca_loadings_{file_name}.csv')

    # Print key findings
    print("\nKey findings:")
    print(f"- First component explains {explained_variance_ratio[0]:.2%} of variance")
    print(f"- First two components explain {explained_variance_ratio[:2].sum():.2%} of variance")

    # Find most important features for first three components
    print("\nTop contributing features by component:")
    for i in range(min(3, len(df.columns))):
        pc = loadings[f'PC{i + 1}'].abs()
        top_features = pc.nlargest(3)
        print(f"\nTop features for PC{i + 1}:")
        for feat, val in top_features.items():
            print(f"- {feat}: {val:.3f}")

    return {
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': cumulative_variance_ratio,
        'n_components_95': n_components_95,
        'loadings': loadings
    }


if __name__ == "__main__":
    # Specify your CSV file path here
    file_path = (r"C:\Users\Ilya Polonsky\PycharmProjects\Data\CSV_DATA_SETS\Fault_0"
                 r"_CSV_minor_secondary_timeseries.csv")
    results = perform_pca_analysis(file_path)