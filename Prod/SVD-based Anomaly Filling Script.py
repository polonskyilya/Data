import pandas as pd
import numpy as np
from pathlib import Path
import os


def create_svd_folder(input_file_path):
    """Create SVD_Filled_Data folder in the same directory as input file"""
    input_path = Path(input_file_path)
    svd_folder = input_path.parent / 'SVD_Filled_Data'
    svd_folder.mkdir(exist_ok=True)
    return svd_folder


def save_svd_matrices(U, S, Vt, original_filename, svd_folder):
    """Save U, Σ, and V^T matrices with original filename as prefix"""
    base_name = Path(original_filename).stem

    # Save U matrix
    pd.DataFrame(U).to_csv(svd_folder / f"{base_name}_U_matrix.csv", index=False)

    # Save Singular values (Σ)
    pd.DataFrame(S).to_csv(svd_folder / f"{base_name}_Sigma_values.csv", index=False)

    # Save V^T matrix
    pd.DataFrame(Vt).to_csv(svd_folder / f"{base_name}_Vt_matrix.csv", index=False)


def fill_anomalies_svd(input_file, voltage_threshold=0.895680, current_threshold=0.254, temp_threshold=0.250):
    """
    Fill anomalies in the data using SVD decomposition

    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    voltage_threshold : float
        Threshold for voltage anomalies
    current_threshold : float
        Threshold for current anomalies
    temp_threshold : float
        Threshold for temperature anomalies
    """
    # Read the data
    df = pd.read_csv(input_file)

    # Create SVD folder
    svd_folder = create_svd_folder(input_file)

    # Create mask for anomalies
    voltage_mask = df['Voltage_main'] > voltage_threshold
    current_mask = df['Current_main'] > current_threshold
    temp_mask = df['Temperature_main'] > temp_threshold

    # Copy original data
    df_filled = df.copy()

    # Replace anomalies with NaN
    df_filled.loc[voltage_mask, 'Voltage_main'] = np.nan
    df_filled.loc[current_mask, 'Current_main'] = np.nan
    df_filled.loc[temp_mask, 'Temperature_main'] = np.nan

    # Prepare data for SVD
    data_matrix = df_filled.values

    # Get indices of missing values
    nan_mask = np.isnan(data_matrix)

    # Initial fill of NaN values with column means
    col_means = np.nanmean(data_matrix, axis=0)
    data_matrix_filled = data_matrix.copy()
    for i in range(data_matrix.shape[1]):
        data_matrix_filled[np.isnan(data_matrix[:, i]), i] = col_means[i]

    # Perform SVD
    U, S, Vt = np.linalg.svd(data_matrix_filled, full_matrices=False)

    # Save SVD matrices
    save_svd_matrices(U, S, Vt, input_file, svd_folder)

    # Reconstruct data using reduced rank (using 80% of singular values)
    k = int(0.8 * len(S))
    data_reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    # Replace only the anomalous values with reconstructed values
    data_matrix_filled[nan_mask] = data_reconstructed[nan_mask]

    # Create filled DataFrame
    df_result = pd.DataFrame(data_matrix_filled, columns=df.columns)

    # Save filled data
    output_filename = Path(input_file).stem + '_SVD_filled.csv'
    df_result.to_csv(svd_folder / output_filename, index=False)

    # Save anomaly statistics
    stats = {
        'Total Rows': len(df),
        'Voltage Anomalies': voltage_mask.sum(),
        'Current Anomalies': current_mask.sum(),
        'Temperature Anomalies': temp_mask.sum()
    }

    pd.DataFrame([stats]).to_csv(svd_folder / f"{Path(input_file).stem}_anomaly_stats.csv", index=False)

    return df_result


# Example usage
if __name__ == "__main__":
    input_file = (r"C:\Users\polon\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\normalized_minmax_Fault_1_to_3_main_secondary_Data_Set.prod.csv")  # Replace with your input file path
    filled_data = fill_anomalies_svd(input_file)