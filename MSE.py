import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def calculate_svd_mse(before_df, after_df):
    """
    Calculate MSE between before and after SVD data for specific measurement columns.

    Parameters:
    before_df (pd.DataFrame): DataFrame before SVD imputation
    after_df (pd.DataFrame): DataFrame after SVD imputation

    Returns:
    dict: MSE scores for each column pair and overall
    """
    columns = [
        'Voltage_main', 'Voltage_secondary',
        'Current_main', 'Current_secondary',
        'Temperature_main', 'Temperature_secondary'
    ]

    mse_scores = {}

    # Calculate MSE for each column
    for column in columns:
        # Get mask of non-missing values in original data
        known_values_mask = ~before_df[column].isna()

        if known_values_mask.any():
            # Get original known values
            original_known = before_df.loc[known_values_mask, column]
            # Get corresponding imputed values
            svd_values = after_df.loc[known_values_mask, column]

            # Calculate MSE for this column
            mse = mean_squared_error(original_known, svd_values)
            mse_scores[column] = mse

    # Calculate overall MSE
    mse_scores['overall'] = np.mean(list(mse_scores.values()))

    return mse_scores


def print_mse_analysis(before_df, after_df):
    """
    Print detailed MSE analysis with grouping by measurement type
    """
    mse_scores = calculate_svd_mse(before_df, after_df)

    print("MSE Analysis for SVD Imputation:")
    print("-" * 50)

    # Group and print by measurement type
    measurements = ['Voltage', 'Current', 'Temperature']
    for measure in measurements:
        print(f"\n{measure} measurements:")
        main_mse = mse_scores.get(f'{measure}_main', 0)
        secondary_mse = mse_scores.get(f'{measure}_secondary', 0)
        print(f"  Main: {main_mse:.6f}")
        print(f"  Secondary: {secondary_mse:.6f}")

    print("\n" + "-" * 50)
    print(f"Overall MSE: {mse_scores['overall']:.6f}")


# Example usage:
if __name__ == "__main__":
    # Load your before and after SVD files
    before_svd_file = (r"C:\Users\polon\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\normalized_minmax"
                       r"_Fault_1_to_3_main_secondary_Data_Set.prod.csv")
    after_svd_file = (r"C:\Users\polon\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\SVD_Filled_"
                      r"Data\normalized_minmax_Fault_1_to_3_main_secondary_Data_Set.prod_SVD_filled.csv")

    before_df = pd.read_csv(before_svd_file)
    after_df = pd.read_csv(after_svd_file)

    # Print MSE analysis
    print_mse_analysis(before_df, after_df)