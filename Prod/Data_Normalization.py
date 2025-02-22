import pandas as pd
import numpy as np
from pathlib import Path


def min_max_normalize(df):
    """
    Perform min-max normalization on all features.
    Formula: (x - min) / (max - min)
    """
    normalized_df = df.copy()

    # Get all feature columns (excluding any non-numeric columns)
    feature_columns = df.select_dtypes(include=[np.number]).columns

    # Store normalization parameters for each feature
    normalization_params = {}

    for column in feature_columns:
        min_val = df[column].min()
        max_val = df[column].max()

        # Store parameters
        normalization_params[column] = {
            'min': min_val,
            'max': max_val
        }

        # Perform min-max normalization
        normalized_df[column] = (df[column] - min_val) / (max_val - min_val)

    return normalized_df, normalization_params


def process_and_save(input_file):
    """
    Process the file and save normalized version
    """
    try:
        # Read the data
        df = pd.read_csv(input_file)

        # Perform normalization
        normalized_df, params = min_max_normalize(df)

        # Create output path
        input_path = Path(input_file)
        output_filename = f"normalized_minmax_{input_path.name}"
        output_path = input_path.parent / output_filename

        # Save normalized data
        normalized_df.to_csv(output_path, index=False)

        # Print summary
        print("\nNormalization Summary:")
        print("-" * 50)
        for column in normalized_df.columns:
            if column in params:
                print(f"\n{column}:")
                print(f"Original range: [{params[column]['min']:.2f}, {params[column]['max']:.2f}]")
                print(f"Normalized range: [{normalized_df[column].min():.2f}, {normalized_df[column].max():.2f}]")
                print(f"Mean: {normalized_df[column].mean():.3f}")

        print(f"\nNormalized data saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Error during normalization: {str(e)}")
        return False


# Example usage
if __name__ == "__main__":
    input_file = (r"C:\Users\polon\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_"
             r"DataSet\Filtered_Fault_0_main_secondary_Data_Set.prod.csv")
    process_and_save(input_file)