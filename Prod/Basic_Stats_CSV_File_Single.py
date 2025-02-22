# Set your file path here - just change this variable to analyze different files
FILE_PATH = (r"C:\Users\Ilya Polonsky\PycharmProjects\Data\CSV_DATA_SETS\Prediction_"
             r"Fault_DataSet\Main_Secondary_DataSet\Filtered\Filtered_"
             r"Fault_0_WO_Fault_L_main_secondary_Voltage_Current_Temp_Only_dataSet_SVD_filled.csv")

import pandas as pd
import numpy as np
from collections import Counter
import os


def analyze_csv(file_path):
    """
    Analyze CSV file and generate statistical measures

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    dict: Dictionary containing statistical measures for each column
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Initialize results dictionary
    results = {}

    # Analyze each column
    for column in df.columns:
        # Convert column to numeric, coercing errors to NaN
        series = pd.to_numeric(df[column], errors='coerce')

        # Skip if column has no numeric values
        if series.isna().all():
            print(f"Skipping column {column} - no numeric values")
            continue

        # Calculate statistics
        stats = {
            'Column': column,
            'Mean': series.mean(),
            'Median': series.median(),
            'Std': series.std(),
            'Variance': series.var(),
            'Q1': series.quantile(0.25),
            'Q2': series.quantile(0.50),
            'Q3': series.quantile(0.75),
            'Q4': series.quantile(1.0),
            'Min': series.min(),
            'Max': series.max(),
            'Count': series.count(),
            'Value_Counts': dict(series.value_counts().head(10))  # Top 10 most common values
        }

        results[column] = stats

    return results


def save_results(results, original_file_path):
    """
    Save results to CSV file and print to console

    Parameters:
    results (dict): Dictionary containing statistical measures
    original_file_path (str): Original CSV file path
    """
    # Create output directory if it doesn't exist
    output_dir = 'analysis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare data for CSV
    csv_data = []
    for column, stats in results.items():
        row = {
            'Column': column,
            'Mean': stats['Mean'],
            'Median': stats['Median'],
            'Std': stats['Std'],
            'Variance': stats['Variance'],
            'Q1': stats['Q1'],
            'Q2': stats['Q2'],
            'Q3': stats['Q3'],
            'Q4': stats['Q4'],
            'Min': stats['Min'],
            'Max': stats['Max'],
            'Count': stats['Count'],
            'Most_Common_Values': str(stats['Value_Counts'])
        }
        csv_data.append(row)

    # Create DataFrame and save to CSV
    output_df = pd.DataFrame(csv_data)
    base_filename = os.path.splitext(os.path.basename(original_file_path))[0]
    output_path = os.path.join(output_dir, f'{base_filename}_analysis.csv')
    output_df.to_csv(output_path, index=False, encoding='utf-8')

    # Print results to console
    print(f"\nStatistical Analysis Results for {original_file_path}")
    print("=" * 80)
    for column, stats in results.items():
        print(f"\nColumn: {column}")
        print("-" * 40)
        for key, value in stats.items():
            if key != 'Value_Counts':
                print(f"{key}: {value}")
        print("Most Common Values:")
        for val, count in stats['Value_Counts'].items():
            print(f"  {val}: {count} occurrences")

    print(f"\nResults saved to: {output_path}")


def main():
    """
    Main function to run the analysis
    """
    try:
        results = analyze_csv(FILE_PATH)
        save_results(results, FILE_PATH)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    main()