import pandas as pd
from pathlib import Path


def extract_main_secondary(value):
    """
    Extract main and secondary values from a dot-delimited number.
    Example: "382.501.409.468.062" -> (382, 501)
    Returns None, None if extraction fails
    """
    try:
        if pd.isna(value) or str(value).strip() == '':
            return None, None

        # Convert to string and split by dots
        parts = str(value).split('.')
        if len(parts) >= 2:
            main = parts[0].strip()
            secondary = parts[1].strip()
            if main and secondary:  # Check if both parts are non-empty
                return int(main), int(secondary)
        return None, None
    except Exception as e:
        print(f"Error processing value {value}: {str(e)}")
        return None, None


def process_csv_file(input_file_path):
    """
    Process CSV file to extract main and secondary values.
    Creates two files:
    1. Main output file with successfully processed rows
    2. Error file with rows that couldn't be processed
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file_path)

        # Create new dataframe for valid rows
        new_df = pd.DataFrame()

        # Create list to store indices of error rows
        error_rows_idx = set()

        # Process each measurement type
        measurements = ['Voltage', 'Current', 'Temperature']
        processed_data = {}

        # First pass: extract all values and identify error rows
        for measure in measurements:
            if measure in df.columns:
                main_values = []
                secondary_values = []

                for idx, value in enumerate(df[measure]):
                    main, secondary = extract_main_secondary(value)
                    if main is None or secondary is None:
                        error_rows_idx.add(idx)
                    main_values.append(main)
                    secondary_values.append(secondary)

                processed_data[measure] = (main_values, secondary_values)

        # Create valid rows dataframe
        valid_rows_idx = set(range(len(df))) - error_rows_idx
        if valid_rows_idx:
            new_df['Fault_Label'] = df.loc[list(valid_rows_idx), 'Fault Label'].values

            for measure in measurements:
                if measure in df.columns:
                    main_values, secondary_values = processed_data[measure]
                    new_df[f'{measure}_main'] = [main_values[i] for i in valid_rows_idx]
                    new_df[f'{measure}_secondary'] = [secondary_values[i] for i in valid_rows_idx]

        # Create error rows dataframe
        error_df = df.loc[list(error_rows_idx)].copy() if error_rows_idx else pd.DataFrame()

        # Create output directory if it doesn't exist
        output_dir = Path(
            r"C:\Users\Ilya Polonsky\PycharmProjects\Data\CSV_DATA_SETS\Prediction_Fault_DataSet\Main_Secondary_DataSet")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filenames
        input_filename = Path(input_file_path).name
        base_name = input_filename.rsplit('.', 1)[0]
        output_filename = f"main_secondary_{input_filename}"
        error_filename = f"error_rows_{input_filename}"

        output_path = output_dir / output_filename
        error_path = output_dir / error_filename

        # Save files
        if not new_df.empty:
            new_df.to_csv(output_path, index=False)
            print(f"Successfully processed {len(new_df)} rows")
            print(f"Created new file: {output_path}")

        if not error_df.empty:
            error_df.to_csv(error_path, index=False)
            print(f"Found {len(error_df)} problematic rows")
            print(f"Created error file: {error_path}")

        # Print summary
        print("\nProcessing Summary:")
        print(f"Total rows: {len(df)}")
        print(f"Successfully processed: {len(new_df)}")
        print(f"Error rows: {len(error_df)}")

        return True

    except Exception as e:
        print(f"Error processing file {input_file_path}: {str(e)}")
        return False


# Example usage
if __name__ == "__main__":
    # Specify the input file path
    input_file = (r"C:\Users\Ilya Polonsky\PycharmProjects\Data\CSV_DATA_SETS\Prediction_Fault_DataSet\Voltage"
                  r"_Current_Temp_Only_dataSet.csv")

    # Process the single CSV file
    process_csv_file(input_file)