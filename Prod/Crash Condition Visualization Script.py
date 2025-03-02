import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Define file paths with your actual paths
NORMAL_DATA_PATH = r"C:\Users\Ilya Polonsky\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\Filtered_Fault_0_main_Data_Set.prod.csv"
FAULT_DATA_PATH = r"C:\Users\Ilya Polonsky\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\Fault_1_to_3_main_Data_Set.prod.csv"
OUTPUT_DIRECTORY = "Crash_Condition_Results"


def plot_parameter_distributions(normal_df, fault_df, output_dir):
    """
    Create 3 distribution charts for Voltage, Current, and Temperature.
    Each chart shows normal data, fault data, and crash data.
    """
    # Apply crash condition to identify crash points
    crash_points = fault_df[
        (fault_df['Temperature_main'] > 5.8) &
        ((fault_df['Current_main'] > 5) | (fault_df['Current_main'] < 4)) &
        (fault_df['Voltage_main'] < 7)
        ]

    false_positives = normal_df[
        (normal_df['Temperature_main'] > 5.8) &
        ((normal_df['Current_main'] > 5) | (normal_df['Current_main'] < 4)) &
        (normal_df['Voltage_main'] < 7)
        ]

    # Calculate metrics for summary
    detection_rate = len(crash_points) / len(fault_df) * 100
    false_positive_rate = len(false_positives) / len(normal_df) * 100
    precision = 100.0 if len(false_positives) == 0 else (
                len(crash_points) / (len(crash_points) + len(false_positives)) * 100)

    print(f"Crash points detected: {len(crash_points)} of {len(fault_df)} ({detection_rate:.2f}%)")
    print(f"False positives: {len(false_positives)} of {len(normal_df)} ({false_positive_rate:.2f}%)")
    print(f"Precision: {precision:.2f}%")

    # Create a figure with 3 rows for the 3 parameters
    plt.figure(figsize=(12, 18))

    # 1. Temperature Distribution
    plt.subplot(3, 1, 1)

    # Plot histograms
    sns.histplot(normal_df['Temperature_main'], color='green', alpha=0.4,
                 label='Normal Data', kde=True, stat="density", linewidth=0)
    sns.histplot(fault_df['Temperature_main'], color='red', alpha=0.4,
                 label='Fault Data', kde=True, stat="density", linewidth=0)
    sns.histplot(crash_points['Temperature_main'], color='blue', alpha=0.6,
                 label='Crash Points', kde=True, stat="density", linewidth=0)

    # Add threshold line
    plt.axvline(x=5.8, color='black', linestyle='--', linewidth=2,
                label='Threshold: temp.main > 5.8')

    # Customize plot
    plt.title('Temperature Distribution', fontsize=16)
    plt.xlabel('Temperature_main', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. Current Distribution
    plt.subplot(3, 1, 2)

    # Plot histograms
    sns.histplot(normal_df['Current_main'], color='green', alpha=0.4,
                 label='Normal Data', kde=True, stat="density", linewidth=0)
    sns.histplot(fault_df['Current_main'], color='red', alpha=0.4,
                 label='Fault Data', kde=True, stat="density", linewidth=0)
    sns.histplot(crash_points['Current_main'], color='blue', alpha=0.6,
                 label='Crash Points', kde=True, stat="density", linewidth=0)

    # Add threshold lines
    plt.axvline(x=4, color='black', linestyle='--', linewidth=2,
                label='Threshold: current.main < 4')
    plt.axvline(x=5, color='black', linestyle=':', linewidth=2,
                label='Threshold: current.main > 5')

    # Customize plot
    plt.title('Current Distribution', fontsize=16)
    plt.xlabel('Current_main', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. Voltage Distribution
    plt.subplot(3, 1, 3)

    # Plot histograms
    sns.histplot(normal_df['Voltage_main'], color='green', alpha=0.4,
                 label='Normal Data', kde=True, stat="density", linewidth=0)
    sns.histplot(fault_df['Voltage_main'], color='red', alpha=0.4,
                 label='Fault Data', kde=True, stat="density", linewidth=0)
    sns.histplot(crash_points['Voltage_main'], color='blue', alpha=0.6,
                 label='Crash Points', kde=True, stat="density", linewidth=0)

    # Add threshold line
    plt.axvline(x=7, color='black', linestyle='--', linewidth=2,
                label='Threshold: voltage.main < 7')

    # Customize plot
    plt.title('Voltage Distribution', fontsize=16)
    plt.xlabel('Voltage_main', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add overall title and crash condition text
    plt.suptitle('Distribution of Parameters with Crash Condition', fontsize=18)

    condition_text = "Crash Condition:\ntemp.main > 5.8 AND (current.main > 5 OR current.main < 4) AND voltage.main < 7\n" + \
                     f"Detection Rate: {detection_rate:.2f}% | Precision: {precision:.2f}%"
    plt.figtext(0.5, 0.01, condition_text, ha='center', fontsize=14,
                bbox=dict(facecolor='#f8f9fa', alpha=0.8, boxstyle='round,pad=0.5',
                          edgecolor='black'))

    # Save figure
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    output_path = output_dir / 'parameter_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDistribution charts saved to: {output_path}")

    return output_path


def main():
    # Create output directory
    output_dir = Path(OUTPUT_DIRECTORY)
    output_dir.mkdir(exist_ok=True)

    try:
        # Load data
        print(f"Loading normal data from: {NORMAL_DATA_PATH}")
        normal_df = pd.read_csv(NORMAL_DATA_PATH)

        print(f"Loading fault data from: {FAULT_DATA_PATH}")
        fault_df = pd.read_csv(FAULT_DATA_PATH)

        # Remove timestamp columns if present
        for df in [normal_df, fault_df]:
            for col in ['TimeStamp', 'Timestamp']:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

        # Check if required columns exist
        required_columns = ['Temperature_main', 'Current_main', 'Voltage_main']
        for col in required_columns:
            if col not in normal_df.columns or col not in fault_df.columns:
                print(f"ERROR: Required column '{col}' not found in data files.")
                return

        # Generate distribution charts
        output_path = plot_parameter_distributions(normal_df, fault_df, output_dir)

        # Show the plot (optional)
        plt.show()

    except FileNotFoundError as e:
        print(f"ERROR: File not found. {e}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()