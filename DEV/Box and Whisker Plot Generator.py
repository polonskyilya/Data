import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set the file path here
FILE_PATH = (r"C:\Users\Ilya Polonsky\PycharmProjects\Data\CSV_DATA_SETS\Prediction_"
             r"Fault_DataSet\Main_Secondary_DataSet\Filtered\Filtered_Fault_0_WO_Fault_L_main_secondary_"
             r"Voltage_Current_Temp_Only_dataSet_SVD_filled.csv")

def create_box_whisker_plot(csv_path):
    """
    Create a box and whisker plot from a CSV file containing electrical measurements.

    Args:
        csv_path (str): Path to the CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Convert all columns to float to ensure consistency
        df = df.astype(float)

        # Create figure and subplots for different measurement types
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))

        # Colors for main and secondary measurements
        colors = ['#2ecc71', '#3498db']

        # 1. Voltage Plot
        voltage_data = pd.DataFrame({
            'Main': df['Voltage_main'],
            'Secondary': df['Voltage_secondary']
        }).melt()
        sns.boxplot(data=voltage_data, x='variable', y='value', hue='variable',
                    ax=ax1, palette=colors, width=0.5, legend=False)
        ax1.set_title('Voltage Distribution', pad=20, size=14)
        ax1.set_xlabel('Measurement Type')
        ax1.set_ylabel('Voltage (V)')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 2. Current Plot
        current_data = pd.DataFrame({
            'Main': df['Current_main'],
            'Secondary': df['Current_secondary']
        }).melt()
        sns.boxplot(data=current_data, x='variable', y='value', hue='variable',
                    ax=ax2, palette=colors, width=0.5, legend=False)
        ax2.set_title('Current Distribution', pad=20, size=14)
        ax2.set_xlabel('Measurement Type')
        ax2.set_ylabel('Current (A)')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # 3. Temperature Plot
        temp_data = pd.DataFrame({
            'Main': df['Temperature_main'],
            'Secondary': df['Temperature_secondary']
        }).melt()
        sns.boxplot(data=temp_data, x='variable', y='value', hue='variable',
                    ax=ax3, palette=colors, width=0.5, legend=False)
        ax3.set_title('Temperature Distribution', pad=20, size=14)
        ax3.set_xlabel('Measurement Type')
        ax3.set_ylabel('Temperature')
        ax3.grid(True, linestyle='--', alpha=0.7)

        # Add overall title
        plt.suptitle(f'Statistical Distribution Analysis\n{Path(csv_path).stem}',
                     size=16, y=1.02)

        # Add statistics table
        stats_text = "Statistical Summary:\n"
        for col in df.columns:
            stats = df[col].describe()
            stats_text += f"\n{col}:\n"
            stats_text += f"Mean: {stats['mean']:.2f}\n"
            stats_text += f"Std: {stats['std']:.2f}\n"
            stats_text += f"Min: {stats['min']:.2f}\n"
            stats_text += f"Max: {stats['max']:.2f}\n"

        plt.figtext(1.02, 0.5, stats_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

        # Adjust layout
        plt.tight_layout()

        # Create output directory if it doesn't exist
        output_dir = Path('Box and Whisker Plots')
        output_dir.mkdir(exist_ok=True)

        # Generate output filename
        input_filename = Path(csv_path).stem
        output_path = output_dir / f"{input_filename}.png"

        # Save the plot with extra space for statistics
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    pad_inches=0.5)
        plt.close()

        print(f"Plot successfully saved as: {output_path}")

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise


if __name__ == "__main__":
    # Process the CSV file
    create_box_whisker_plot(FILE_PATH)