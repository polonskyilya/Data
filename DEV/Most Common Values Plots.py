import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set the file path here
FILE_PATH = (r"C:\Users\Ilya Polonsky\PycharmProjects\Data\CSV_DATA_SETS\Prediction_Fault_DataSet\Main_Secondary_"
             r"DataSet\Filtered\SVD_Filled_Data\Normalized\normalized_minmax_Filtered_Fault_1_to_3_WO_Fault_L_main_secondary"
             r"_Voltage_Current_Temp_Only_dataSet_SVD_filled.csv")
def create_common_values_plot(csv_path):
    """
    Create visualization for most common values from a CSV file.

    Args:
        csv_path (str): Path to the CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Create figure and subplots for different measurement types
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))

        # Colors for main and secondary measurements
        colors = ['#2ecc71', '#3498db']

        # Process each measurement type
        measurement_pairs = [
            (['Voltage_main', 'Voltage_secondary'], 'Voltage Distribution', 'V', ax1),
            (['Current_main', 'Current_secondary'], 'Current Distribution', 'A', ax2),
            (['Temperature_main', 'Temperature_secondary'], 'Temperature Distribution', '°C', ax3)
        ]

        for columns, title, unit, ax in measurement_pairs:
            # Get value counts for both columns
            main_values = df[columns[0]].value_counts().nlargest(10)
            sec_values = df[columns[1]].value_counts().nlargest(10)

            # Prepare data for plotting
            main_data = pd.DataFrame({
                'Value': main_values.index,
                'Count': main_values.values,
                'Type': 'Main'
            })
            sec_data = pd.DataFrame({
                'Value': sec_values.index,
                'Count': sec_values.values,
                'Type': 'Secondary'
            })

            # Combine data
            plot_data = pd.concat([main_data, sec_data])

            # Create grouped bar plot
            sns.barplot(data=plot_data, x='Value', y='Count', hue='Type',
                        palette=colors, ax=ax)

            # Customize the plot
            ax.set_title(f'Most Common {title}', pad=20, size=14)
            ax.set_xlabel(f'Value ({unit})')
            ax.set_ylabel('Frequency')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on top of bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', padding=3)

        # Add overall title
        plt.suptitle(f'Most Common Values Analysis\n{Path(csv_path).stem}',
                     size=16, y=1.02)

        # Add summary statistics
        stats_text = "Data Summary:\n"
        for col in df.columns:
            stats = df[col].describe()
            stats_text += f"\n{col}:\n"
            stats_text += f"Count: {stats['count']:.0f}\n"
            stats_text += f"Mean: {stats['mean']:.2f}\n"
            stats_text += f"Std: {stats['std']:.2f}\n"
            stats_text += f"Min: {stats['min']:.2f}\n"
            stats_text += f"Max: {stats['max']:.2f}\n"

        plt.figtext(1.02, 0.5, stats_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

        # Adjust layout
        plt.tight_layout()

        # Create output directory if it doesn't exist
        output_dir = Path('Most Common Values Plots')
        output_dir.mkdir(exist_ok=True)

        # Generate output filename
        input_filename = Path(csv_path).stem
        output_path = output_dir / f"{input_filename}_common_values.png"

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
    create_common_values_plot(FILE_PATH)