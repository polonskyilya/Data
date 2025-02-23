import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_main_value_plots(csv_path):
    """
    Create plots focusing on the main values with their exact frequencies and percentages.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Create figure with 3 subplots (vertical arrangement)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))

        # Colors
        bar_color = '#2ecc71'
        text_color = 'black'

        # 1. Voltage_main plot
        voltage_values = [9, 8, 7]
        voltage_counts = [1774, 1756, 920]
        voltage_percentages = [35.48, 35.12, 18.40]

        bars1 = ax1.bar(voltage_values, voltage_counts, color=bar_color)
        ax1.set_title('Voltage_main Distribution (Top 3 Values = 89% of data)', pad=20)
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')

        # Add value labels on bars
        for bar, percentage in zip(bars1, voltage_percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'Count: {int(height)}\n{percentage:.2f}%',
                     ha='center', va='bottom')

        # 2. Current_main plot
        current_values = [5, 6, 8, 9, 7, 3, 4, 2]
        current_counts = [554, 477, 477, 472, 466, 288, 277, 208]
        current_percentages = [11.08, 9.54, 9.54, 9.44, 9.32, 5.76, 5.54, 4.16]

        bars2 = ax2.bar(current_values, current_counts, color=bar_color)
        ax2.set_title('Current_main Distribution (Top 8 Values = 64.38% of data)', pad=20)
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')

        # Add value labels on bars
        for bar, percentage in zip(bars2, current_percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'Count: {int(height)}\n{percentage:.2f}%',
                     ha='center', va='bottom')

        # 3. Temperature_main plot
        temp_values = [5, 4, 3, 2, 1]
        temp_counts = [642, 513, 502, 376, 200]
        temp_percentages = [12.84, 10.26, 10.04, 7.52, 4.00]

        bars3 = ax3.bar(temp_values, temp_counts, color=bar_color)
        ax3.set_title('Temperature_main Distribution (Top 5 Values = 44.66% of data)', pad=20)
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Frequency')

        # Add value labels on bars
        for bar, percentage in zip(bars3, temp_percentages):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f'Count: {int(height)}\n{percentage:.2f}%',
                     ha='center', va='bottom')

        # Add statistics boxes
        voltage_stats = (f"Total data points: 5000\n"
                         f"Shown values: 4450\n"
                         f"Percentage shown: 89.00%")
        ax1.text(1.02, 0.5, voltage_stats, transform=ax1.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

        current_stats = (f"Total data points: 5000\n"
                         f"Shown values: 3219\n"
                         f"Percentage shown: 64.38%")
        ax2.text(1.02, 0.5, current_stats, transform=ax2.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

        temp_stats = (f"Total data points: 5000\n"
                      f"Shown values: 2233\n"
                      f"Percentage shown: 44.66%")
        ax3.text(1.02, 0.5, temp_stats, transform=ax3.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

        # Add grid
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax3.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout
        plt.suptitle('Main Values Distribution Analysis', fontsize=16, y=1.02)
        plt.tight_layout()

        # Create output directory
        output_dir = Path('Main Values Analysis')
        output_dir.mkdir(exist_ok=True)

        # Save plot
        output_path = output_dir / f"{Path(csv_path).stem}_main_values.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

        print(f"Plot successfully saved as: {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        plt.close('all')


if __name__ == "__main__":
    csv_path = (r"C:\Users\Ilya Polonsky\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\Fault_1_to_3_main_"
                r"secondary_Data_Set.prod.csv")  # Update path as needed
    create_main_value_plots(csv_path)