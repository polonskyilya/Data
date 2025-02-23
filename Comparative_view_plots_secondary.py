import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def get_values_until_90_percent(data, column):
    """Get values that make up 75% of the data"""
    counts = data[column].value_counts().sort_index()
    total = len(data)
    cumsum = 0
    values = []
    counts_list = []
    percentages = []

    for value, count in counts.items():
        if cumsum >= 75:
            break
        values.append(int(value))  # Ensure values are kept as integers
        counts_list.append(count)
        percentage = (count / total) * 100
        percentages.append(percentage)
        cumsum += percentage

    return values, counts_list, percentages, cumsum


def create_secondary_value_plots(csv_path1, csv_path2):
    """
    Create separate plots showing values that make up 75% of data for secondary variables.
    """
    try:
        # Read both CSV files
        df1 = pd.read_csv(csv_path1)  # Fault_0
        df2 = pd.read_csv(csv_path2)  # Fault_1_to_3

        # Create output directory
        output_dir = Path('Secondary_Values_Analysis')
        output_dir.mkdir(exist_ok=True)

        # Define colors for secondary variables
        colors = ['#e67e22', '#1f77b4']  # Orange for Fault_0, Blue for Fault_1_to_3

        # Function to plot single distribution
        def plot_distribution(values, counts, percentages, cumsum, title, color, total, filename):
            plt.figure(figsize=(35, 25))
            bars = plt.bar(values, counts, color=color, alpha=0.8, edgecolor='black', width=0.9)
            plt.xlabel('Value', fontsize=32)
            plt.ylabel('Frequency', fontsize=32)
            plt.xticks(rotation=90, fontsize=28)
            plt.yticks(fontsize=28)
            plt.tick_params(axis='both', which='major', labelsize=28)

            # Add value labels
            for bar, percentage in zip(bars, percentages):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'Count: {int(height)}\n{percentage:.2f}%',
                         ha='center', va='bottom', fontsize=18, rotation=90)

            # Add detailed grid
            plt.grid(True, linestyle='--', alpha=0.7, which='both')
            plt.minorticks_on()
            plt.gca().xaxis.set_major_locator(plt.FixedLocator(values))  # Ensure only dataset values appear
            plt.gca().yaxis.set_major_locator(plt.FixedLocator(counts))

            # Add title above the stats box
            plt.title(f'{title}\n(Shows values up to {cumsum:.2f}% of data)', fontsize=36, pad=100)

            # Add stats box
            stats = f"Total data points: {total}\n"
            stats += f"Shown values: {sum(counts)}\n"
            stats += f"Percentage shown: {cumsum:.2f}%"
            plt.text(1.02, 0.5, stats, transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.8), fontsize=26)

            # Save plot
            output_path = output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()
            print(f"Plot successfully saved as: {output_path}")

        # Identify secondary variable columns
        secondary_columns = [col for col in df1.columns if 'secondary' in col.lower()]

        # Create and save separate plots for each secondary column
        for column in secondary_columns:
            for df, color, title in [(df1, colors[0], 'Fault_0'), (df2, colors[1], 'Fault_1_to_3')]:
                values, counts, percentages, cumsum = get_values_until_90_percent(df, column)
                filename = f"{column}_Secondary_Distribution_{title}.png"
                plot_distribution(values, counts, percentages, cumsum, f'{column} Distribution - {title}', color,
                                  len(df), filename)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    csv_path1 = (r"C:\Users\Ilya Polonsky\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\Fault_0_main_"
                 r"secondary_Data_Set.prod.csv")
    csv_path2 = (
        r"C:\Users\Ilya Polonsky\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\Fault_1_to_3_main_"
        r"secondary_Data_Set.prod.csv")
    create_secondary_value_plots(csv_path1, csv_path2)