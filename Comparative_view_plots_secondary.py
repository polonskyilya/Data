import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Define your CSV file paths directly in the script
file1_path = r"C:\Users\Ilya Polonsky\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\Filtered_Fault_0_secondary_Data_Set.prod.csv"
file2_path = r"C:\Users\Ilya Polonsky\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\Filtered_Fault_1_to_3_secondary_Data_Set.prod.csv"


def load_data(file1, file2):
    """Load data from two CSV files"""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Add fault type column to identify the source
    df1['fault_type'] = 'Fault 0'
    df2['fault_type'] = 'Fault 1-3'

    return df1, df2


def calculate_statistics(df1, df2):
    """Calculate statistics for each dataset"""
    stats_dict = {}

    # Calculate statistics for each dataframe
    for name, df in [('Fault 0', df1), ('Fault 1-3', df2)]:
        stats_dict[name] = {}

        for column in ['Voltage_secondary', 'Current_secondary', 'Temperature_secondary']:
            stats_dict[name][column] = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'q1': df[column].quantile(0.25),
                'q3': df[column].quantile(0.75),
                'skewness': stats.skew(df[column]),
                'kurtosis': stats.kurtosis(df[column])
            }

    return stats_dict


def plot_distributions(df1, df2):
    """Create distribution plots for each variable"""
    combined_df = pd.concat([df1, df2])

    # Set up the figure
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    # Variables to plot
    variables = ['Voltage_secondary', 'Current_secondary', 'Temperature_secondary']

    for i, var in enumerate(variables):
        # Histogram
        sns.histplot(data=combined_df, x=var, hue='fault_type',
                     kde=True, alpha=0.6, bins=30, ax=axes[i, 0])
        axes[i, 0].set_title(f'Distribution of {var}')
        axes[i, 0].grid(alpha=0.3)

        # Boxplot
        sns.boxplot(data=combined_df, x='fault_type', y=var, ax=axes[i, 1])
        axes[i, 1].set_title(f'Boxplot of {var} by Fault Type')
        axes[i, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('distributions.png', dpi=300)
    plt.close()


def plot_correlations(df1, df2):
    """Create correlation heatmaps and scatterplots"""
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))

    # Correlation heatmaps
    for i, (name, df) in enumerate([('Fault 0', df1), ('Fault 1-3', df2)]):
        # Drop the fault_type column for correlation
        df_corr = df.drop('fault_type', axis=1)

        # Compute correlation matrix
        corr_matrix = df_corr.corr()

        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                    linewidths=0.5, ax=axes[0, i])
        axes[0, i].set_title(f'Correlation Matrix for {name}')

    # Scatterplots with regression line
    combined_df = pd.concat([df1, df2])

    # Voltage vs Current
    sns.scatterplot(data=combined_df, x='Voltage_secondary', y='Current_secondary',
                    hue='fault_type', alpha=0.6, ax=axes[1, 0])
    sns.regplot(data=combined_df, x='Voltage_secondary', y='Current_secondary',
                scatter=False, ax=axes[1, 0])
    axes[1, 0].set_title('Voltage vs Current')
    axes[1, 0].grid(alpha=0.3)

    # Voltage vs Temperature
    sns.scatterplot(data=combined_df, x='Voltage_secondary', y='Temperature_secondary',
                    hue='fault_type', alpha=0.6, ax=axes[1, 1])
    sns.regplot(data=combined_df, x='Voltage_secondary', y='Temperature_secondary',
                scatter=False, ax=axes[1, 1])
    axes[1, 1].set_title('Voltage vs Temperature')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('correlations.png', dpi=300)
    plt.close()


def plot_time_series_proxy(df1, df2):
    """Create a proxy time series plot assuming sequential ordering"""
    # Assuming samples are in temporal order
    df1 = df1.copy()
    df2 = df2.copy()

    df1['Sample'] = range(len(df1))
    df2['Sample'] = range(len(df2))

    # Set up the figure
    fig, axes = plt.subplots(3, 1, figsize=(18, 15), sharex=True)

    # Variables to plot
    variables = ['Voltage_secondary', 'Current_secondary', 'Temperature_secondary']

    for i, var in enumerate(variables):
        # Plot for fault 0
        axes[i].plot(df1['Sample'], df1[var], alpha=0.7,
                     label='Fault 0', color='blue')

        # Plot for fault 1-3
        axes[i].plot(df2['Sample'][:len(df1)], df2[var][:len(df1)], alpha=0.7,
                     label='Fault 1-3', color='red')

        axes[i].set_title(f'Time Series of {var}')
        axes[i].set_ylabel(var)
        axes[i].grid(alpha=0.3)
        axes[i].legend()

    axes[2].set_xlabel('Sample Index (Proxy for Time)')
    plt.tight_layout()
    plt.savefig('time_series.png', dpi=300)
    plt.close()


def plot_statistical_summary(stats_dict):
    """Create a visual summary of key statistics"""
    # Set up the figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))

    variables = ['Voltage_secondary', 'Current_secondary', 'Temperature_secondary']
    fault_types = ['Fault 0', 'Fault 1-3']
    metrics = ['mean', 'median', 'std', 'min', 'max']

    for i, var in enumerate(variables):
        # Extract data for plotting
        data = {metric: [stats_dict[fault][var][metric] for fault in fault_types]
                for metric in metrics}

        # Create DataFrame for plotting
        plot_df = pd.DataFrame(data, index=fault_types)

        # Plot
        plot_df.plot(kind='bar', ax=axes[i], rot=0)
        axes[i].set_title(f'Statistical Summary for {var}')
        axes[i].set_ylabel('Value')
        axes[i].grid(alpha=0.3)
        axes[i].legend(title='Metric')

    plt.tight_layout()
    plt.savefig('statistical_summary.png', dpi=300)
    plt.close()


def perform_hypothesis_tests(df1, df2):
    """Perform statistical hypothesis tests to compare the two datasets"""
    results = {}
    variables = ['Voltage_secondary', 'Current_secondary', 'Temperature_secondary']

    for var in variables:
        # T-test
        t_stat, p_value = stats.ttest_ind(df1[var], df2[var], equal_var=False)
        results[f'{var}_ttest'] = {'t_statistic': t_stat, 'p_value': p_value}

        # Mann-Whitney U test (non-parametric)
        u_stat, p_value = stats.mannwhitneyu(df1[var], df2[var])
        results[f'{var}_mannwhitney'] = {'u_statistic': u_stat, 'p_value': p_value}

        # Kolmogorov-Smirnov test (distribution comparison)
        ks_stat, p_value = stats.ks_2samp(df1[var], df2[var])
        results[f'{var}_ks_test'] = {'ks_statistic': ks_stat, 'p_value': p_value}

    return results


def create_summary_report(stats_dict, hypothesis_results, output_file='summary_report.txt'):
    """Create a text summary report with all statistics and test results"""
    with open(output_file, 'w') as f:
        f.write("=====================================\n")
        f.write("STATISTICAL ANALYSIS SUMMARY REPORT\n")
        f.write("=====================================\n\n")

        # Write descriptive statistics
        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("-------------------------------------\n")

        for fault in stats_dict:
            f.write(f"\n{fault}:\n")
            for var in stats_dict[fault]:
                f.write(f"\n  {var}:\n")
                for stat, value in stats_dict[fault][var].items():
                    f.write(f"    {stat}: {value:.4f}\n")

        # Write hypothesis test results
        f.write("\n\nHYPOTHESIS TESTS (Fault 0 vs Fault 1-3)\n")
        f.write("-------------------------------------\n")

        for test, results in hypothesis_results.items():
            f.write(f"\n{test}:\n")
            for stat, value in results.items():
                f.write(f"  {stat}: {value:.6f}\n")

            # Interpret p-value
            if 'p_value' in results:
                if results['p_value'] < 0.05:
                    f.write("  Interpretation: Statistically significant difference (p < 0.05)\n")
                else:
                    f.write("  Interpretation: No statistically significant difference (p >= 0.05)\n")

        f.write("\n\nPlots saved as:\n")
        f.write("- distributions.png\n")
        f.write("- correlations.png\n")
        f.write("- time_series.png\n")
        f.write("- statistical_summary.png\n")


def main():
    """Main function to run the analysis"""
    print("Loading data...")
    df1, df2 = load_data(file1_path, file2_path)

    print("Calculating statistics...")
    stats_dict = calculate_statistics(df1, df2)

    print("Performing hypothesis tests...")
    hypothesis_results = perform_hypothesis_tests(df1, df2)

    print("Generating plots...")
    plot_distributions(df1, df2)
    plot_correlations(df1, df2)
    plot_time_series_proxy(df1, df2)
    plot_statistical_summary(stats_dict)

    print("Creating summary report...")
    create_summary_report(stats_dict, hypothesis_results)

    print("Analysis complete!")
    print("Results saved to:")
    print("- distributions.png")
    print("- correlations.png")
    print("- time_series.png")
    print("- statistical_summary.png")
    print("- summary_report.txt")


if __name__ == "__main__":
    # Just call the main function directly - no arguments needed
    main()