import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Define your CSV file paths directly in the script
file1_path = r"C:\Users\Ilya Polonsky\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\Fault_1_to_3_main_Data_Set.prod.csv"
file2_path = r"C:\Users\Ilya Polonsky\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\Filtered_Fault_0_main_Data_Set.prod.csv"


def load_data(file1, file2):
    """Load data from two CSV files"""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Print the column names to debug
    print("Columns in file 1:", df1.columns.tolist())
    print("Columns in file 2:", df2.columns.tolist())

    # Add fault type column to identify the source
    df1['fault_type'] = 'Fault 0'
    df2['fault_type'] = 'Fault 1-3'

    return df1, df2


def calculate_statistics(df1, df2):
    """Calculate statistics for each dataset"""
    stats_dict = {}

    # Get the columns that are actually in the dataframes
    # Skip the fault_type column we added
    columns = [col for col in df1.columns if col != 'fault_type']

    # Calculate statistics for each dataframe
    for name, df in [('Fault 0', df1), ('Fault 1-3', df2)]:
        stats_dict[name] = {}

        for column in columns:
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

    # Get all columns except fault_type
    variables = [col for col in df1.columns if col != 'fault_type']

    # Set up the figure with enough rows for all variables
    fig, axes = plt.subplots(len(variables), 2, figsize=(18, 6 * len(variables)))

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
    """Create correlation heatmaps"""
    # Set up the figure for correlation heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Correlation heatmaps
    for i, (name, df) in enumerate([('Fault 0', df1), ('Fault 1-3', df2)]):
        # Drop the fault_type column for correlation
        df_corr = df.drop('fault_type', axis=1)

        # Compute correlation matrix
        corr_matrix = df_corr.corr()

        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                    linewidths=0.5, ax=axes[i])
        axes[i].set_title(f'Correlation Matrix for {name}')

    plt.tight_layout()
    plt.savefig('correlation_heatmaps.png', dpi=300)
    plt.close()

    # Create a pairplot for relationships between variables
    plt.figure(figsize=(12, 10))

    # Get all columns except fault_type
    cols = [col for col in df1.columns if col != 'fault_type']

    # Select a subset if there are too many columns
    if len(cols) > 4:
        subset_cols = cols[:4] + ['fault_type']
    else:
        subset_cols = cols + ['fault_type']

    combined_df = pd.concat([df1, df2])
    sns.pairplot(combined_df[subset_cols], hue='fault_type', diag_kind='kde')
    plt.savefig('pairplot.png', dpi=300)
    plt.close()


def plot_time_series_proxy(df1, df2):
    """Create a proxy time series plot assuming sequential ordering"""
    # Assuming samples are in temporal order
    df1 = df1.copy()
    df2 = df2.copy()

    df1['Sample'] = range(len(df1))
    df2['Sample'] = range(len(df2))

    # Get all columns except fault_type and Sample
    variables = [col for col in df1.columns if col not in ['fault_type', 'Sample']]

    # Determine how many plots per figure (max 3 per figure)
    plots_per_fig = 3
    num_figures = (len(variables) + plots_per_fig - 1) // plots_per_fig

    for fig_num in range(num_figures):
        # Calculate which variables go in this figure
        start_idx = fig_num * plots_per_fig
        end_idx = min(start_idx + plots_per_fig, len(variables))
        fig_vars = variables[start_idx:end_idx]

        # Create figure
        fig, axes = plt.subplots(len(fig_vars), 1, figsize=(18, 5 * len(fig_vars)), sharex=True)

        # Handle case with only one subplot
        if len(fig_vars) == 1:
            axes = [axes]

        for i, var in enumerate(fig_vars):
            # Plot for fault 0
            axes[i].plot(df1['Sample'], df1[var], alpha=0.7,
                         label='Fault 0', color='blue')

            # Plot for fault 1-3 (use same length as fault 0 for comparison)
            axes[i].plot(df2['Sample'][:len(df1)], df2[var][:len(df1)], alpha=0.7,
                         label='Fault 1-3', color='red')

            axes[i].set_title(f'Time Series of {var}')
            axes[i].set_ylabel(var)
            axes[i].grid(alpha=0.3)
            axes[i].legend()

        axes[-1].set_xlabel('Sample Index (Proxy for Time)')
        plt.tight_layout()
        plt.savefig(f'time_series_group{fig_num + 1}.png', dpi=300)
        plt.close()


def plot_statistical_summary(stats_dict):
    """Create a visual summary of key statistics"""
    # Get variables from the stats dictionary
    variables = list(stats_dict['Fault 0'].keys())
    fault_types = ['Fault 0', 'Fault 1-3']
    metrics = ['mean', 'median', 'std']

    # Create separate figures for groups of variables (max 3 per figure)
    plots_per_fig = 3
    num_figures = (len(variables) + plots_per_fig - 1) // plots_per_fig

    for fig_num in range(num_figures):
        # Calculate which variables go in this figure
        start_idx = fig_num * plots_per_fig
        end_idx = min(start_idx + plots_per_fig, len(variables))
        fig_vars = variables[start_idx:end_idx]

        # Create figure
        fig, axes = plt.subplots(len(fig_vars), 1, figsize=(14, 5 * len(fig_vars)))

        # Handle case with only one subplot
        if len(fig_vars) == 1:
            axes = [axes]

        for i, var in enumerate(fig_vars):
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
        plt.savefig(f'statistical_summary_group{fig_num + 1}.png', dpi=300)
        plt.close()


def perform_hypothesis_tests(df1, df2):
    """Perform statistical hypothesis tests to compare the two datasets"""
    results = {}

    # Get all columns except fault_type
    variables = [col for col in df1.columns if col != 'fault_type']

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
        f.write("- correlation_heatmaps.png\n")
        f.write("- pairplot.png\n")
        f.write("- time_series_*.png\n")
        f.write("- statistical_summary_*.png\n")


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
    print("- correlation_heatmaps.png")
    print("- pairplot.png")
    print("- time_series_*.png")
    print("- statistical_summary_*.png")
    print("- summary_report.txt")


if __name__ == "__main__":
    # Just call the main function directly - no arguments needed
    main()