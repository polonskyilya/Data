import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def gradient_descent_fault_detector(normal_file_path, fault_file_path):
    """
    Analyze normal and fault data using gradient descent to find points of failure.

    Parameters:
    normal_file_path (str): Path to CSV file with normal behavior data (fault_0)
    fault_file_path (str): Path to CSV file with fault behavior data (fault_1_to_3)
    """
    # Load data
    print(f"Loading normal behavior data from: {normal_file_path}")
    print(f"Loading fault behavior data from: {fault_file_path}")

    try:
        # Load CSV files
        normal_df = pd.read_csv(normal_file_path)
        fault_df = pd.read_csv(fault_file_path)

        # Remove timestamp if present (as specified)
        if 'TimeStamp' in normal_df.columns:
            normal_df = normal_df.drop('TimeStamp', axis=1)
        if 'Timestamp' in normal_df.columns:
            normal_df = normal_df.drop('Timestamp', axis=1)
        if 'TimeStamp' in fault_df.columns:
            fault_df = fault_df.drop('TimeStamp', axis=1)
        if 'Timestamp' in fault_df.columns:
            fault_df = fault_df.drop('Timestamp', axis=1)

        # Add labels
        normal_df['label'] = 0  # Normal behavior
        fault_df['label'] = 1  # Fault behavior

        # Combine datasets for analysis
        combined_df = pd.concat([normal_df, fault_df], ignore_index=True)

        # Extract features (all columns except label)
        features = [col for col in combined_df.columns if col != 'label']
        X = combined_df[features].values
        y = combined_df['label'].values

        # Check if data is already normalized (values between 0 and 1)
        min_vals = combined_df[features].min()
        max_vals = combined_df[features].max()
        is_normalized = (min_vals >= -0.1).all() and (max_vals <= 1.1).all()

        if is_normalized:
            print("Data appears to be normalized (values between 0 and 1)")
        else:
            print("WARNING: Data does not appear to be normalized. Results may be less reliable.")

        # Initial weights and bias (starting point for gradient descent)
        weights = np.zeros(len(features))
        bias = 0
        learning_rate = 0.01
        iterations = 1000

        # Implement gradient descent
        print("Performing gradient descent to find decision boundary...")
        weights, bias, cost_history = logistic_regression_gradient_descent(
            X, y, weights, bias, learning_rate, iterations
        )

        # Analyze results
        feature_importance = list(zip(features, weights))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        # Calculate point of failure thresholds
        thresholds = calculate_normalized_thresholds(weights, bias, features, normal_df, fault_df)

        # Calculate prediction accuracy
        accuracy, confusion_matrix = evaluate_model(weights, bias, features, normal_df, fault_df)

        # Print results
        print("\nGradient Descent Complete")
        print(f"Final model accuracy: {accuracy * 100:.2f}%")
        print("\nFeature Importance (by weight magnitude):")
        for feature, weight in feature_importance:
            print(f"{feature}: {weight:.6f}")

        print("\nPoint of Failure Thresholds:")
        for feature, threshold_info in thresholds.items():
            direction = ">" if weights[features.index(feature)] > 0 else "<"
            print(f"{feature}: {direction} {threshold_info['value']:.6f}")
            print(f"  Normal mean: {threshold_info['normal_mean']:.6f}, Fault mean: {threshold_info['fault_mean']:.6f}")

        print("\nConfusion Matrix:")
        print(f"True Positives: {confusion_matrix['tp']}")
        print(f"False Positives: {confusion_matrix['fp']}")
        print(f"True Negatives: {confusion_matrix['tn']}")
        print(f"False Negatives: {confusion_matrix['fn']}")

        # Generate visualizations
        output_dir = Path('Fault_Analysis_Results')
        output_dir.mkdir(exist_ok=True)
        plot_results(normal_df, fault_df, weights, bias, features, cost_history, thresholds, output_dir)

        return weights, bias, thresholds

    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
        return None, None, None
    except Exception as e:
        print(f"Error processing files: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def sigmoid(z):
    """Sigmoid activation function with clipping to prevent overflow"""
    return 1 / (1 + np.exp(-np.clip(z, -20, 20)))


def logistic_regression_gradient_descent(X, y, weights, bias, learning_rate, iterations):
    """
    Implement gradient descent for logistic regression from scratch.
    """
    m = len(y)
    cost_history = []

    for i in range(iterations):
        # Linear combination
        z = np.dot(X, weights) + bias
        predictions = sigmoid(z)

        # Calculate gradients
        dw = (1 / m) * np.dot(X.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Calculate cost (optional, for monitoring)
        if i % 100 == 0:
            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            safe_predictions = np.clip(predictions, epsilon, 1 - epsilon)
            cost = (-1 / m) * np.sum(y * np.log(safe_predictions) + (1 - y) * np.log(1 - safe_predictions))
            cost_history.append(cost)
            print(f"Iteration {i}, Cost: {cost:.6f}")

    return weights, bias, cost_history


def calculate_normalized_thresholds(weights, bias, features, normal_df, fault_df):
    """
    Calculate reasonable threshold values constrained to the data range.
    Returns thresholds with additional context about normal and fault means.
    """
    thresholds = {}

    for i, feature in enumerate(features):
        # Skip features with very small weights
        if abs(weights[i]) < 0.01:
            continue

        # For normalized data, get feature range
        feature_min = min(normal_df[feature].min(), fault_df[feature].min())
        feature_max = max(normal_df[feature].max(), fault_df[feature].max())

        # Calculate average contributions from other features
        other_features_contribution = 0
        for j, other_feature in enumerate(features):
            if j != i:
                # Use average of means from normal and fault
                normal_mean = normal_df[other_feature].mean()
                fault_mean = fault_df[other_feature].mean()
                avg_val = (normal_mean + fault_mean) / 2
                other_features_contribution += weights[j] * avg_val

        # Calculate raw threshold where z = 0 (p = 0.5)
        raw_threshold = (-bias - other_features_contribution) / weights[i]

        # Constrain threshold to actual data range
        constrained_threshold = max(feature_min, min(feature_max, raw_threshold))

        # If raw threshold is far outside data range, use midpoint between means
        if (raw_threshold < feature_min - 0.5) or (raw_threshold > feature_max + 0.5):
            normal_mean = normal_df[feature].mean()
            fault_mean = fault_df[feature].mean()
            constrained_threshold = (normal_mean + fault_mean) / 2

        # Store threshold with context
        normal_mean = normal_df[feature].mean()
        fault_mean = fault_df[feature].mean()
        thresholds[feature] = {
            'value': constrained_threshold,
            'normal_mean': normal_mean,
            'fault_mean': fault_mean,
            'raw_threshold': raw_threshold
        }

    return thresholds


def evaluate_model(weights, bias, features, normal_df, fault_df):
    """
    Evaluate model accuracy and generate confusion matrix
    """
    # Prepare data
    normal_X = normal_df[features].values
    normal_y = np.zeros(len(normal_df))

    fault_X = fault_df[features].values
    fault_y = np.ones(len(fault_df))

    # Combine data
    X = np.vstack([normal_X, fault_X])
    y = np.hstack([normal_y, fault_y])

    # Make predictions
    z = np.dot(X, weights) + bias
    predictions = sigmoid(z)
    predicted_labels = (predictions >= 0.5).astype(int)

    # Calculate accuracy
    accuracy = np.mean(predicted_labels == y)

    # Calculate confusion matrix
    tp = np.sum((predicted_labels == 1) & (y == 1))
    fp = np.sum((predicted_labels == 1) & (y == 0))
    tn = np.sum((predicted_labels == 0) & (y == 0))
    fn = np.sum((predicted_labels == 0) & (y == 1))

    confusion_matrix = {
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

    return accuracy, confusion_matrix


def plot_results(normal_df, fault_df, weights, bias, features, cost_history, thresholds, output_dir):
    """Generate visualizations for the analysis"""
    # Setup figure
    plt.figure(figsize=(20, 15))

    # 1. Feature Importance Plot
    plt.subplot(2, 2, 1)
    feature_weights = [(f, w) for f, w in zip(features, weights)]
    feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)

    features_sorted = [f[0] for f in feature_weights]
    importance = [f[1] for f in feature_weights]

    plt.barh(features_sorted, importance)
    plt.title('Feature Importance in Fault Detection', size=14)
    plt.xlabel('Weight Value')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # 2. Cost history
    plt.subplot(2, 2, 2)
    plt.plot(range(0, len(cost_history) * 100, 100), cost_history, '-o')
    plt.title('Gradient Descent Convergence', size=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. Top two features scatter plot
    if len(features) >= 2:
        plt.subplot(2, 2, 3)
        top_features = [f[0] for f in feature_weights[:2]]
        f1, f2 = top_features[0], top_features[1]

        plt.scatter(normal_df[f1], normal_df[f2], alpha=0.5, label='Normal')
        plt.scatter(fault_df[f1], fault_df[f2], alpha=0.5, label='Fault')

        # Draw threshold lines
        if f1 in thresholds:
            threshold_value = thresholds[f1]['value']
            direction = ">" if weights[features.index(f1)] > 0 else "<"
            plt.axvline(x=threshold_value, color='r', linestyle='--',
                        label=f'Threshold {f1} {direction} {threshold_value:.4f}')

        if f2 in thresholds:
            threshold_value = thresholds[f2]['value']
            direction = ">" if weights[features.index(f2)] > 0 else "<"
            plt.axhline(y=threshold_value, color='g', linestyle='--',
                        label=f'Threshold {f2} {direction} {threshold_value:.4f}')

        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title(f'Decision Boundaries: {f1} vs {f2}', size=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

    # 4. Feature distributions for top feature
    plt.subplot(2, 2, 4)
    top_feature = feature_weights[0][0]

    # Plot histograms with KDE
    sns.histplot(normal_df[top_feature], color='green', label='Normal',
                 alpha=0.5, kde=True, stat="density")
    sns.histplot(fault_df[top_feature], color='red', label='Fault',
                 alpha=0.5, kde=True, stat="density")

    if top_feature in thresholds:
        threshold_value = thresholds[top_feature]['value']
        plt.axvline(x=threshold_value, color='k', linestyle='--',
                    label=f'Threshold: {threshold_value:.4f}')

        # Also mark the means
        plt.axvline(x=thresholds[top_feature]['normal_mean'], color='green', linestyle=':',
                    label=f'Normal Mean: {thresholds[top_feature]["normal_mean"]:.4f}')
        plt.axvline(x=thresholds[top_feature]['fault_mean'], color='red', linestyle=':',
                    label=f'Fault Mean: {thresholds[top_feature]["fault_mean"]:.4f}')

    plt.title(f'Distribution of Most Important Feature: {top_feature}', size=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Overall title
    plt.suptitle('Gradient Descent Fault Analysis', size=16, y=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / 'gradient_descent_fault_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Analysis visualization saved to: {output_path}")
    plt.close()

    # Create a second visualization focusing on all features
    plot_feature_distributions(normal_df, fault_df, thresholds, output_dir)


def plot_feature_distributions(normal_df, fault_df, thresholds, output_dir):
    """
    Create a visualization showing distributions for all features with thresholds
    """
    features = [col for col in normal_df.columns if col != 'label']
    n_features = len(features)

    # Calculate grid dimensions
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 4 * n_rows))

    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)

        # Plot histograms with KDE
        sns.histplot(normal_df[feature], color='green', label='Normal',
                     alpha=0.5, kde=True, stat="density")
        sns.histplot(fault_df[feature], color='red', label='Fault',
                     alpha=0.5, kde=True, stat="density")

        if feature in thresholds:
            # Add threshold line
            threshold_value = thresholds[feature]['value']
            plt.axvline(x=threshold_value, color='k', linestyle='--',
                        label=f'Threshold: {threshold_value:.4f}')

        plt.title(f'Distribution: {feature}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    output_path = output_dir / 'feature_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Feature distributions saved to: {output_path}")
    plt.close()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find point of failure using gradient descent')
    parser.add_argument('normal_file_path',
                        help='Path to CSV file with normal behavior data (fault_0)')
    parser.add_argument('fault_file_path',
                        help='Path to CSV file with fault behavior data (fault_1_to_3)')

    args = parser.parse_args()

    # Run the analysis
    gradient_descent_fault_detector(args.normal_file_path, args.fault_file_path)


if __name__ == "__main__":
    # Example usage with file paths similar to the provided example:
    # You can either use these default paths or provide paths as command line arguments
    DEFAULT_NORMAL_PATH = (
        r"C:\Users\polon\PycharmProjects\Dataa\Prod\CSV_DATA_SETS\Main_Secondary_"
             r"DataSet\Filtered_Fault_0_main_secondary_Data_Set.prod.csv")

    DEFAULT_FAULT_PATH = (
        r"C:\Users\polon\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_"
             r"DataSet\Filtered_Fault_1_to_3_main_secondary_Data_Set.prod.csv")

    # Check if command line arguments were provided
    import sys

    if len(sys.argv) > 1:
        main()  # Use command line arguments
    else:
        # Use default paths
        print("No command line arguments provided. Using default file paths.")
        gradient_descent_fault_detector(DEFAULT_NORMAL_PATH, DEFAULT_FAULT_PATH)