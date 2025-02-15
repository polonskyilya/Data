import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt


def prepare_prophet_data(csv_file):
    try:
        # Read CSV
        df = pd.read_csv(csv_file)

        # Print columns for debugging
        print(f"Columns found in {csv_file}:")
        print(df.columns.tolist())

        # Find timestamp column (case-insensitive)
        timestamp_col = [col for col in df.columns if col.lower() == 'timestamp'][0]

        # Convert timestamp to datetime
        df['ds'] = pd.to_datetime(df[timestamp_col])
        return df
    except Exception as e:
        print(f"Error reading file {csv_file}:")
        print(f"Detailed error: {str(e)}")
        raise

class MultiProphetModel:
    def __init__(self):
        self.voltage_model = None
        self.current_model = None
        self.temp_model = None
        self.normal_data = None
        self.fault_data = None

    def train(self, normal_csv, fault_csv):
        # Store historical data for plotting
        self.normal_data = prepare_prophet_data(normal_csv)
        self.fault_data = prepare_prophet_data(fault_csv)

        # Create and train voltage model
        voltage_df = pd.DataFrame({
            'ds': pd.concat([self.normal_data['ds'], self.fault_data['ds']]),
            'y': pd.concat([self.normal_data['Voltage_main'],
                            self.fault_data['Voltage_main']])
        })
        self.voltage_model = Prophet(changepoint_prior_scale=0.05)
        self.voltage_model.fit(voltage_df)

        # Create and train current model
        current_df = pd.DataFrame({
            'ds': pd.concat([self.normal_data['ds'], self.fault_data['ds']]),
            'y': pd.concat([self.normal_data['Current_main'],
                            self.fault_data['Current_main']])
        })
        self.current_model = Prophet(changepoint_prior_scale=0.05)
        self.current_model.fit(current_df)

        # Create and train temperature model
        temp_df = pd.DataFrame({
            'ds': pd.concat([self.normal_data['ds'], self.fault_data['ds']]),
            'y': pd.concat([self.normal_data['Temperature_main'],
                            self.fault_data['Temperature_main']])
        })
        self.temp_model = Prophet(changepoint_prior_scale=0.05)
        self.temp_model.fit(temp_df)

    def predict_new_data(self, new_data_csv):
        """Predict fault probability for new data"""
        # Read and prepare new data
        new_data = prepare_prophet_data(new_data_csv)

        # Create prediction dataframes
        pred_df = pd.DataFrame({'ds': new_data['ds']})

        # Make predictions
        voltage_forecast = self.voltage_model.predict(pred_df)
        current_forecast = self.current_model.predict(pred_df)
        temp_forecast = self.temp_model.predict(pred_df)

        # Check critical conditions
        conditions = self.check_critical_conditions(voltage_forecast, current_forecast, temp_forecast)

        # Calculate fault probability
        fault_prob = self.calculate_fault_probability(conditions)

        return voltage_forecast, current_forecast, temp_forecast, fault_prob

    def check_critical_conditions(self, voltage_forecast, current_forecast, temp_forecast):
        """Check against normalized critical thresholds"""
        critical_conditions = {
            'voltage_critical': voltage_forecast['yhat'] < 0.05,
            'current_critical': current_forecast['yhat'] < 0.04,
            'temp_critical': temp_forecast['yhat'] > 0.07
        }
        return critical_conditions

    def calculate_fault_probability(self, conditions):
        """Calculate probability of fault based on critical conditions"""
        # Count how many metrics are in critical range
        fault_indicators = np.sum([conditions[key].astype(int) for key in conditions], axis=0)
        # Convert to probability (0-1 range)
        fault_prob = fault_indicators / 3.0
        return fault_prob

    def plot_predictions(self, new_data_csv):
        """Plot historical data and predictions for new data"""
        # Get predictions for new data
        v_forecast, c_forecast, t_forecast, fault_prob = self.predict_new_data(new_data_csv)

        # Create subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))

        # Plot Voltage
        ax1.plot(self.normal_data['ds'], self.normal_data['Voltage_main'], 'g.', label='Normal', alpha=0.5)
        ax1.plot(self.fault_data['ds'], self.fault_data['Voltage_main'], 'r.', label='Fault', alpha=0.5)
        ax1.plot(v_forecast['ds'], v_forecast['yhat'], 'b-', label='Prediction')
        ax1.fill_between(v_forecast['ds'], v_forecast['yhat_lower'], v_forecast['yhat_upper'], color='b', alpha=0.2)
        ax1.axhline(y=0.05, color='r', linestyle='--', label='Critical Threshold')
        ax1.set_title('Voltage Predictions')
        ax1.legend()

        # Plot Current
        ax2.plot(self.normal_data['ds'], self.normal_data['Current_main'], 'g.', label='Normal', alpha=0.5)
        ax2.plot(self.fault_data['ds'], self.fault_data['Current_main'], 'r.', label='Fault', alpha=0.5)
        ax2.plot(c_forecast['ds'], c_forecast['yhat'], 'b-', label='Prediction')
        ax2.fill_between(c_forecast['ds'], c_forecast['yhat_lower'], c_forecast['yhat_upper'], color='b', alpha=0.2)
        ax2.axhline(y=0.04, color='r', linestyle='--', label='Critical Threshold')
        ax2.set_title('Current Predictions')
        ax2.legend()

        # Plot Temperature
        ax3.plot(self.normal_data['ds'], self.normal_data['Temperature_main'], 'g.', label='Normal', alpha=0.5)
        ax3.plot(self.fault_data['ds'], self.fault_data['Temperature_main'], 'r.', label='Fault', alpha=0.5)
        ax3.plot(t_forecast['ds'], t_forecast['yhat'], 'b-', label='Prediction')
        ax3.fill_between(t_forecast['ds'], t_forecast['yhat_lower'], t_forecast['yhat_upper'], color='b', alpha=0.2)
        ax3.axhline(y=0.07, color='r', linestyle='--', label='Critical Threshold')
        ax3.set_title('Temperature Predictions')
        ax3.legend()

        # Plot Fault Probability
        ax4.plot(v_forecast['ds'], fault_prob, 'k-', label='Fault Probability')
        ax4.fill_between(v_forecast['ds'], np.zeros_like(fault_prob), fault_prob, color='r', alpha=0.2)
        ax4.set_title('Fault Probability')
        ax4.set_ylim(0, 1)
        ax4.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # File paths with raw strings
    normal_csv = r"C:\Users\Ilya Polonsky\PycharmProjects\Data\CSV_DATA_SETS\Prediction_Fault_DataSet\Main_Secondary_DataSet\Normalized\Timestamped_normalized_minmax_Fault_0_WO_Fault_L_main_secondary_Voltage_Current_Temp_Only_dataSet.csv"
    fault_csv = r"C:\Users\Ilya Polonsky\PycharmProjects\Data\CSV_DATA_SETS\Prediction_Fault_DataSet\Main_Secondary_DataSet\Normalized\Timestamped_normalized_minmax_Fault_1_to_3_WO_Fault_L_main_secondary_Voltage_Current_Temp_Only_dataSet.csv"

    model = MultiProphetModel()

    try:
        print("Training models...")
        model.train(normal_csv, fault_csv)

        new_data_csv = input("Enter path to new data CSV file: ")
        print("\nAnalyzing new data and generating plots...")
        model.plot_predictions(new_data_csv)

    except Exception as e:
        print(f"An error occurred: {str(e)}")