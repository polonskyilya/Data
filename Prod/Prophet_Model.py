import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet


def prepare_prophet_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        print(f"Columns found in {csv_file}: {df.columns.tolist()}")

        timestamp_cols = [col for col in df.columns if col.lower() in ['timestamp', 'ts']]
        if not timestamp_cols:
            raise ValueError(f"No timestamp column found in {csv_file}. Available columns: {df.columns.tolist()}")

        timestamp_col = timestamp_cols[0]
        df['ds'] = pd.to_datetime(df[timestamp_col])
        df = df.dropna(subset=['ds'])
        return df
    except Exception as e:
        print(f"Error reading file {csv_file}: {str(e)}")
        raise


class MultiProphetModel:
    def __init__(self):
        self.models = {}
        self.data = {}

    def train(self, normal_csv, fault_csv):
        normal_data = prepare_prophet_data(normal_csv)
        fault_data = prepare_prophet_data(fault_csv)
        print(f"Normal data shape: {normal_data.shape}, Fault data shape: {fault_data.shape}")

        for param in ['Voltage_main', 'Current_main', 'Temperature_main']:
            try:
                concat_ds = pd.concat([normal_data['ds'], fault_data['ds']])
                concat_param = pd.concat([normal_data[param], fault_data[param]])
                cap = concat_param.max() * 1.1  # Setting a slightly higher cap for logistic growth
                df = pd.DataFrame({'ds': concat_ds, 'y': concat_param, 'cap': cap})
                df = df.dropna()
                model = Prophet(growth='logistic', changepoint_prior_scale=0.05)
                model.fit(df)
                self.models[param] = model
                self.data[param] = df
                print(f"Trained model for {param}")
            except Exception as e:
                print(f"Error training model for {param}: {str(e)}")
                raise

    def predict_new_data(self, new_data_csv):
        new_data = prepare_prophet_data(new_data_csv)
        pred_df = pd.DataFrame({'ds': new_data['ds']})
        predictions = {}
        trend_rates = {}

        for param, model in self.models.items():
            pred_df['cap'] = self.data[param]['cap'].max()
            forecast = model.predict(pred_df)
            predictions[param] = forecast

            # Calculate trend rate (slope of linear regression on forecast)
            yhat_values = forecast['yhat'].values
            time_index = np.arange(len(yhat_values))
            trend_slope = np.polyfit(time_index, yhat_values, 1)[0]  # Linear trend slope
            trend_rates[param] = trend_slope

        return predictions, trend_rates

    def plot_trends(self, new_data_csv):
        predictions, trend_rates = self.predict_new_data(new_data_csv)
        plt.figure(figsize=(12, 6))

        for param, forecast in predictions.items():
            plt.plot(forecast['ds'], forecast['yhat'], label=f'{param} Trend (Rate: {trend_rates[param]:.4f})')

        plt.axhline(y=7, color='r', linestyle='--', label='Voltage Critical < 7')
        plt.axhline(y=5, color='orange', linestyle='--', label='Current Critical > 5')
        plt.axhline(y=4, color='orange', linestyle='--', label='Current Critical < 4')
        plt.axhline(y=5.8, color='purple', linestyle='--', label='Temperature Critical > 5.8')

        plt.title('Voltage, Current, and Temperature Trends with Rate Analysis')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

        # Print trend rates for decision-making
        print("\nTrend Rate Analysis:")
        for param, rate in trend_rates.items():
            direction = "increasing" if rate > 0 else "decreasing"
            print(f"- {param}: {direction} at rate {rate:.4f} per time step")
            if param == 'Voltage_main' and rate < 0 and min(predictions[param]['yhat']) < 7:
                print("  ⚠ Voltage is decreasing toward critical level!")
            if param == 'Current_main' and (rate > 0 and max(predictions[param]['yhat']) > 5) or (
                    rate < 0 and min(predictions[param]['yhat']) < 4):
                print("  ⚠ Current is moving toward critical level!")
            if param == 'Temperature_main' and rate > 0 and max(predictions[param]['yhat']) > 5.8:
                print("  ⚠ Temperature is increasing beyond safe threshold!")


def main():
    normal_csv = r"C:\Users\Ilya Polonsky\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\TimeS_Filtered_Fault_0_main_Data_Set.prod.csv"
    fault_csv = r"C:\Users\Ilya Polonsky\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\TimeS_Filtered_Fault_1_to_3_main_Data_Set.prod.csv"
    new_data_csv = r"C:\Users\Ilya Polonsky\PycharmProjects\Data\Prod\CSV_DATA_SETS\Main_Secondary_DataSet\NewData_Full_Fault_TimeS_main_Data_Set.prod.csv"

    model = MultiProphetModel()
    try:
        print("Training models...")
        model.train(normal_csv, fault_csv)

        print("\nGenerating trend analysis plot...")
        model.plot_trends(new_data_csv)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()