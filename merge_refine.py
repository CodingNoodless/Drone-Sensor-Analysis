import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def run_merge(sensor_csv: str, gps_csv: str, output_dir: str):
    """
    Reads sensor_csv and gps_csv, merges on nearest timestamps,
    refines gas readings, detects anomalies, and writes out:
      - merged_refined_data.csv
      - anomalies.csv
    into output_dir.
    """

    # 1) Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 2) Load & index
    sensor_df = (
        pd.read_csv(sensor_csv, parse_dates=['timestamp'])
          .set_index('timestamp')
    )
    gps_df = (
        pd.read_csv(gps_csv, parse_dates=['timestamp'])
          .set_index('timestamp')[['latitude','longitude','altitude']]
    )

    # 3) Join nearest GPS within 5s tolerance
    sensor_df['latitude']  = gps_df['latitude'] .reindex(sensor_df.index, method='nearest', tolerance=pd.Timedelta('5s'))
    sensor_df['longitude'] = gps_df['longitude'].reindex(sensor_df.index, method='nearest', tolerance=pd.Timedelta('5s'))
    sensor_df['altitude']  = gps_df['altitude'] .reindex(sensor_df.index, method='nearest', tolerance=pd.Timedelta('5s'))

    sensor_df.dropna(subset=['latitude','longitude'], inplace=True)
    sensor_df.reset_index(inplace=True)

    # 4) Gas refinement helper
    def refine_gas(raw, temp, humidity):
        temp_corr = 1 + 0.005 * (temp - 25)
        hum_corr  = 1 - 0.003 * (humidity - 50)
        return raw / (temp_corr * hum_corr)

    # 5) Apply refinement to each pollutant
    for gas in ['CO','CH4','NOx','LPG']:
        sensor_df[f'{gas}_refined'] = sensor_df.apply(
            lambda row: refine_gas(row[gas], row['temperature'], row['humidity']),
            axis=1
        )

    # 6) Anomaly detection
    features = sensor_df[[f'{g}_refined' for g in ['CO','CH4','NOx','LPG']]]
    model = IsolationForest(contamination=0.05, random_state=42)
    sensor_df['anomaly'] = model.fit_predict(features)
    sensor_df['anomaly'] = sensor_df['anomaly'].map({1: 'normal', -1: 'anomaly'})

    # 7) Drop raw pollutant columns before saving
    sensor_df.drop(columns=['CO', 'CH4', 'NOx', 'LPG'], inplace=True)

    # 8) Write out CSVs
    merged_path    = os.path.join(output_dir, 'merged_refined_data.csv')
    anomalies_path = os.path.join(output_dir, 'anomalies.csv')

    sensor_df.to_csv(merged_path, index=False)
    sensor_df[sensor_df['anomaly']=='anomaly'].to_csv(anomalies_path, index=False)

    print(f"Saved merged data to {merged_path}")
    print(f"Saved anomalies to {anomalies_path}")

# Add this so the script can be run directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and refine sensor and GPS data.")
    parser.add_argument("--sensor_csv", required=True, help="Path to sensor CSV file")
    parser.add_argument("--gps_csv", required=True, help="Path to GPS CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files")

    args = parser.parse_args()
    run_merge(args.sensor_csv, args.gps_csv, args.output_dir)
