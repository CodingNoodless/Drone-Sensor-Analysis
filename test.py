import pandas as pd

gps_csv = r"C:\Users\getan\Documents\GitHub\Drone-Sensor-Analysis\data\gps_log.csv"
sensor_csv = r"C:\Users\getan\Documents\GitHub\Drone-Sensor-Analysis\data\sensor_data.csv"

print("GPS CSV header:", pd.read_csv(gps_csv, nrows=1).columns.tolist())
print("Sensor CSV header:", pd.read_csv(sensor_csv, nrows=1).columns.tolist())
