import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import plotly.express as px
import os


SENSOR_CSV = 'data/sensor_data.csv'
GPS_CSV = 'data/gps_log.csv'
OUTPUT_DIR = 'analysis_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sensor_df = pd.read_csv(SENSOR_CSV, parse_dates=['timestamp'])
gps_df = pd.read_csv(GPS_CSV, parse_dates=['timestamp'])
gps_df = gps_df[['timestamp', 'latitude', 'longitude', 'altitude']]
gps_df.set_index('timestamp', inplace=True)
sensor_df.set_index('timestamp', inplace=True)
sensor_df['latitude'] = gps_df['latitude'].reindex(sensor_df.index, method='nearest', tolerance=pd.Timedelta('5s'))
sensor_df['longitude'] = gps_df['longitude'].reindex(sensor_df.index, method='nearest', tolerance=pd.Timedelta('5s'))
sensor_df['altitude'] = gps_df['altitude'].reindex(sensor_df.index, method='nearest', tolerance=pd.Timedelta('5s'))
sensor_df.dropna(subset=['latitude', 'longitude'], inplace=True)
sensor_df.reset_index(inplace=True)

def refine_gas(raw, temp, humidity):
    temp_correction = 1 + 0.005 * (temp - 25)
    hum_correction = 1 - 0.003 * (humidity - 50)
    return raw / (temp_correction * hum_correction)

for gas in ['CO', 'CH4', 'NOx', 'LPG']:
    sensor_df[f'{gas}_refined'] = sensor_df.apply(
        lambda row: refine_gas(row[gas], row['temperature'], row['humidity']), axis=1
    )
features = sensor_df[[f'{g}_refined' for g in ['CO', 'CH4', 'NOx', 'LPG']]]
model = IsolationForest(contamination=0.05, random_state=42)
sensor_df['anomaly'] = model.fit_predict(features)
sensor_df['anomaly'] = sensor_df['anomaly'].map({1: 'normal', -1: 'anomaly'})
sensor_df.to_csv(os.path.join(OUTPUT_DIR, 'merged_refined_data.csv'), index=False)
sensor_df[sensor_df['anomaly'] == 'anomaly'].to_csv(os.path.join(OUTPUT_DIR, 'anomalies.csv'), index=False)

for gas in ['CO', 'CH4', 'NOx', 'LPG']:
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=sensor_df, x='timestamp', y=f'{gas}_refined', hue='anomaly', palette='Set1')
    plt.title(f'{gas} (Refined) Over Time')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{gas}_timeplot.png'))
    plt.close()

m = folium.Map(location=[sensor_df.latitude.mean(), sensor_df.longitude.mean()], zoom_start=13)
heat_data = [[row['latitude'], row['longitude'], row['CO_refined']] for _, row in sensor_df.iterrows()]
HeatMap(heat_data, radius=10).add_to(m)
m.save(os.path.join(OUTPUT_DIR, 'co_heatmap.html'))
fig = px.scatter_3d(sensor_df,
                    x='latitude', y='longitude', z='CO_refined',
                    color='anomaly',
                    hover_data=['timestamp', 'CH4_refined', 'NOx_refined', 'LPG_refined'],
                    title='3D Refined CO Levels')
fig.write_html(os.path.join(OUTPUT_DIR, '3d_pollution_plot.html'))
