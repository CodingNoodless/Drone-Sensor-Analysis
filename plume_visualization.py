import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import os

def get_colorscale():
    return [
        [0.0, "green"], [0.2, "green"],
        [0.2, "yellow"], [0.4, "yellow"],
        [0.4, "orange"], [0.6, "orange"],
        [0.6, "red"],    [0.8, "red"],
        [0.8, "darkred"], [1.0, "darkred"]
    ]

def main(input_csv="analysis_output/merged_refined_data.csv", out_dir="final_plumes"):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv, parse_dates=["timestamp"])

    nx, ny, nz = 50, 50, 25
    lon_lin = np.linspace(df.longitude.min(), df.longitude.max(), nx)
    lat_lin = np.linspace(df.latitude.min(), df.latitude.max(), ny)
    alt_lin = np.linspace(df.altitude.min(), df.altitude.max(), nz)
    grid_lon, grid_lat, grid_alt = np.meshgrid(lon_lin, lat_lin, alt_lin, indexing="ij")

    for pollutant in ["CO_refined", "CH4_refined", "NOx_refined", "LPG_refined"]:
        pts = df[["longitude", "latitude", "altitude"]].values
        vals = df[pollutant].values
        vol_lin = griddata(pts, vals, (grid_lon, grid_lat, grid_alt), method="linear")
        vol_nn = griddata(pts, vals, (grid_lon, grid_lat, grid_alt), method="nearest")
        vol_lin[np.isnan(vol_lin)] = vol_nn[np.isnan(vol_lin)]

        norm = (vol_lin - np.nanmin(vol_lin)) / (np.nanmax(vol_lin) - np.nanmin(vol_lin) + 1e-9)
        smooth = gaussian_filter(norm, sigma=0.7)

        fig = go.Figure(go.Volume(
            x=grid_lon.ravel(), y=grid_lat.ravel(), z=grid_alt.ravel(),
            value=smooth.ravel(),
            isomin=0.05, isomax=1.0,
            opacity=0.2,
            surface_count=50,
            colorscale=get_colorscale(),
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=True,
            colorbar=dict(title=f"{pollutant} (norm)", len=0.7, x=1.05)
        ))

        fig.update_layout(scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Altitude (m)",
            xaxis=dict(range=[df.longitude.min(), df.longitude.max()]),
            yaxis=dict(range=[df.latitude.min(), df.latitude.max()]),
            zaxis=dict(range=[df.altitude.min(), df.altitude.max()])
        ), margin=dict(l=0, r=200, t=0, b=0))

        out_path = os.path.join(out_dir, f"{pollutant}.html")
        fig.write_html(out_path)
        print(f"✅ Exported {out_path}")

if __name__ == "__main__":
    main()