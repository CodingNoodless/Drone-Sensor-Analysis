import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# 1) Load full dataset and isolate last timestamp
df_full = pd.read_csv("analysis_output/merged_refined_data.csv", parse_dates=["timestamp"])
last_t   = df_full["timestamp"].max()
df_last  = df_full[df_full["timestamp"] == last_t]

# 2) Build a shared 3D grid for interpolation (full bounding box)
nx, ny, nz = 50, 50, 25
lon_lin = np.linspace(df_full.longitude.min(), df_full.longitude.max(), nx)
lat_lin = np.linspace(df_full.latitude.min(),  df_full.latitude.max(),  ny)
alt_lin = np.linspace(df_full.altitude.min(),  df_full.altitude.max(),  nz)
grid_lon, grid_lat, grid_alt = np.meshgrid(lon_lin, lat_lin, alt_lin, indexing="ij")
pollutants = ["CO_refined","CH4_refined","NOx_refined","LPG_refined"]
vol_traces = []
dot_traces = []

# colormap from yellow to red
def get_colorscale():
    return [
        [0.0,  "green"],
        [0.2,  "green"],
        [0.2,  "yellow"],
        [0.4,  "yellow"],
        [0.4,  "orange"],
        [0.6,  "orange"],
        [0.6,  "red"],
        [0.8,  "red"],
        [0.8,  "darkred"],
        [1.0,  "darkred"]
    ]



for p in pollutants:
    # interpolate as before...
    pts_all  = df_full[["longitude","latitude","altitude"]].values
    vals_all = df_full[p].values

    lin = griddata(pts_all, vals_all,
                   (grid_lon, grid_lat, grid_alt), method="linear")
    nn  = griddata(pts_all, vals_all,
                   (grid_lon, grid_lat, grid_alt), method="nearest")
    lin[np.isnan(lin)] = nn[np.isnan(lin)]

    norm = (lin - np.nanmin(lin)) / (np.nanmax(lin) - np.nanmin(lin) + 1e-9)
    smooth = gaussian_filter(norm, sigma=0.7)

    vol_traces.append(go.Volume(
    x=grid_lon.ravel(),
    y=grid_lat.ravel(),
    z=grid_alt.ravel(),
    value=smooth.ravel(),
    isomin=0.05,
    isomax=1.0,
    opacity=0.3,              # more transparent
    surface_count=40,         # more layers = smoother blobs
    colorscale=get_colorscale(),
    caps=dict(x_show=False, y_show=False, z_show=False),
    showscale=True,
    colorbar=dict(title=f"{p} (norm)", len=0.7, x=1.05),
    visible=(p == pollutants[0])
    ))





    # d) raw dots overlay at last time (optional)
    pts_last  = df_last[["longitude","latitude","altitude"]].values
    vals_last = df_last[p].values
    norm_last = (vals_last - vals_last.min()) / (vals_last.max() - vals_last.min() + 1e-9)

    dot_traces.append(go.Scatter3d(
        x=pts_last[:,0], y=pts_last[:,1], z=pts_last[:,2],
        mode="markers",
        marker=dict(
            size=4,
            color=norm_last,
            colorscale=get_colorscale(),
            cmin=0, cmax=1
        ),
        name=f"{p} raw",
        visible=False,
        legendgroup=p
    ))

# 3) Assemble figure
fig = go.Figure(data=vol_traces + dot_traces)
gfig = go.Figure(data=vol_traces + dot_traces)

# 4) Dropdown to pick pollutant
n = len(pollutants)
dropdown_buttons = []
for i, p in enumerate(pollutants):
    # volume traces first n, dot traces next n
    vis = [False] * (2 * n)
    vis[i] = True            # show this pollutant volume
    # preserve raw dot toggle state
    vis[n + i] = gfig.data[n + i].visible
    dropdown_buttons.append(dict(
        label=p.replace("_", " "),
        method="update",
        args=[
            {"visible": vis},
            {"scene": dict(
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                zaxis_title="Altitude (m)",
                xaxis=dict(range=[df_full.longitude.min(), df_full.longitude.max()]),
                yaxis=dict(range=[df_full.latitude.min(), df_full.latitude.max()]),
                zaxis=dict(range=[df_full.altitude.min(), df_full.altitude.max()])
            )}
        ]
    ))

# 5) Buttons to toggle raw dots
toggle_buttons = [
    dict(
        label="Show Raw Dots",
        method="update",
        args=[{"visible": [
            True if idx >= n else gfig.data[idx].visible
            for idx in range(2 * n)
        ]}]
    ),
    dict(
        label="Hide Raw Dots",
        method="update",
        args=[{"visible": [
            False if idx >= n else gfig.data[idx].visible
            for idx in range(2 * n)
        ]}]
    )
]

gfig.update_layout(
    updatemenus=[
        dict(buttons=dropdown_buttons,
             direction="down", x=0.0, y=1.15,
             xanchor="left", yanchor="top"),
        dict(buttons=toggle_buttons,
             direction="right", x=0.3, y=1.15,
             xanchor="left", yanchor="top")
    ],
    margin=dict(l=0, r=200, t=0, b=0)
)

# 6) Export HTML
gfig.write_html("final_plume.html", auto_open=True)