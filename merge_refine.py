# merge_refine.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# ---------------- Utilities ----------------
def _read_header_set(path: str):
    """Return set of lowercased header names for fast inspection."""
    try:
        cols = pd.read_csv(path, nrows=0, skipinitialspace=True).columns.tolist()
    except Exception as e:
        raise RuntimeError(f"Unable to read header from {path}: {e}")
    return {c.strip().lower() for c in cols}

def _normalize_cols(df: pd.DataFrame):
    """Strip and lowercase column names in-place (returns df)."""
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _map_gps_columns(cols):
    """
    Return a mapping to normalize GPS-ish column names to
    ('latitude','longitude','altitude') when possible.
    """
    mapping = {}
    for c in cols:
        low = c.lower()
        if 'lat' in low and 'latitude' not in mapping.values():
            mapping[c] = 'latitude'
        elif ('lon' in low or 'lng' in low or (low == 'long')) and 'longitude' not in mapping.values():
            mapping[c] = 'longitude'
        elif ('alt' in low or 'elev' in low or 'height' in low) and 'altitude' not in mapping.values():
            mapping[c] = 'altitude'
    return mapping

def _preview_file(path, n=8):
    """Print first n lines + pandas header for debugging (safe to call)."""
    print(f"--- PREVIEW {path} ---")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                print(f"{i:03d}: {line.rstrip()}")
                if i >= n - 1:
                    break
    except Exception as e:
        print("Could not print file body:", e)
    try:
        hdr = pd.read_csv(path, nrows=0, skipinitialspace=True).columns.tolist()
        print("PANDAS HEADER:", hdr)
    except Exception as e:
        print("Could not read header with pandas:", e)
    print(f"--- END PREVIEW {path} ---\n")


# ---------------- Core merge/refine ----------------
def run_merge(sensor_csv: str, gps_csv: str, output_dir: str):
    """
    Merge sensor CSV with GPS CSV (sensor timestamps matched to nearest GPS within 5s),
    refine pollutant readings for temperature/humidity, run an IsolationForest anomaly detector,
    and write merged_refined_data.csv and anomalies.csv to output_dir.

    Expects sensor_csv to contain: timestamp, CO/CH4/NOx/LPG, temperature, humidity
    Expects gps_csv to contain: timestamp, latitude, longitude, altitude  (column names flexible)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load sensor CSV
    sensor_df = pd.read_csv(sensor_csv, parse_dates=['timestamp'], skipinitialspace=True)
    sensor_df = _normalize_cols(sensor_df)
    if 'timestamp' not in sensor_df.columns:
        raise ValueError(f"Sensor CSV '{sensor_csv}' missing required 'timestamp' column. Found: {sensor_df.columns.tolist()}")
    sensor_df = sensor_df.set_index('timestamp')

    # Load GPS CSV
    gps_df = pd.read_csv(gps_csv, parse_dates=['timestamp'], skipinitialspace=True)
    gps_df = _normalize_cols(gps_df)
    if 'timestamp' not in gps_df.columns:
        raise ValueError(f"GPS CSV '{gps_csv}' missing required 'timestamp' column. Found: {gps_df.columns.tolist()}")

    # Map GPS-like columns to standard names
    rename_map = _map_gps_columns(gps_df.columns)
    if rename_map:
        gps_df = gps_df.rename(columns=rename_map)

    required = ['latitude', 'longitude', 'altitude']
    missing = [c for c in required if c not in gps_df.columns]
    if missing:
        # helpful diagnostic
        raise ValueError(
            f"GPS CSV '{gps_csv}' missing required columns {missing}. "
            f"Found: {gps_df.columns.tolist()}. If your GPS file uses nonstandard column names, "
            "rename them to include 'lat','lon'/'lng','alt' or similar."
        )

    gps_df = gps_df.set_index('timestamp')[required]

    # Reindex nearest GPS onto sensor timestamps (5s tolerance)
    sensor_df['latitude']  = gps_df['latitude'].reindex(sensor_df.index, method='nearest', tolerance=pd.Timedelta('5s'))
    sensor_df['longitude'] = gps_df['longitude'].reindex(sensor_df.index, method='nearest', tolerance=pd.Timedelta('5s'))
    sensor_df['altitude']  = gps_df['altitude'].reindex(sensor_df.index, method='nearest', tolerance=pd.Timedelta('5s'))

    # Drop rows where lat/lon not found within tolerance
    before = len(sensor_df)
    sensor_df = sensor_df.dropna(subset=['latitude', 'longitude'])
    after = len(sensor_df)
    print(f"Dropped {before - after} sensor rows due to missing GPS within 5s tolerance")

    # Reset index to make timestamp a column again
    sensor_df = sensor_df.reset_index()

    # Gas refinement helper
    def refine_gas(raw, temp, humidity):
        temp_corr = 1 + 0.005 * (temp - 25)
        hum_corr  = 1 - 0.003 * (humidity - 50)
        return raw / (temp_corr * hum_corr)

    pollutants = ['co', 'ch4', 'nox', 'lpg']
    found_pollutants = []
    for g in pollutants:
        if g in sensor_df.columns:
            found_pollutants.append(g)
            # apply refinement safely: if temperature/humidity missing, propagate NaN
            sensor_df[f'{g}_refined'] = sensor_df.apply(
                lambda row: refine_gas(row[g], row['temperature'], row['humidity']) if pd.notnull(row[g]) and pd.notnull(row.get('temperature')) and pd.notnull(row.get('humidity')) else np.nan,
                axis=1
            )
        else:
            print(f"[merge_refine] Warning: pollutant column '{g}' not found in sensor CSV; skipping refinement for it.")

    # If no pollutant columns found, raise error
    if not found_pollutants:
        raise ValueError(f"No pollutant columns found in sensor CSV '{sensor_csv}'. Expected one or more of {pollutants}. Found: {sensor_df.columns.tolist()}")

    # Anomaly detection using refined pollutant features that exist
    feature_cols = [f'{g}_refined' for g in pollutants if f'{g}_refined' in sensor_df.columns]
    features = sensor_df[feature_cols].dropna(how='all')  # allow some NaNs but not all
    if features.shape[0] == 0:
        # no data to train on; mark all as normal
        sensor_df['anomaly'] = 'normal'
        print("[merge_refine] Warning: no valid feature rows for anomaly detection; marking all as 'normal'.")
    else:
        model = IsolationForest(contamination=0.05, random_state=42)
        # For rows with partial NaNs, IsolationForest expects no NaNs -> fill with column mean
        features_filled = features.fillna(features.mean())
        preds = model.fit_predict(features_filled)
        sensor_df['anomaly'] = pd.Series(preds, index=features_filled.index).map({1: 'normal', -1: 'anomaly'})
        # For rows that were completely NaN and not present in features_filled index, set anomaly='normal'
        sensor_df['anomaly'] = sensor_df['anomaly'].fillna('normal')

    # Drop raw pollutant columns (keep refined)
    for g in pollutants:
        if g in sensor_df.columns:
            sensor_df.drop(columns=[g], inplace=True)

    # Write outputs
    merged_path = os.path.join(output_dir, 'merged_refined_data.csv')
    anomalies_path = os.path.join(output_dir, 'anomalies.csv')
    sensor_df.to_csv(merged_path, index=False)
    sensor_df[sensor_df['anomaly'] == 'anomaly'].to_csv(anomalies_path, index=False)

    print(f"[merge_refine] Saved merged data to {merged_path}")
    print(f"[merge_refine] Saved anomalies to {anomalies_path}")


# ---------------- Auto-detect wrapper ----------------
def detect_file_type(path: str):
    """
    Return 'gps' or 'sensor' or None.
    GPS: must contain latitude+longitude+altitude (or variants)
    Sensor: must contain at least one pollutant column (co/ch4/nox/lpg)
    """
    cols = _read_header_set(path)
    gps_req = {'latitude', 'longitude', 'altitude'}
    sensor_sig = {'co', 'ch4', 'nox', 'lpg'}
    if gps_req.issubset(cols):
        return 'gps'
    if sensor_sig.intersection(cols):
        return 'sensor'
    return None


def run_merge_auto(file_a: str, file_b: str, output_dir: str):
    """
    Auto-detect which file is GPS and which is sensor and call run_merge.
    If detection ambiguous, will attempt both orders and provide helpful diagnostics.
    """
    a_type = detect_file_type(file_a)
    b_type = detect_file_type(file_b)
    print(f"[merge_refine] detect_file_type: {file_a} -> {a_type}, {file_b} -> {b_type}")

    # if unambiguous, run and return
    if a_type == 'sensor' and b_type == 'gps':
        return run_merge(sensor_csv=file_a, gps_csv=file_b, output_dir=output_dir)
    if a_type == 'gps' and b_type == 'sensor':
        return run_merge(sensor_csv=file_b, gps_csv=file_a, output_dir=output_dir)

    # Ambiguous: try heuristics / attempts and collect errors
    errors = []
    tries = [
        (file_a, file_b),
        (file_b, file_a)
    ]
    for sensor_file, gps_file in tries:
        print(f"[merge_refine] Trying order sensor='{sensor_file}' gps='{gps_file}'")
        # preview for diagnostics
        _preview_file(sensor_file)
        _preview_file(gps_file)
        try:
            run_merge(sensor_file, gps_file, output_dir)
            print(f"[merge_refine] Success with sensor='{sensor_file}', gps='{gps_file}'")
            return
        except Exception as e:
            print(f"[merge_refine] Attempt failed for sensor={sensor_file}, gps={gps_file}: {e}")
            errors.append((sensor_file, gps_file, str(e)))

    # If we reach here, both attempts failed — raise combined error
    msg_lines = ["Could not auto-detect or merge files. Attempts:"]
    for s, g, err in errors:
        msg_lines.append(f" sensor='{s}' gps='{g}' => error: {err}")
    raise RuntimeError("\n".join(msg_lines))


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge & refine sensor+GPS CSV files.")
    parser.add_argument("file1", help="First CSV file (order doesn't matter with --auto).")
    parser.add_argument("file2", help="Second CSV file (order doesn't matter with --auto).")
    parser.add_argument("--output_dir", "-o", default="analysis_output", help="Output directory")
    parser.add_argument("--auto", action="store_true", help="Auto-detect which file is GPS vs sensor (default tries file1=sensor file2=gps)")
    args = parser.parse_args()

    if args.auto:
        run_merge_auto(args.file1, args.file2, args.output_dir)
    else:
        # try the straightforward ordering (file1 = sensor, file2 = gps)
        try:
            run_merge(args.file1, args.file2, args.output_dir)
        except Exception as e:
            print("run_merge failed for given order (file1 as sensor, file2 as gps):", e)
            print("Trying swapped order...")
            run_merge(args.file2, args.file1, args.output_dir)
