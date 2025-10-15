import os
import traceback
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

# Use auto-detecting merge function
from merge_refine import run_merge_auto   # <-- IMPORTANT: auto-detects sensor vs GPS
from plume_visualization import main      # plume_visualization.py: main(csv_path, out_dir)

# Configuration
data_dir = "data"
UPLOAD_FOLDER = data_dir           # upload sensor and GPS files here
MERGED_DIR = "analysis_output"     # default merge output dir used by merge_refine
STATIC_PLUMES = "static/final_plumes"
ALLOWED_EXT = {"csv"}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(STATIC_PLUMES, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """
    Endpoint accepts exactly two CSV files (order doesn't matter).
    It saves them into data/, auto-detects sensor vs GPS, runs the merge,
    runs visualization, and returns static URLs to the generated HTML plots.
    """

    files = request.files.getlist('csvs')
    if len(files) != 2:
        return {"error": "Please upload exactly two CSV files (sensor + GPS)."}, 400

    # Clear old uploaded data
    for fname in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, fname))
        except Exception:
            pass

    # Save new uploads to data directory
    input_paths = []
    for f in files:
        if f.filename == '' or not allowed_file(f.filename):
            return {"error": "Invalid file type."}, 400
        filename = secure_filename(f.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(save_path)
        input_paths.append(save_path)

    # Diagnostic logging (helpful for debugging upload order / header issues)
    print("Saved upload paths:", input_paths)
    for p in input_paths:
        try:
            import pandas as _pd
            hdr = _pd.read_csv(p, nrows=0, skipinitialspace=True).columns.tolist()
            print("Header for", p, ":", hdr)
        except Exception as e:
            print("Could not read header for", p, ":", e)

    # Merge & refine into MERGED_DIR using auto-detection (order-agnostic)
    try:
        run_merge_auto(input_paths[0], input_paths[1], MERGED_DIR)
    except Exception as e:
        # Print traceback to server console and return helpful error for dev debugging
        tb = traceback.format_exc()
        print("ERROR during merge:", e)
        print(tb)
        # Return the error + stacktrace in JSON (dev mode). Remove stacktrace in prod.
        return {"error": f"Merging failed: {e}", "traceback": tb}, 500

    # Generate plume visualizations from merged CSV
    merged_csv = os.path.join(MERGED_DIR, 'merged_refined_data.csv')
    try:
        main(merged_csv, STATIC_PLUMES)
    except Exception as e:
        tb = traceback.format_exc()
        print("ERROR during visualization:", e)
        print(tb)
        return {"error": f"Visualization failed: {e}", "traceback": tb}, 500

    # Respond with pollutant URLs
    pollutants = ["CO_refined", "CH4_refined", "NOx_refined", "LPG_refined"]
    urls = {
        p: url_for('static', filename=f'final_plumes/{p}.html')
        for p in pollutants
    }
    return {"pollutants": pollutants, "urls": urls}


if __name__ == '__main__':
    app.run(debug=True)
