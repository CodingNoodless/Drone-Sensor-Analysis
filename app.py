import os
import traceback
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from merge_refine import run_merge       # merge_refine.py: run_merge(input1, input2, output_dir)
from plume_visualization import main     # plume_visualization.py: main(csv_path, out_dir)

# Configuration
data_dir = "data"
UPLOAD_FOLDER = data_dir           # upload sensor and GPS files here
MERGED_DIR = "analysis_output"    # default merge output dir used by merge_refine
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
    # Expect two CSVs: sensor_data.csv and gps_log.csv
    files = request.files.getlist('csvs')
    if len(files) != 2:
        return {"error": "Please upload exactly two CSV files."}, 400

    # Clear old uploaded data
    for fname in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, fname))
        except:
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

    # Merge & refine into analysis_output
    try:
        run_merge(input_paths[0], input_paths[1], MERGED_DIR)
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Merging failed: {e}"}, 500

    # Generate plume visualizations from merged CSV
    merged_csv = os.path.join(MERGED_DIR, 'merged_refined_data.csv')
    try:
        main(merged_csv, STATIC_PLUMES)
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Visualization failed: {e}"}, 500

    # Respond with pollutant URLs
    pollutants = ["CO_refined", "CH4_refined", "NOx_refined", "LPG_refined"]
    urls = {
        p: url_for('static', filename=f'final_plumes/{p}.html')
        for p in pollutants
    }
    return {"pollutants": pollutants, "urls": urls}


if __name__ == '__main__':
    app.run(debug=True)