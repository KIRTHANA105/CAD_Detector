from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os, shutil
import psycopg2
from datetime import date
import wfdb
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -------------------- LOAD ENV --------------------
load_dotenv()

# -------------------- POSTGRES CONNECTION --------------------
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    sslmode=os.getenv("DB_SSLMODE")
)
cursor = conn.cursor()

# -------------------- FASTAPI APP --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- PATHS --------------------
UPLOAD_DIR = "uploads"
MODEL_DIR = "model"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "cad_vs_norm_cnn_lstm_feat_improved.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "tabular_scaler_improved.joblib")

# -------------------- LOAD MODEL --------------------
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully.")
except Exception as e:
    print(f"⚠️ Model/scaler load error: {e}")
    model = None
    scaler = None


# -------------------- UPLOAD ENDPOINT --------------------
@app.post("/upload")
async def upload_ecg(
    dat_file: UploadFile = File(...),   # mandatory
    hea_file: UploadFile = File(None),  # optional
    patient_name: str = Form("Unknown"),
    patient_id: str = Form("")
):
    # ---- Save Files ----
    dat_path = os.path.join(UPLOAD_DIR, dat_file.filename)
    with open(dat_path, "wb") as f:
        shutil.copyfileobj(dat_file.file, f)

    hea_path = None
    lead, sampling_rate = None, None
    header_info = {}

    if hea_file:
        hea_path = os.path.join(UPLOAD_DIR, hea_file.filename)
        with open(hea_path, "wb") as f:
            shutil.copyfileobj(hea_file.file, f)

        try:
            # Read header details (without .hea extension)
            header = wfdb.rdheader(os.path.splitext(hea_path)[0])
            lead = header.sig_name[0] if header.sig_name else None
            sampling_rate = header.fs

            header_info = {
                "record_name": header.record_name,
                "num_signals": header.n_sig,
                "signal_names": header.sig_name,
                "sampling_rate": header.fs,
                "base_date": str(header.base_date) if header.base_date else None,
                "base_time": str(header.base_time) if header.base_time else None,
                "adc_gain": header.adc_gain.tolist() if hasattr(header.adc_gain, "tolist") else header.adc_gain,
                "baseline": header.baseline.tolist() if hasattr(header.baseline, "tolist") else header.baseline,
                "units": header.units,
            }

            print("📘 Header Info:", header_info)

        except Exception as e:
            print(f"⚠️ Header read error: {e}")
            header_info = {"error": str(e)}

    # ---- Save Basic Info to DB ----
        # ---- Save Basic Info to DB ----
    cursor.execute("""
        INSERT INTO ecg_records 
        (patient_name, dat_link, hea_link, lead, sampling_rate, recording_date)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (patient_name, dat_path, hea_path, lead, sampling_rate, date.today()))
    ecg_id = cursor.fetchone()[0]
    conn.commit()

    # ---- Prediction ----
    prediction = "Unknown"

    if model and scaler:
        try:
            record = wfdb.rdrecord(os.path.splitext(dat_path)[0])
            sig = record.p_signal[:, 0]
            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

            seq_len = 1000
            if len(sig) < seq_len:
                sig = np.pad(sig, (0, seq_len - len(sig)))
            else:
                sig = sig[:seq_len]

            X_seq = np.expand_dims(sig, axis=(0, -1))
            X_tab = np.zeros((1, scaler.mean_.shape[0]))
            X_tab = scaler.transform(X_tab)

            pred_prob = model.predict({"seq_input": X_seq, "tab_input": X_tab})[0][0]
            prob = float(pred_prob)
            prediction = "Coronary Artery Disease" if pred_prob >= 0.5 else "Normal"

        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            prediction = "Prediction Error"
    else:
        prediction = "Model Not Loaded"

    # ---- Update DB ----
    cursor.execute(
        "UPDATE ecg_records SET prediction=%s WHERE id=%s",
        (prediction, ecg_id)
    )
    conn.commit()

    # ---- Return Response ----
    return {
        "message": "Upload successful",
        "ecg_id": ecg_id,
        "patient_name": patient_name,
        "dat_link": dat_path,
        "hea_link": hea_path,
        "header_info": header_info,
        "prediction": prediction,
        "probability": prob if model else None
    }

# -------------------- REPORTS --------------------
@app.get("/reports/")
def get_reports():
    cursor.execute("""
        SELECT id, patient_name, patient_id, dat_link, hea_link, lead, 
               sampling_rate, recording_date, prediction, severity
        FROM ecg_records
        ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    return {
        "reports": [
            {
                "id": r[0], "patient_name": r[1], "patient_id": r[2], "dat_link": r[3], "hea_link": r[4],
                "lead": r[5], "sampling_rate": r[6],
                "recording_date": str(r[7]), "prediction": r[8], "severity": r[9]
            } for r in rows
        ]
    }


# -------------------- ROOT --------------------
@app.get("/")
def root():
    return {"status": "Backend Running ✅", "endpoints": ["/upload", "/reports/"]}
