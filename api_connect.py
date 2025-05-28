# api_connect.py

import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from fft_db import compute_fft_db
from predict import load_trained_model, analyze_and_classify_chunks, is_audio_acceptable, is_general_audio_clean
import tempfile
import shutil
import os
import numpy as np
import qrcode
import io
from base64 import b64encode
import librosa

app = FastAPI()


if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_credentials.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

def save_scan_to_firestore(data):
    db.collection("scans").add(data)



# Allow frontend to connect
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory = "static"), name = "static")
model = load_trained_model()

import pickle
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.get("/generate-qr/")
def get_qr_code():
    tone_url = "http://192.168.1.9:8000/tone-page"
    qr_img = qrcode.make(tone_url)
    buffer = io.BytesIO()
    qr_img.save(buffer, format = "PNG")
    base64_qr = b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "qr_base64": base64_qr,
        "tone_url": tone_url
    }


@app.get("/tone-page", response_class = HTMLResponse)
def serve_tone_page():
    return """
    <html>
      <body style="text-align:center; font-family:sans-serif;">
        <h2>🔊 Calibration Tone</h2>
        <p>Playing static tone...</p>
        <audio autoplay controls>
          <source src="/static/calibration_tone.wav" type = "audio/wav">
          Your browser does not support the audio element.
        </audio>
      </body>
    </html>
    """


@app.post("/calibrate")
async def calibrate_microphone(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".wav") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        y, sr = librosa.load(temp_path, sr = 22050)

        valid, feedback = is_audio_acceptable(y, sr)
        if not valid:
            os.remove(temp_path)
            return JSONResponse(status_code = 400, content = {"status": "fail", "message": feedback})

        _, peak_db, peak_time = compute_fft_db(temp_path)
        os.remove(temp_path)

        return{
            "status": "success",
            "peak_db": round(peak_db, 2),
            "message": "Calibration Successful",
            "peak_time": round(peak_time)
        }

    except Exception as e:
        os.remove(temp_path)
        return JSONResponse(status_code = 500, content = {"status": "error", "message": str(e)})





def make_json_safe(obj):
    if isinstance(obj, np.generic):
        return obj.item()

    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return obj

from fastapi import Form

@app.post("/predict-audio/")
async def predict_audio(
        file: UploadFile = File(...),
        user_id: str = Form(...),
        device_id: str = Form(...),
        latitude: float = Form(...),
        longitude: float = Form(...),
        location_desc: str = Form(...)
):

    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".wav") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    print(f"Received file: {temp_path}")

    try:
        y, sr = librosa.load(temp_path, sr = 22050)
        valid, feedback = is_general_audio_clean(y, sr)
        if not valid:
            os.remove(temp_path)
            return JSONResponse(status_code = 400, content = {"status": "fail", "message": feedback})


        result = analyze_and_classify_chunks(temp_path, model, label_encoder)
        print(f"Sending result: {result}")
        os.remove(temp_path)

        if result is None or "peak_db" not in result:
            return JSONResponse(content = {"error": "Invalid or empty audio."}, status_code = 400)

        scan_data = {
            "user_id": user_id,
            "device_id": device_id,
            "timestamp": datetime.utcnow().isoformat(),
            "latitude": latitude,
            "longitude": longitude,
            "location_desc": location_desc,
            "peak_db": result["peak_db"],
            "peak_time": result["peak_time"],
            "peak_class": result["peak_class"],
            "class_durations": result["class_durations"],
            "chunk_results": result["chunk_results"]  # optional, can remove if too heavy
        }

        save_scan_to_firestore(scan_data)
        return JSONResponse(content = make_json_safe(result))

    except Exception as e:
        os.remove(temp_path)
        return JSONResponse(status_code = 500, content = {"error": str(e)})
