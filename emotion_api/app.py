from functions import build_dynamic_model
import os
import time
import uuid
import datetime
import pickle
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
import whisper
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from textblob import TextBlob
from pydub import AudioSegment

# ==========================================
# CONFIGURATION
# ==========================================
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///api_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Resources
MODEL_PATH = 'model_bimodal_6class.h5'
ENCODER_PATH = 'label_encoder_6class.pkl'
SCALER_PATH = 'scaler_6class.pkl'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# AI Constants
SR = 16000
DURATION = 3
EXPECTED_MFCC_LEN = 94
MIN_ASR_DURATION_MS = 3000

# ==========================================
# LOAD AI RESOURCES
# ==========================================
print("Loading AI Resources...")
try:
    input_shapes = {
        'MFCC': (94, 40), 'Chroma': (12,), 'Spectral_Contrast': (7,),
        'Zero_Crossing_Rate': (1,), 'RMS_Energy': (1,), 'Sentiment': (1,)
    }
    input_types = {
        'MFCC': 'sequence', 'Chroma': 'numeric', 'Spectral_Contrast': 'numeric',
        'Zero_Crossing_Rate': 'numeric', 'RMS_Energy': 'numeric', 'Sentiment': 'text'
    }

    model = build_dynamic_model(input_types, input_shapes, num_classes=6, num_sent_classes=3)
    model.load_weights(MODEL_PATH)
    
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    print("Loading Whisper Model...")
    asr_model = whisper.load_model("base") 
    print("✅ Whisper Ready.")

    print("✅ AI System Ready.")

except Exception as e:
    print(f"❌ CRITICAL ERROR loading AI: {e}")

# ==========================================
# 2. DATABASE MODELS
# ==========================================
class Client(db.Model):
    __tablename__ = 'Client'
    ClientID = db.Column(db.String(255), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    ip_address = db.Column(db.String(45), nullable=False)

class RequestLog(db.Model):
    __tablename__ = 'Request'
    RequestID = db.Column(db.String(255), primary_key=True)
    ClientID = db.Column(db.String(255), db.ForeignKey('Client.ClientID'), nullable=False)
    http_method = db.Column(db.String(10), nullable=False)
    endpoint = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.String(255), nullable=True)
    status_code = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    processing_time = db.Column(db.Integer, nullable=False)
    upload_time_ms = db.Column(db.Integer, nullable=True)   
    prep_time_ms = db.Column(db.Integer, nullable=True)
    asr_time_ms = db.Column(db.Integer, nullable=True)      
    inference_time_ms = db.Column(db.Integer, nullable=True)

class Audio(db.Model):
    __tablename__ = 'Audio'
    AudioID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    RequestID = db.Column(db.String(255), db.ForeignKey('Request.RequestID'), unique=True, nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    duration_in_sec = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Result(db.Model):
    __tablename__ = 'Result'
    ResultID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    AudioID = db.Column(db.Integer, db.ForeignKey('Audio.AudioID'), unique=True, nullable=False)
    emotion = db.Column(db.String(255), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    text = db.Column(db.Text, nullable=True)
    sentiment_index = db.Column(db.Integer, nullable=True)
    sentiment_polarity = db.Column(db.Float, nullable=True)      
    audio_only_emotion = db.Column(db.String(255), nullable=True)
    audio_only_confidence = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow) 

with app.app_context():
    db.create_all()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def transcribe_audio(audio_path):
    # r = sr.Recognizer()
    # try:
    #     with sr.AudioFile(audio_path) as source:
    #         audio_data = r.record(source)
    #         text = r.recognize_google(audio_data)
    #         return text
    # except Exception:
    #     return ""
    try:
        # Whisper processes audio directly
        result = asr_model.transcribe(audio_path)
        text = result["text"].strip()
        print(f"Whisper Transcribed: {text}")
        return text
    except Exception as e:
        print(f"Whisper Error: {e}")
        return ""

def prepare_inputs(audio_path, text):
    y, sr = librosa.load(audio_path, sr=SR, duration=DURATION)
    
    # Pad/Truncate
    target_len = SR * DURATION
    if len(y) < target_len: 
        y = np.pad(y, (0, target_len - len(y)))
    else: 
        y = y[:target_len]
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    if mfcc.shape[0] < EXPECTED_MFCC_LEN:
        mfcc = np.pad(mfcc, ((0, EXPECTED_MFCC_LEN - mfcc.shape[0]), (0,0)))
    else: mfcc = mfcc[:EXPECTED_MFCC_LEN, :]
        
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)

    stats_vec = np.hstack([chroma, contrast, zcr, rms]).reshape(1, -1)

    stats_vec = scaler.transform(stats_vec)

    in_chroma = stats_vec[:, 0:12]
    in_contrast = stats_vec[:, 12:19]
    in_zcr = stats_vec[:, 19:20]
    in_rms = stats_vec[:, 20:21]
    
    polarity = 0.0 
    if not text: 
        sent_idx = 1 
    else:
        polarity = TextBlob(text).sentiment.polarity
        sent_idx = 2 if polarity > 0.1 else (0 if polarity < -0.1 else 1)

    inputs = {
        'in_MFCC': np.expand_dims(mfcc, axis=0),
        'in_Chroma': in_chroma,
        'in_Spectral_Contrast': in_contrast,
        'in_RMS_Energy': in_rms,
        'in_Zero_Crossing_Rate': in_zcr,
        'in_Sentiment': np.array([[sent_idx]])
    }
    return inputs, sent_idx, polarity

# ==========================================
# 4. API ENDPOINTS
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    print("--- DEBUG INFO ---")
    t_start = time.time()
    req_id = str(uuid.uuid4())
    client_id = request.headers.get('Client-ID')

    client = Client.query.filter_by(ClientID=client_id).first()
    if not client:
        return jsonify({'error': 'Unauthorized: Invalid or missing Client-ID'}), 401

    try:
        if 'file' not in request.files: raise ValueError("No file provided")
        
        # --- 1. HANDLE FILE UPLOAD & CONVERSION ---
        t_upload_start = time.time()
        
        file = request.files['file']
        temp_filename = f"temp_{req_id}_{file.filename}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(temp_path)
        
        final_wav_filename = f"{req_id}.wav"
        final_wav_path = os.path.join(UPLOAD_FOLDER, final_wav_filename)
        
        try:
            # Convert everything to standard WAV
            raw_audio = AudioSegment.from_file(temp_path)
            raw_temp_path = os.path.join(UPLOAD_FOLDER, f"raw_{req_id}.wav")
            raw_audio.export(raw_temp_path, format="wav")

            # Noise Reduction
            y, sr_lib = librosa.load(raw_temp_path, sr=SR)
            y_clean = nr.reduce_noise(y=y, sr=sr_lib, prop_decrease=0.8)
            clean_temp_path = os.path.join(UPLOAD_FOLDER, f"clean_{req_id}.wav")
            sf.write(clean_temp_path, y_clean, sr_lib)

            # Audio Padding Logic for Short Clips
            audio = AudioSegment.from_file(clean_temp_path)
            original_duration = len(audio) / 1000.0
            if len(audio) < MIN_ASR_DURATION_MS:
                # Calculate how much silence is needed
                silence_needed = MIN_ASR_DURATION_MS - len(audio)
                # Create a silence segment (ms)
                silence = AudioSegment.silent(duration=silence_needed)
                # Append silence to the end of the original audio
                audio = audio + silence
                print(f"Audio padded with {silence_needed}ms of silence.")

            # Export as standard WAV (16kHz, Mono)
            audio = audio.set_frame_rate(SR).set_channels(1)
            audio.export(final_wav_path, format="wav")
            
            # Clean temp
            if os.path.exists(temp_path): os.remove(temp_path)
                
        except Exception as e:
            if os.path.exists(temp_path): os.remove(temp_path)
            return jsonify({'error': f"Audio Conversion Error: {str(e)}"}), 500

        # FIX: Define file_path here so it is available later
        file_path = final_wav_path 

        t_upload_end = time.time()
    
        # --- 2. TRANSCRIBE ---
        t_asr_start = time.time()
        transcribed_text = transcribe_audio(file_path) # Now file_path exists!
        t_asr_end = time.time()

        # --- 3. PREPARE INPUTS ---
        t_prep_start = time.time()
        inputs, sent_idx, polarity = prepare_inputs(file_path, transcribed_text)
        t_prep_end = time.time()

        # --- 4. PREDICT ---
        t_infer_start = time.time()
        preds = model.predict(inputs, verbose=0)
        pred_idx = np.argmax(preds)
        emotion_label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(np.max(preds))
        t_infer_end = time.time()

        # --- 5. AUDIO-ONLY SIMULATION ---
        inputs_audio_only = inputs.copy()
        inputs_audio_only['in_Sentiment'] = np.array([[1]]) 
        preds_audio = model.predict(inputs_audio_only, verbose=0)
        audio_emotion = label_encoder.inverse_transform([np.argmax(preds_audio)])[0]
        audio_confidence = float(np.max(preds_audio))

        # --- 6. LOGGING ---
        upload_ms = int((t_upload_end - t_upload_start) * 1000)
        asr_ms = int((t_asr_end - t_asr_start) * 1000)
        prep_ms = int((t_prep_end - t_prep_start) * 1000)
        inference_ms = int((t_infer_end - t_infer_start) * 1000)
        total_latency_ms = int((time.time() - t_start) * 1000)

        new_req = RequestLog(
            RequestID=req_id, ClientID=client_id, http_method=request.method,
            endpoint=request.path, user_id=request.form.get('user_id'),
            status_code=200, processing_time=total_latency_ms,
            upload_time_ms=upload_ms, asr_time_ms=asr_ms,
            prep_time_ms=prep_ms, inference_time_ms=inference_ms
        )
        db.session.add(new_req)
        db.session.flush()
        
        new_audio = Audio(RequestID=req_id, file_path=file_path, duration_in_sec=original_duration)
        db.session.add(new_audio)
        db.session.flush()
        
        new_result = Result(
            AudioID=new_audio.AudioID, emotion=emotion_label, text=transcribed_text,
            confidence_score=confidence, sentiment_index=int(sent_idx), sentiment_polarity=float(round(polarity, 4)),
            audio_only_emotion=audio_emotion, audio_only_confidence=audio_confidence
        )
        db.session.add(new_result)
        db.session.commit()
        
        return jsonify({
            'request_id': req_id,
            'emotion': emotion_label.upper(),
            'confidence': f"{confidence:.2%}",
            'text_detected': transcribed_text,
            'analysis': {
                'audio_only_emotion': audio_emotion.upper(),
                'audio_only_confidence': f"{audio_confidence:.2%}",
                'sentiment_index': int(sent_idx),
                'sentiment_polarity': float(round(polarity, 3)),
                'audio_duration': float(round(original_duration, 2))
            },
            'performance': {
                'upload_ms': upload_ms, 'asr_ms': asr_ms, 
                'inference_ms': inference_ms, 'total_latency_sec': total_latency_ms
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.cli.command('seed-db')
def seed_db():
    db.create_all()
    if not Client.query.filter_by(ClientID='test_app_01').first():
        db.session.add(Client(ClientID='test_app_01', name='Test App', ip_address='127.0.0.1'))
        db.session.commit()
        print("✅ Added Test Client.")

if __name__ == '__main__':
    app.run(debug=True, port=5000)