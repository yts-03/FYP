emotion_api/
├── app.py                  # Main Flask Application (API Endpoints & Music Mapping Logic)
├── functions.py            # Core AI Logic (Data Loading, Feature Extraction, Model Building)
├── experiment.ipynb        # Jupyter Notebook for Training, Evaluation & Visualisation
├── model_bimodal_6class.h5 # Pre-trained Keras Model Weights
├── scaler_6class.pkl       # Saved StandardScaler for feature normalisation
├── label_encoder_6class.pkl # Saved LabelEncoder for emotion classes
├── instance/
│   └── api_database.db     # SQLite Database
├── uploads/                # Temporary storage for processing audio files
└── templates/
    └── index.html          # Simple Frontend for testing the API


-----------------------
Installation & Setup
-----------------------
1. Prerequisites
- Python 3.8+
- FFmpeg

2. Install Dependencies
- requirement.txt

3. Database Initialisation (Optional)
>> flask --app app seed-db

4. Start the Server
>> python app.py

5. Open new terminal (need to install ngrok)
>> ngrok http 5000

6. Access website
>> https://untenebrous-arvilla-unbureaucratic.ngrok-free.dev

