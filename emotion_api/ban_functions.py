import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tf_keras
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from tf_keras import layers, models, optimizers, callbacks, regularizers

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # Data Settings
    DATA_PATH = 'dataset_meld/'
    SR = 16000
    MAX_DURATION = 3
    MAX_SAMPLES = SR * MAX_DURATION
    
    # Feature Settings
    N_MFCC = 40
    MAX_MFCC_LEN = 94 
    
    # Training Parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    EPOCHS = 50
    DROPOUT = 0.4
    
    # Target Count for Balancing (Cap & Floor)
    TARGET_COUNT = 1700 
    
    # Masking Rate (Probability to replace text with Index 0)
    SENTIMENT_MASK_RATE = 0.60 
    
    # Default Classes
    DEFAULT_CLASSES = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'disgust']

# ==========================================
# 2. AUGMENTATION ENGINE
# ==========================================
def augment_audio_custom(y, sr, aug_type='random'):
    """
    Applies scientifically accurate augmentations based on Arousal/Valence theory.
    """
    try:
        # PITCH UP (Fear, Anger, Joy)
        if aug_type == 'pitch_shift_up':
            step = np.random.uniform(1.5, 2.5) # Shift up 1.5-2.5 semitones
            return librosa.effects.pitch_shift(y, sr=sr, n_steps=step)
        
        # PITCH DOWN (Sadness, Disgust)
        elif aug_type == 'pitch_shift_down':
            step = np.random.uniform(-2.5, -1.5) # Shift down 1.5-2.5 semitones
            return librosa.effects.pitch_shift(y, sr=sr, n_steps=step)
            
        # FAST TEMPO (Fear, Anger, Joy)
        elif aug_type == 'time_stretch_fast':
            rate = np.random.uniform(1.1, 1.25) # 10-25% faster
            return librosa.effects.time_stretch(y, rate=rate)
        
        # SLOW TEMPO (Sadness, Disgust)
        elif aug_type == 'time_stretch_slow':
            rate = np.random.uniform(0.75, 0.9) # 10-25% slower
            return librosa.effects.time_stretch(y, rate=rate)
            
        # NOISE (Robustness for all)
        elif aug_type == 'noise':
            noise_amp = 0.005 * np.random.uniform() * np.amax(y)
            return y + noise_amp * np.random.normal(size=y.shape[0])
            
        # RANDOM (Fallback)
        elif aug_type == 'random':
            choice = np.random.randint(0, 3)
            if choice == 0: # Time
                rate = np.random.uniform(0.8, 1.2)
                return librosa.effects.time_stretch(y, rate=rate)
            elif choice == 1: # Pitch
                steps = np.random.randint(-2, 3)
                return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            elif choice == 2: # Noise
                noise_amp = 0.005 * np.random.uniform() * np.amax(y)
                return y + noise_amp * np.random.normal(size=y.shape[0])
                
        return y
    except Exception:
        return y

# ==========================================
# 3. FEATURE EXTRACTION & BALANCING
# ==========================================
def extract_features_balanced(input_data, target_emotions, augment=False, target_count=1700):
    """
    Extracts features with SMART BALANCING:
    1. Downsamples if count > target_count (e.g. Neutral).
    2. Expands (Augments) if count < target_count.
    """
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            print(f"❌ Error: {input_data} not found.")
            return np.array([]), np.array([]), np.array([]), np.array([])
        df = pd.read_csv(input_data)
        df = df[df['Emotion'].str.lower().isin(target_emotions)]
    else:
        df = input_data

    X_mfcc, X_stats, X_sent, y_labels = [], [], [], []
    
    # AUGMENTATION STRATEGY MAP (Per Class)
    STRATEGY = {
        'neutral': [], 
        'joy':     ['time_stretch_fast', 'pitch_shift_up'],
        'sadness': ['time_stretch_slow', 'pitch_shift_down'],
        'anger':   ['noise', 'pitch_shift_up', 'time_stretch_fast'],
        'fear':    ['pitch_shift_up', 'time_stretch_fast', 'noise'], 
        'disgust': ['time_stretch_slow', 'pitch_shift_down', 'noise'] 
    }

    for emotion in target_emotions:
        sub_df = df[df['Emotion'].str.lower() == emotion]
        current_count = len(sub_df)
        if current_count == 0: continue
        
        # --- A. DOWNSAMPLING LOGIC ---
        if augment and current_count > target_count:
            # Randomly sample exactly target_count rows
            sub_df = sub_df.sample(n=target_count, random_state=42)
            print(f" -> Class '{emotion}': {current_count} original. Downsampling to {target_count}.")
            multiplier = 1
            
        # --- B. EXPANSION LOGIC ---
        elif augment and current_count < target_count:
            multiplier = int(np.ceil(target_count / current_count))
            print(f" -> Class '{emotion}': {current_count} original. Expanding {multiplier}x to reach ~{target_count}.")
            
        # --- C. NO AUGMENTATION ---
        else:
            multiplier = 1
            # print(f" -> Class '{emotion}': {current_count} (Original)")
        
        aug_methods = STRATEGY.get(emotion, ['random'])
        
        for _, row in tqdm(sub_df.iterrows(), total=len(sub_df), desc=f"Processing {emotion}"):
            try:
                # Load Audio ONCE
                y_orig, _ = librosa.load(row['Filepath'], sr=Config.SR)
                if len(y_orig) < 500: continue
                
                # Generate Copies
                for i in range(multiplier):
                    y = y_orig.copy()
                    
                    # Apply augmentation to copies (keep 1st copy clean if desired)
                    if augment and i > 0: 
                        method = aug_methods[i % len(aug_methods)]
                        y = augment_audio_custom(y, Config.SR, aug_type=method)
                    
                    # --- Feature Extraction ---
                    if len(y) > Config.MAX_SAMPLES: y = y[:Config.MAX_SAMPLES]
                    else: y = np.pad(y, (0, int(Config.MAX_SAMPLES - len(y))), 'constant')

                    mfcc = librosa.feature.mfcc(y=y, sr=Config.SR, n_mfcc=Config.N_MFCC).T
                    if mfcc.shape[0] > Config.MAX_MFCC_LEN: mfcc = mfcc[:Config.MAX_MFCC_LEN, :]
                    else: mfcc = np.pad(mfcc, ((0, Config.MAX_MFCC_LEN - mfcc.shape[0]), (0, 0)), 'constant')
                    
                    try:
                        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=Config.SR).T, axis=0)
                        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=Config.SR).T, axis=0)
                        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
                        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
                        stats = np.hstack([chroma, contrast, zcr, rms])
                    except: continue
                    
                    X_mfcc.append(mfcc)
                    X_stats.append(stats)
                    X_sent.append(row['Sentiment'])
                    y_labels.append(emotion)
                    
            except Exception: continue
            
    return np.array(X_mfcc), np.array(X_stats), np.array(X_sent), np.array(y_labels)

# ==========================================
# 4. DATASET PREPARATION WRAPPER
# ==========================================
def prepare_datasets(target_classes, augment=True):
    """
    Main function to load Train/Dev/Test.
    Applies BALANCING (Downsampling + Expansion) to Training Data ONLY.
    """
    print(f"\n--- Preparing Datasets (Classes: {target_classes}) ---")
    data = {}
    
    # 1. Train Set (With Balancing)
    print("\n--- Processing Training Set ---")
    path = f"{Config.DATA_PATH}prepared_train_sent_emo.csv"
    m, s, txt, y = extract_features_balanced(path, target_classes, augment=augment, target_count=Config.TARGET_COUNT)
    data['X_m_train'], data['X_s_train'], data['sent_train'], data['y_train'] = m, s, txt, y
    
    # 2. Dev/Test (No Augmentation)
    for split in ['dev', 'test']:
        print(f"\n--- Processing {split} Set (Original Only) ---")
        path = f"{Config.DATA_PATH}prepared_{split}_sent_emo.csv"
        m, s, txt, y = extract_features_balanced(path, target_classes, augment=False)
        data[f'X_m_{split}'], data[f'X_s_{split}'], data[f'sent_{split}'], data[f'y_{split}'] = m, s, txt, y

    # 3. Label Encoding
    le_target = LabelEncoder().fit(target_classes) 
    for split in ['train', 'dev', 'test']:
        if f'y_{split}' in data and len(data[f'y_{split}']) > 0:
            data[f'y_{split}'] = le_target.transform(data[f'y_{split}'])
    data['le_target'] = le_target

    # 4. Sentiment Encoding & Masking
    if 'sent_train' in data and len(data['sent_train']) > 0:
        le_sent = LabelEncoder().fit(data['sent_train'])
        
        def encode(text, mask=False):
            # Handle unknown labels
            safe = [t if t in le_sent.classes_ else le_sent.classes_[0] for t in text]
            enc = le_sent.transform(safe) + 1 # Shift +1 for Mask Index 0
            if mask:
                m = np.random.rand(len(enc)) < Config.SENTIMENT_MASK_RATE
                enc[m] = 0
            return enc

        data['sent_train_enc'] = encode(data['sent_train'], mask=True)
        data['sent_dev_enc'] = encode(data['sent_dev'], mask=False)
        data['sent_test_enc'] = encode(data['sent_test'], mask=False)
        data['num_sent_classes'] = len(le_sent.classes_) + 1
        data['le_sent'] = le_sent

    # 5. Scaler
    if 'X_s_train' in data and len(data['X_s_train']) > 0:
        scaler = StandardScaler().fit(data['X_s_train'])
        for split in ['train', 'dev', 'test']:
            if f'X_s_{split}' in data:
                data[f'X_s_{split}'] = scaler.transform(data[f'X_s_{split}'])
        data['scaler'] = scaler
        
    return data

# ==========================================
# 5. HELPER: MODEL INPUT PREP
# ==========================================
def prep_inputs(m, s, txt):
    return [m, s, txt]

# ==========================================
# 6. ATTENTION MODEL ARCHITECTURE
# ==========================================
def build_attention_model(input_shapes, num_classes, num_sent_classes):
    """
    CNN-LSTM + Sentiment Embedding + Multi-Head Attention Fusion
    """
    # --- Inputs ---
    in_mfcc = layers.Input(shape=input_shapes['MFCC'], name="MFCC_Input")
    
    # Calculate stats dim
    stats_dim = 21 # Default
    if 'Chroma' in input_shapes:
         stats_dim = input_shapes['Chroma'][0] + input_shapes['Spectral_Contrast'][0] + input_shapes['Zero_Crossing_Rate'][0] + input_shapes['RMS_Energy'][0]
    in_stats = layers.Input(shape=(stats_dim,), name="Stats_Input")
    
    in_sent = layers.Input(shape=(1,), name="Sentiment_Input") 

    # --- Audio Branch ---
    x_aud = layers.Conv1D(64, 3, padding='same', activation='relu')(in_mfcc)
    x_aud = layers.BatchNormalization()(x_aud)
    x_aud = layers.MaxPooling1D(2)(x_aud)
    x_aud = layers.Dropout(0.2)(x_aud)
    
    x_aud = layers.Conv1D(128, 3, padding='same', activation='relu')(x_aud)
    x_aud = layers.BatchNormalization()(x_aud)
    x_aud = layers.MaxPooling1D(2)(x_aud)
    
    x_aud = layers.LSTM(128, return_sequences=True)(x_aud)
    x_aud = layers.LSTM(64, return_sequences=False)(x_aud)
    
    # Merge Audio + Stats
    x_stats = layers.Dense(32, activation='relu')(in_stats)
    x_audio_final = layers.Concatenate()([x_aud, x_stats])
    
    # Project to 64
    x_audio_proj = layers.Dense(64, activation='tanh', name="Audio_Projection")(x_audio_final)

    # --- Text Branch ---
    x_text = layers.Embedding(input_dim=num_sent_classes, output_dim=64, mask_zero=True)(in_sent)
    x_text = layers.Flatten()(x_text)
    x_text_proj = layers.Dense(64, activation='tanh', name="Text_Projection")(x_text)

    # --- Attention Fusion ---
    # Reshape for Attention: (Batch, 1, 64)
    x_aud_exp = layers.Reshape((1, 64), name="Reshape_Audio")(x_audio_proj)
    x_txt_exp = layers.Reshape((1, 64), name="Reshape_Text")(x_text_proj)
    
    # Concat: (Batch, 2, 64)
    combined_seq = layers.Concatenate(axis=1)([x_aud_exp, x_txt_exp]) 

    # Self-Attention
    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=64)(combined_seq, combined_seq)
    
    # Add & Norm
    x_fused = layers.Add()([combined_seq, attn_output])
    x_fused = layers.LayerNormalization()(x_fused)
    x_fused = layers.Flatten()(x_fused)

    # --- Head ---
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_fused)
    x = layers.Dropout(Config.DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Compile
    model = models.Model(inputs=[in_mfcc, in_stats, in_sent], outputs=outputs, name="Bimodal_Attention_Network")
    opt = optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# ==========================================
# 7. EVALUATION & VISUALIZATION TOOLS
# ==========================================
def run_evaluation(model, inputs_test, y_test, history, class_names):
    # 1. Plot Curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['loss'], label='Train'); ax[0].plot(history.history['val_loss'], label='Val')
    ax[0].set_title('Loss'); ax[0].legend()
    ax[1].plot(history.history['accuracy'], label='Train'); ax[1].plot(history.history['val_accuracy'], label='Val')
    ax[1].set_title('Accuracy'); ax[1].legend()
    plt.show()

    # 2. Report & Matrix
    y_pred = np.argmax(model.predict(inputs_test, verbose=0), axis=1)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.show()

def plot_3d_tsne(features, labels, class_names):
    print("Generating 3D t-SNE...")
    tsne = TSNE(n_components=3, perplexity=30, init='pca', random_state=42)
    proj = tsne.fit_transform(features)
    
    df = pd.DataFrame(proj, columns=['x', 'y', 'z'])
    df['label'] = [class_names[y] for y in labels]
    
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', opacity=0.7, size_max=5)
    fig.show()