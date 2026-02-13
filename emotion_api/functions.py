import os
import gc
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tf_keras
from tf_keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.manifold import TSNE
import plotly.express as px
from tqdm import tqdm

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
    MAX_MFCC_LEN = 94  # (16000*3)/512 hop_length ~= 94
    
    # Training Parameters Settings
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 40
    DROPOUT = 0.3
    
    # Default Classes
    DEFAULT_CLASSES = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']

def augment_audio(y, sr):
    """
    Applies augmentations: Time Stretch, Pitch Shift, or Noise Injection.
    """
    choice = np.random.randint(0, 3)
    
    # A. Time Stretch (Speed up/slow down)
    if choice == 0:
        rate = np.random.uniform(0.8, 1.2)
        y = librosa.effects.time_stretch(y, rate=rate)
        
    # B. Pitch Shift (Change tone)
    elif choice == 1:
        steps = np.random.randint(-2, 3) # -2 to +2 semitones
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        
    # C. Noise Injection (Add static)
    elif choice == 2:
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
        
    return y

# ==========================================
# 2. DATA PREPROCESSING & FEATURE EXTRACTION
# ==========================================
def extract_features(csv_path, target_emotions, clean_noise=False, trim_silence=False, augment=False):
    """
    Core engine to load audio and extract ALL features.
    Includes safeguards for short audio files.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV File not found {csv_path}")
        return None, None, None, None

    df = pd.read_csv(csv_path)
    df = df[df['Emotion'].str.lower().isin(target_emotions)]
    
    print(f"Processing {len(df)} samples from {csv_path} (Augment={augment})...")
    
    X_mfcc, X_stats, X_sent, y_labels = [], [], [], []
    
    success_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # --- Path Logic ---
            file_path = row['Filepath']

            # 1. Load Audio
            # We use a try block here in case the file is corrupted
            try:
                y, _ = librosa.load(file_path, sr=Config.SR)
            except Exception:
                continue 

            # Safeguard: If audio is too short (< 0.1s), skip it to prevent crashes
            if len(y) < 500: 
                continue

            # 2. Preprocessing
            if clean_noise:
                try:
                    # Noise reduce can fail on very short clips
                    if len(y) > Config.SR * 0.1: 
                        y = nr.reduce_noise(y=y, sr=Config.SR, prop_decrease=0.8)
                except Exception:
                    pass # Keep original y if NR fails

            if trim_silence:
                try:
                    y, _ = librosa.effects.trim(y, top_db=20)
                except Exception:
                    pass

            # 3. Augmentation (Train Only)
            if augment:
                try:
                    y = augment_audio(y, Config.SR)
                except Exception:
                    pass 
            
            # Check length again after processing
            if len(y) < 500:
                continue

            # 4. Pad/Crop
            if len(y) > Config.MAX_SAMPLES:
                y = y[:Config.MAX_SAMPLES]
            else:
                y = np.pad(y, (0, int(Config.MAX_SAMPLES - len(y))), 'constant')

            # 5. Feature Extraction
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=Config.SR, n_mfcc=Config.N_MFCC).T
            if mfcc.shape[0] > Config.MAX_MFCC_LEN:
                mfcc = mfcc[:Config.MAX_MFCC_LEN, :]
            else:
                mfcc = np.pad(mfcc, ((0, Config.MAX_MFCC_LEN - mfcc.shape[0]), (0, 0)), 'constant')
            
            # Stats (Wrap in try/except because spectral contrast hates silence)
            try:
                chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=Config.SR).T, axis=0)
                contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=Config.SR).T, axis=0)
                zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
                rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
                stats = np.hstack([chroma, contrast, zcr, rms])
            except Exception:
                # If statistical extraction fails, skip this sample
                continue
            
            sent = row['Sentiment']

            X_mfcc.append(mfcc)
            X_stats.append(stats)
            X_sent.append(sent)
            y_labels.append(row['Emotion'].lower())
            
            success_count += 1
            
        except Exception as e:
            continue
            
    if success_count == 0:
        print("❌ CRITICAL: No files processed. Check paths and libraries.")
        
    return np.array(X_mfcc), np.array(X_stats), np.array(X_sent), np.array(y_labels)

# ==========================================
# 3. (2) + DATA ENCODING & SCALING
# ==========================================
def prepare_datasets(target_classes, clean_noise=False, trim_silence=False, augment=False):
    """
    Wrapper to load Train/Dev/Test sets, encode labels, and scale features.
    Returns dictionary containing all prepared data.
    """
    print(f"\n--- Preparing Data for Classes: {target_classes} ---")
    data = {}
    
    # 1. Extract
    for split in ['train', 'dev', 'test']:
        path = f"{Config.DATA_PATH}prepared_{split}_sent_emo.csv"
        m, s, txt, y = extract_features(path, target_classes, clean_noise=clean_noise, trim_silence=trim_silence, augment=augment)
        data[f'X_m_{split}'] = m
        data[f'X_s_{split}'] = s
        data[f'sent_{split}'] = txt
        data[f'y_{split}'] = y

    # 2. Encoding & Scaling
    # Labels
    le_target = LabelEncoder().fit(target_classes)
    for split in ['train', 'dev', 'test']:
        data[f'y_{split}'] = le_target.transform(data[f'y_{split}'])
    data['le_target'] = le_target # Save encoder for later decoding

    # Sentiment (Fit on Train, Transform All)
    le_sent = LabelEncoder().fit(data['sent_train'])
    for split in ['train', 'dev', 'test']:
        data[f'sent_{split}'] = le_sent.transform(data[f'sent_{split}'])
    data['num_sent_classes'] = len(le_sent.classes_)

    # Stats Scaling (Fit on Train, Transform All)
    scaler = StandardScaler().fit(data['X_s_train'])
    for split in ['train', 'dev', 'test']:
        data[f'X_s_{split}'] = scaler.transform(data[f'X_s_{split}'])
        
    return data, scaler

# ==========================================
# 4. DYNAMIC FEATURE SLICING (ABLATION STUDY: FEATURE IMPORTANCE)
# ==========================================
def slice_features(data, active_features):
    """
    Dynamically creates input lists for the model based on requested features.
    active_features: list of names ['MFCC', 'Chroma', 'Sentiment', etc.]
    """
    # Map feature names to data array slices
    # X_s indices: 0-11(Chroma), 12-18(Contrast), 19(ZCR), 20(RMS)
    feature_map = {
        'MFCC': ('sequence', lambda d, s: d[f'X_m_{s}']),
        'Chroma': ('numeric', lambda d, s: d[f'X_s_{s}'][:, 0:12]),
        'Spectral_Contrast': ('numeric', lambda d, s: d[f'X_s_{s}'][:, 12:19]),
        'Zero_Crossing_Rate': ('numeric', lambda d, s: d[f'X_s_{s}'][:, 19:20]),
        'RMS_Energy': ('numeric', lambda d, s: d[f'X_s_{s}'][:, 20:21]),
        'Sentiment': ('text', lambda d, s: d[f'sent_{s}'])
    }
    
    inputs = {'train': [], 'dev': [], 'test': []}
    shapes = {}
    input_types = {}

    for name in active_features:
        type_tag, accessor = feature_map[name]
        input_types[name] = type_tag
        
        for split in ['train', 'dev', 'test']:
            arr = accessor(data, split)
            inputs[split].append(arr)
            if split == 'train':
                # Store shape for model builder (exclude batch dim)
                shapes[name] = arr.shape[1:] 

    return inputs, shapes, input_types

# ==========================================
# 5. DYNAMIC MODEL BUILDER
# ==========================================
def build_dynamic_model(active_inputs_config, input_shapes, num_classes, num_sent_classes):
    """
    Constructs a Keras model that only includes the specified branches.
    """
    inputs = []
    merges = []
    
    for name, f_type in active_inputs_config.items():
        if f_type == 'sequence': # MFCC
            in_layer = layers.Input(shape=input_shapes[name], name=f"in_{name}")
            inputs.append(in_layer)
            x = layers.Conv1D(64, 3, activation='relu', padding='same')(in_layer)
            x = layers.MaxPooling1D(2)(x)
            x = layers.LSTM(64)(x)
            merges.append(x)
            
        elif f_type == 'numeric': # Stats
            in_layer = layers.Input(shape=input_shapes[name], name=f"in_{name}")
            inputs.append(in_layer)
            x = layers.Dense(32, activation='relu')(in_layer)
            x = layers.BatchNormalization()(x)
            merges.append(x)
            
        elif f_type == 'text': # Sentiment
            in_layer = layers.Input(shape=(1,), name=f"in_{name}")
            inputs.append(in_layer)
            x = layers.Embedding(input_dim=num_sent_classes, output_dim=16)(in_layer)
            x = layers.Flatten()(x)
            merges.append(x)

    # Fusion
    if len(merges) > 1:
        x = layers.Concatenate()(merges)
    else:
        x = merges[0] # Single feature case
        
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(Config.DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    opt = optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 6. EVALUATION (TRAINING CURVES + CLASSIFICATION REPORT + CONFUSION MATRIX + TOP 5 CONFUSED PAIRS)
# ==========================================
def run_evaluation(model, X_test, y_test, history, class_names, title="Experiment"):
    print(f"\n=== Results: {title} ===")
    
    # 1. Training Curves (Loss & Accuracy)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss Plot
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Val Loss')
    ax[0].set_title('Loss Curve')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    # Accuracy Plot
    ax[1].plot(history.history['accuracy'], label='Train Acc')
    ax[1].plot(history.history['val_accuracy'], label='Val Acc')
    ax[1].set_title('Accuracy Curve')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # 2. Classification Report
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 4. Top 5 Confused Pairs
    # Create a copy and zero out diagonal to find errors only
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    
    # Sort indices by count (descending)
    sorted_indices = np.argsort(cm_no_diag, axis=None)[::-1]
    
    print("\n--- Top 5 Confused Pairs ---")
    count = 0
    for idx in sorted_indices:
        true_idx, pred_idx = np.unravel_index(idx, cm.shape)
        error_count = cm[true_idx, pred_idx]
        
        # Stop if count is 0 (no more errors) or we reached 5 pairs
        if error_count == 0 or count >= 5:
            break
            
        true_label = class_names[true_idx]
        pred_label = class_names[pred_idx]
        print(f"{count+1}. True: {true_label} -> Pred: {pred_label} (Count: {error_count})")
        count += 1
    
    return history.history['val_accuracy'][-1]

# ==========================================
# 7. EXPERIMENT RUNNERS
# ==========================================
def run_experiment_class_optimization():
    """
    Experiment 3: Test different combinations of classes (e.g. Drop Surprise).
    """
    scenarios = [
        ("7_Classes_with_Neutral", ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']),
        ("Base_6_Classes", ['joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']),
        ("Drop_Surprise",  ['joy', 'sadness', 'anger', 'fear', 'disgust']),
        ("Drop_Fear",      ['joy', 'sadness', 'anger', 'surprise', 'disgust'])
    ]
    
    for name, classes in scenarios:
        tf.keras.backend.clear_session()
        gc.collect()
        
        # 1. Load Data for this specific class set
        data = prepare_datasets(classes)
        
        # 2. Use All Features
        all_feats = ['MFCC', 'Chroma', 'Spectral_Contrast', 'Zero_Crossing_Rate', 'RMS_Energy', 'Sentiment']
        inputs, shapes, types = slice_features(data, all_feats)
        
        # 3. Train
        model = build_dynamic_model(types, shapes, len(classes), data['num_sent_classes'])
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(data['y_train']), y=data['y_train'])
        
        hist = model.fit(
            inputs['train'], data['y_train'],
            validation_data=(inputs['dev'], data['y_dev']),
            epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, verbose=1,
            class_weight=dict(enumerate(cw)),
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        
        run_evaluation(model, inputs['test'], data['y_test'], hist, data['le_target'].classes_, title=name)

def run_experiment_feature_ablation(mode='single'):
    """
    Experiment 6 & 9: Feature Importance.
    mode='single': Train on ONE feature at a time.
    mode='loo': Leave ONE feature OUT at a time.
    """
    # 1. Load Data once (using default classes)
    data = prepare_datasets(Config.DEFAULT_CLASSES, augment=False)
    full_feats = ['MFCC', 'Chroma', 'Spectral_Contrast', 'Zero_Crossing_Rate', 'RMS_Energy', 'Sentiment']
    
    results = {}
    
    if mode == 'single':
        # Add Fusion for comparison
        scenarios = full_feats + ['Fusion_All']
    else:
        # Leave-One-Out Scenarios
        scenarios = [None] + full_feats # None means "Baseline All"

    for scenario in scenarios:
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Determine active features
        if mode == 'single':
            if scenario == 'Fusion_All': active = full_feats
            else: active = [scenario]
            exp_name = f"Train_{scenario}"
        else: # LOO mode
            if scenario is None: active = full_feats
            else: active = [f for f in full_feats if f != scenario]
            exp_name = f"Exclude_{scenario}" if scenario else "Baseline_All"

        print(f"\nRunning: {exp_name} (Active: {active})")
        
        # Slice Data
        inputs, shapes, types = slice_features(data, active)
        
        # Build & Train
        model = build_dynamic_model(types, shapes, len(Config.DEFAULT_CLASSES), data['num_sent_classes'])
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(data['y_train']), y=data['y_train'])
        
        hist = model.fit(
            inputs['train'], data['y_train'],
            validation_data=(inputs['dev'], data['y_dev']),
            epochs=40, batch_size=Config.BATCH_SIZE, verbose=0, # Silent training
            class_weight=dict(enumerate(cw)),
            callbacks=[callbacks.EarlyStopping(patience=6, restore_best_weights=True)]
        )
        
        val_acc = hist.history['val_accuracy'][-1]
        results[exp_name] = val_acc
        print(f" -> Accuracy: {val_acc:.4f}")

    # Plotting Results
    print("\n=== Feature Experiment Results ===")
    plt.figure(figsize=(12, 7)) # Slightly larger for labels
    
    # Convert to DataFrame for easier sorting/plotting
    df_res = pd.DataFrame(list(results.items()), columns=['Scenario', 'Accuracy'])
    df_res = df_res.sort_values('Accuracy', ascending=True) # Sort for better visual
    
    # Create Bar Chart
    bars = plt.barh(df_res['Scenario'], df_res['Accuracy'], color='#1f77b4')
    
    # --- 1. ADD BASELINE LINE ---
    baseline_key = 'Train_Fusion_All' if mode == 'single' else 'Baseline_All'
    
    # Check if baseline exists in results (it should)
    if baseline_key in results:
        baseline_val = results[baseline_key]
        plt.axvline(x=baseline_val, color='r', linestyle='--', linewidth=2, label=f'Baseline ({baseline_val:.2%})')
        plt.legend()

    # --- 2. ADD VALUE LABELS ---
    for bar in bars:
        width = bar.get_width()
        # Place text slightly to the right of the bar end
        plt.text(width + 0.01,         # X-position
                 bar.get_y() + bar.get_height()/2, # Y-position (center of bar)
                 f'{width:.2%}',       # Text (Format as percentage)
                 va='center',          # Vertical align
                 fontweight='bold',
                 fontsize=10)

    plt.xlabel('Validation Accuracy')
    plt.title(f'Feature Importance ({mode.upper()})')
    plt.xlim(0, max(df_res['Accuracy']) * 1.15) # Add space on right for labels
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ==========================================
# 8. VISUALIZATION
# ==========================================
import plotly.express as px
from sklearn.manifold import TSNE

# ==========================================
# NEW: ADVANCED VISUALIZATION TOOLS
# ==========================================
def plot_3d_tsne(features, labels, class_names, title="3D t-SNE Visualization"):
    """
    Generates an interactive 3D plot of the feature embeddings.
    features: The extracted feature vectors (e.g., flattened MFCCs or Fusion output).
    labels: Integer class labels.
    """
    print(f"Generating {title}...")
    
    # 1. Flatten if necessary (e.g., if input is (N, Time, Feats))
    if len(features.shape) > 2:
        n_samples = features.shape[0]
        features_flat = features.reshape(n_samples, -1)
    else:
        features_flat = features

    # 2. Run t-SNE
    # Perplexity=30 is standard. Early exaggeration helps create gaps between clusters.
    tsne = TSNE(n_components=3, perplexity=30, early_exaggeration=12, 
                metric='cosine', init='pca', learning_rate='auto', random_state=42)
    
    projections = tsne.fit_transform(features_flat)
    
    # 3. Prepare DataFrame for Plotly
    df_viz = pd.DataFrame(projections, columns=['x', 'y', 'z'])
    df_viz['label'] = [class_names[y] for y in labels]
    
    # 4. Plot
    fig = px.scatter_3d(
        df_viz, x='x', y='y', z='z',
        color='label',
        title=title,
        opacity=0.7,
        size_max=5,
        width=900, height=600
    )
    fig.show()

def plot_feature_importance(results_dict, title="Feature Importance Analysis"):
    """
    Plots a horizontal bar chart comparing accuracies from the ablation study.
    results_dict: Dictionary {'Feature Name': Accuracy_Float}
    """
    # Sort results by accuracy
    sorted_items = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    
    # Define colors based on Domain (Optional logic for better reports)
    colors = []
    for n in names:
        if 'MFCC' in n or 'Chroma' in n: colors.append('#1f77b4')   # Blue (Freq)
        elif 'RMS' in n or 'Zero' in n: colors.append('#ff7f0e')    # Orange (Time)
        elif 'Sentiment' in n: colors.append('#2ca02c')             # Green (Text)
        else: colors.append('#d62728')                              # Red (Fusion/Baseline)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, values, color=colors)
    plt.axvline(x=max(values), color='black', linestyle='--', alpha=0.3) # Line at max accuracy
    
    plt.xlabel('Validation Accuracy')
    plt.title(title)
    plt.xlim(0, max(values) * 1.15) # Add space for labels
    plt.gca().invert_yaxis() # Highest accuracy on top
    
    # Add text labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2%}', va='center', fontweight='bold')
                 
    plt.tight_layout()
    plt.show()

def plot_class_distribution(y_data, class_names, set_name="Dataset"):
    """
    Visualizes the number of samples per class to check for imbalance.
    """
    counts = pd.Series(y_data).value_counts().sort_index()
    labels = [class_names[i] for i in counts.index]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=counts.values, palette="viridis")
    plt.title(f"Class Distribution: {set_name}")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    
    for i, v in enumerate(counts.values):
        plt.text(i, v + 5, str(v), ha='center')
        
    plt.show()

def plot_feature_correlation(X_mfcc, X_stats, X_sent, title="Multimodal Feature Correlation"):
    """
    Plots a correlation heatmap between all feature types.
    Note: Collapses MFCC time-series to Global Mean for visualization.
    """
    print(f"Generating {title}...")
    
    # 1. Aggregating MFCCs (Time Series -> Global Mean)
    # X_mfcc shape: (N, 94, 40) -> (N, 40)
    if len(X_mfcc.shape) == 3:
        mfcc_mean = np.mean(X_mfcc, axis=1)
    else:
        mfcc_mean = X_mfcc

    # 2. Reshaping Sentiment
    # (N,) -> (N, 1)
    if len(X_sent.shape) == 1:
        sent_vec = X_sent.reshape(-1, 1)
    else:
        sent_vec = X_sent

    # 3. Create DataFrame
    # Feature Names
    mfcc_cols = [f'MFCC_{i+1}' for i in range(mfcc_mean.shape[1])]
    
    # Stats indices: 0-11(Chroma), 12-18(Contrast), 19(ZCR), 20(RMS)
    chroma_cols = [f'Chroma_{i+1}' for i in range(12)]
    contrast_cols = [f'Contrast_{i+1}' for i in range(7)]
    stats_cols = chroma_cols + contrast_cols + ['ZCR', 'RMS']
    
    sent_col = ['Sentiment']
    
    # Combine Data
    # Note: X_stats is already (N, 21)
    data_matrix = np.hstack([mfcc_mean, X_stats, sent_vec])
    all_cols = mfcc_cols + stats_cols + sent_col
    
    df_corr = pd.DataFrame(data_matrix, columns=all_cols)
    
    # 4. Compute Correlation
    corr = df_corr.corr()
    
    # 5. Plot
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr, dtype=bool)) # Hide upper triangle (redundant)
    
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=True, yticklabels=True)
    
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_feature_label_correlation(X_mfcc, X_stats, X_sent, y_data, class_names, title="Feature vs. Emotion Correlation"):
    """
    Plots a heatmap showing the correlation between each feature and each specific emotion class.
    Use this to answer: "Which features are high/low for Anger?"
    """
    print(f"Generating {title}...")
    
    # 1. Flatten MFCCs (Time Series -> Global Mean) for simple correlation
    if len(X_mfcc.shape) == 3:
        mfcc_mean = np.mean(X_mfcc, axis=1)
    else:
        mfcc_mean = X_mfcc
        
    # 2. Prepare Feature DataFrame
    # Define Column Names
    mfcc_cols = [f'MFCC_{i+1}' for i in range(mfcc_mean.shape[1])]
    # Stats indices: 0-11(Chroma), 12-18(Contrast), 19(ZCR), 20(RMS)
    stats_cols = [f'Chroma_{i+1}' for i in range(12)] + [f'Contrast_{i+1}' for i in range(7)] + ['ZCR', 'RMS']
    
    # Reshape sentiment
    if len(X_sent.shape) == 1:
        sent_vec = X_sent.reshape(-1, 1)
    else:
        sent_vec = X_sent
        
    # Stack all features
    X_matrix = np.hstack([mfcc_mean, X_stats, sent_vec])
    feat_cols = mfcc_cols + stats_cols + ['Sentiment']
    df_features = pd.DataFrame(X_matrix, columns=feat_cols)
    
    # 3. Prepare One-Hot Encoded Labels
    # We convert the single 'Label' column into multiple binary columns (e.g., "is_joy", "is_sadness")
    df_labels = pd.DataFrame()
    for idx, name in enumerate(class_names):
        # Create binary column: 1 if sample is this emotion, 0 otherwise
        df_labels[name] = (y_data == idx).astype(int)
        
    # 4. Compute Correlation
    # We correlate every feature column with every emotion binary column
    correlation_data = pd.DataFrame(index=feat_cols, columns=class_names)
    
    for emotion in class_names:
        correlation_data[emotion] = df_features.corrwith(df_labels[emotion])
        
    # 5. Plot Heatmap
    plt.figure(figsize=(10, 18)) # Tall plot to fit all features
    sns.heatmap(correlation_data, cmap='RdBu_r', center=0, annot=False, linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title(title, fontsize=15)
    plt.xlabel("Emotion Class", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.show()