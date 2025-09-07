## Meditation Module: Component Documentation

This document outlines the key components of the multi-modal meditation module. Each component is described with its required inputs, core processing tasks, and expected outputs.

### Data Input Layer
Responsible for gathering raw data from various sources before any processing begins.

- **1. Diagnosis Data (DI)**
  - **Input**: Raw diagnosis information in JSON or plain text format (symptoms, medical history, mental health status)
  - **Processing**: Raw data ingestion only
  - **Output**: Unprocessed diagnosis data

- **2. Video Stream (VI)**
  - **Input**: Real-time video stream, typically 30 fps
  - **Processing**: Ingestion of raw RGB frames
  - **Output**: Continuous stream of raw RGB frames

- **3. Audio Input (AI)**
  - **Input**: Raw audio, e.g., microphone 16 kHz WAV
  - **Processing**: Capture raw audio signal
  - **Output**: Unprocessed audio data

- **4. User Feedback (UI)**
  - **Input**: Ratings, free-text responses, surveys
  - **Processing**: Raw feedback reception
  - **Output**: Unprocessed user feedback

### Data Preprocessing Layer
Cleans and transforms raw inputs into a standardized format for feature extraction.

- **1. Diagnosis Preprocessor (DP)**
  - **Input**: Raw diagnosis data
  - **Processing**:
    - Text Cleaning: remove special characters, stop words
    - Tokenization: split into words/sub-words
    - Feature Engineering: symptom counts, severity scores
  - **Output**: Cleaned, tokenized, engineered diagnosis data

- **2. Video Preprocessor (VP)**
  - **Input**: Raw video frames
  - **Processing**:
    - Frame Extraction: sample/extract frames
    - Resizing: standard dimensions (e.g., 224×224)
    - Normalization: scale pixels to [0, 1]
  - **Output**: Preprocessed frames for vision model

- **3. Audio Preprocessor (AP)**
  - **Input**: Raw audio
  - **Processing**:
    - MFCC Features
    - Spectrogram Generation
    - Noise Reduction
  - **Output**: Processed audio features (MFCCs, spectrograms)

- **4. User Data Preprocessor (UP)**
  - **Input**: Raw user feedback
  - **Processing**:
    - Feedback Aggregation (multi-session)
    - Historical Trend Analysis
  - **Output**: Aggregated and analyzed user data

### Feature Extraction Layer
Transforms preprocessed data into numerical representations (embeddings).

- **1. Diagnosis Encoder (DE)**
  - **Input**: Preprocessed diagnosis text
  - **Processing**:
    - BERT/BioBERT embeddings
    - Medical Entity Recognition
    - Symptom Embeddings
  - **Output**: Numerical diagnosis embeddings
  - Implementations: `Encoders/diagnosis_encoder.py`, `Encoders/diagnosis_encoder_simple.py`

- **2. Vision Encoder (VE)**
  - **Input**: Preprocessed video frames
  - **Processing**:
    - ResNet50/EfficientNet visual features
    - Keypoint Detection
    - Pose Embeddings
  - **Output**: Posture and visual embeddings
  - Implementation: `Encoders/vision_encoder.py`

- **3. Audio Encoder (AE)**
  - **Input**: Preprocessed audio features
  - **Processing**:
    - Mel-frequency analysis on MFCCs
    - Voice Activity Detection (VAD)
    - Emotion Recognition (heuristic)
  - **Output**: Audio and emotion embeddings
  - Implementation: `Encoders/audio_encoder.py`

- **4. User Profile Encoder (UE)**
  - **Input**: Aggregated user data
  - **Processing**:
    - Preference Embeddings
    - Progress Vectors
    - Behavioral Patterns
  - **Output**: Comprehensive user profile embedding
  - Implementation: `Encoders/user_profile_encoder.py`

### Core ML Models Layer
Primary ML models that drive the system.

- **1. Meditation Selector Model (MSM)**
  - **Input**: Diagnosis + User Profile embeddings
  - **Processing**: Multi-class classifier (e.g., Random Forest + NN)
  - **Output**: Recommended meditation type + confidence
  - Current Module: `Core_engine/meditation_selector.py` (hybrid rule-based + ML-enhanced similarity)

- **2. Posture Detection Model (PDM)**
  - **Input**: Pose embeddings + visual features (VE)
  - **Processing**: Hybrid CNN + LSTM (real-time keypoint tracking, posture quality over time)
  - **Output**: Real-time posture score + keypoint locations
  - Implementation: `Core_engine/Posture_detector.py`

- **3. Text-to-Speech Model (TTSM)**
  - **Input**: Text script + emotion data (e.g., from DE)
  - **Processing**: Multi-speaker, emotion-aware TTS
  - **Output**: Synthesized speech audio

- **4. Audio Response Model (ARM)**
  - **Input**: Audio embeddings (AE)
  - **Processing**:
    - Speech-to-Text
    - Natural Language Understanding (NLU)
    - Sentiment Analysis
  - **Output**: Recognized intent, sentiment, and transcript

### Model Fusion Layer
Combines outputs from core models for holistic decisions.

- **1. Multi-Modal Fusion (MF)**
  - **Input**: Outputs from MSM, PDM, TTSM, ARM
  - **Processing**:
    - Attention Mechanism
    - Weighted Feature Combination
    - Cross-modal Correlation
  - **Output**: Fused, multi-modal feature set

- **2. Decision Manager (DM)**
  - **Input**: Fused features + Quality Monitor confidence
  - **Processing**:
    - Rule-based + ML Hybrid policies
    - Real-time Adaptation
  - **Output**: Actionable decisions for output generation

- **3. Quality Monitor (QM)**
  - **Input**: Core model outputs + confidence scores
  - **Processing**:
    - Performance Tracking
    - Confidence Thresholding
    - Fallback Mechanisms
  - **Output**: Reliability metrics, alerts, fallback flags

### Output Generation Layer
Translates decisions into user-facing content and actions.

- **1. Meditation Recommendation (MR)**
  - **Input**: Decision Manager outputs
  - **Processing**:
    - Personalized Scripts
    - Duration Optimization
    - Difficulty Adjustment
  - **Output**: Structured meditation session plan

- **2. Posture Corrections (PC)**
  - **Input**: Decision Manager outputs
  - **Processing**:
    - Real-time Feedback
    - Gentle Voice Prompts
    - Visual Cues Generation
  - **Output**: Actionable posture correction prompts

- **3. Audio Generation (AG)**
  - **Input**: Decision Manager outputs
  - **Processing**:
    - High-quality Speech (TTSM)
    - Background Music Mixing
    - Spatial Audio Effects
  - **Output**: Complete audio track for the session

- **4. Progress Monitoring (PM)**
  - **Input**: Decision Manager outputs
  - **Processing**:
    - Session Tracking
    - Improvement Metrics
    - Adaptive Learning
  - **Output**: Progress reports and data for storage

### Model Training & Inference Layer
Manages model lifecycle.

- **1. Training Pipeline (TD)**
  - **Input**: Datasets from Storage Layer
  - **Processing**:
    - Data Augmentation
    - Cross-validation
    - Hyperparameter Tuning
  - **Output**: Trained, validated models

- **2. Model Inference (MI)**
  - **Input**: New data + trained models
  - **Processing**:
    - Batch Processing
    - Real-time Inference
    - Edge Optimization
  - **Output**: Real-time predictions and outputs

- **3. Model Updates (MU)**
  - **Input**: New data + Quality Monitor metrics
  - **Processing**:
    - Continuous Learning
    - A/B Testing
    - Performance Monitoring
  - **Output**: Updated model versions

### Data Storage Layer
Repository for all data used by the system.

- **1. Medical Dataset (MDS)**
  - **Input**: Clinical data
  - **Processing**: N/A
  - **Output**: Stored medical data (diagnosis–meditation pairs, ontologies)

- **2. Posture Dataset (PDS)**
  - **Input**: Annotated posture data
  - **Processing**: N/A
  - **Output**: Stored posture data (annotated poses, corrections)

- **3. Voice Dataset (VDS)**
  - **Input**: Audio data
  - **Processing**: N/A
  - **Output**: Stored audio data (multi-speaker corpus, scripts)

- **4. User Dataset (UDS)**
  - **Input**: User data
  - **Processing**: N/A
  - **Output**: Stored user data (session history, preferences, feedback)


