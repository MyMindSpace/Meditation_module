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

---

## 🚀 Meditation Module Workflow & Usage Guide

### Overview
The Meditation Module is a complete AI-powered system that captures live audio/video input, processes user data, and generates personalized meditation recommendations with TTS-ready scripts.

### 📋 System Requirements
- **Python 3.11+**
- **Audio/Video Hardware**: Microphone and camera for live capture
- **Dependencies**: See installation section below

### 🔧 Installation & Setup

#### 1. Install Core Dependencies
```bash
pip install librosa soundfile opencv-python transformers xgboost joblib pandas numpy matplotlib
```

#### 2. Install Live Capture Dependencies
```bash
python setup_live_capture.py
# OR manually:
pip install sounddevice soundfile opencv-python
```

#### 3. Setup API Keys (Optional - for advanced script generation)
- Get Gemini API key from https://makersuite.google.com/app/apikey
- Edit `Core_engine/meditation_script_generator.py` and add your key

### 🎯 Quick Start - Live Meditation Session

#### Option 1: Complete Interactive Flow
```bash
python run_live_pipeline.py
```
**What it does:**
1. 🎙️ Captures live audio from microphone
2. 🎥 Captures live video from camera 
3. 📝 Prompts for diary entry
4. 🔄 Processes all data through AI pipeline
5. 🧘 Generates personalized meditation script

#### Option 2: Manual Capture + Processing
```bash
# Step 1: Capture live audio/video
python live_capture.py --mode both --duration 30 --create-librispeech

# Step 2: Run pipeline
python run_full_pipeline.py
```

#### Option 3: Using Existing Files
```bash
# Place your files in preprocess_input/ and run:
python run_full_pipeline.py
```

### 📊 Detailed Workflow

#### Phase 1: Data Input & Capture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Live Audio     │    │   Live Video     │    │  Diary Entry    │
│  (Microphone)   │    │   (Camera)       │    │  (Text Input)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ .wav files in   │    │ .mp4 files in    │    │ .json files in  │
│ LibriSpeech     │    │ preprocess_input │    │ preprocess_input│
│ structure       │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

#### Phase 2: Preprocessing
```
Audio Input → Audio Preprocessor → MFCC Features, Spectrograms
Video Input → Video Preprocessor → Frame Extraction, Pose Analysis  
Diary Entry → User Preprocessor  → Text Analysis, Sentiment
Diagnosis   → Diagnosis Processor → Symptom Analysis, Medical NLP
```

#### Phase 3: Feature Encoding
```
Audio Features    → Audio Encoder      → Audio Embeddings
Video Frames      → Vision Encoder     → Posture Embeddings
User Feedback     → User Profile Encoder → User Embeddings
Diagnosis Data    → Diagnosis Encoder  → Medical Embeddings
```

#### Phase 4: AI Processing & Fusion
```
All Embeddings → Fusion Module → Combined Analysis
              → Decision Manager → Rule-based + ML Decisions
              → Quality Monitor → Confidence Scoring
```

#### Phase 5: Meditation Selection & Script Generation
```
Fused Data → Meditation Selector → Personalized Recommendations
          → Script Generator → TTS-Ready Meditation Script
```

### 📁 File Structure & Outputs

#### Input Files (preprocess_input/)
```
preprocess_input/
├── LibriSpeech_Live_TIMESTAMP/     # Live captured audio
├── live_video_TIMESTAMP.mp4       # Live captured video
├── diary_entry.json               # User diary/intent
├── diagnosis_data.json            # Medical diagnosis (optional)
└── user_feedback.json             # Historical feedback (optional)
```

#### Output Files
```
meditation_recommendations_output.json           # Top meditation recommendations
Core_engine/generated_meditation_script.json    # Complete TTS-ready script
preprocess_output/                               # Intermediate processing files
encoder_output/                                  # AI embeddings and features
```

### 🎨 Usage Examples

#### Example 1: Anxiety Relief Session
```bash
python run_live_pipeline.py
# Choose: Audio + Video (30 seconds)
# Diary: "Feeling anxious about work presentation tomorrow"
# Output: Breathing meditation with grounding techniques
```

#### Example 2: Sleep Preparation
```bash
python live_capture.py --mode audio --duration 60
# Diary: "Having trouble sleeping, mind is racing"
# Output: Yoga Nidra or body scan meditation
```

#### Example 3: Focus Enhancement
```bash
python run_live_pipeline.py
# Diary: "Need to improve concentration for studying"
# Output: Focused attention or mantra meditation
```

### 🔧 Advanced Configuration

#### Customize Capture Settings
```bash
# List available devices
python live_capture.py --list-devices

# Use specific audio/video device
python live_capture.py --audio-device 1 --video-device 0 --duration 45

# Audio only with custom duration
python live_capture.py --mode audio --duration 120
```

#### Debug Pipeline Steps
```bash
# Run individual components
python preprocessing_unit/audio_preprocessor.py --input [path] --output [path]
python Encoders/audio_encoder.py --input [path] --output [path]
python Core_engine/meditation_selector.py --rule-only
```

### 📈 Output Format

#### Meditation Recommendations (JSON)
```json
{
  "recommendations": [
    {
      "meditation_type": "Mindful Breathing Meditation",
      "confidence": 0.92,
      "rationale": "Recommended for anxiety and stress relief",
      "source": "rule_base_disorder"
    }
  ],
  "method": "hybrid"
}
```

#### Generated Script (JSON)
```json
{
  "meditation_type": "Body Scan Meditation",
  "instructions": "Step-by-step guidance...",
  "script": "Complete TTS-ready meditation narration...",
  "duration_minutes": "8-10",
  "format": "TTS-ready"
}
```

### 🐛 Troubleshooting

#### Common Issues
1. **Audio/Video Capture Fails**
   - Check device permissions
   - Run `python live_capture.py --list-devices`
   - Try different device indices

2. **Pipeline Timeout**
   - Large audio/video files take longer to process
   - Default timeout is 10 minutes
   - Check terminal output for specific errors

3. **Missing Dependencies**
   - Run `python setup_live_capture.py`
   - Install missing packages individually

4. **Script Generation Fails**
   - Check Gemini API key in `meditation_script_generator.py`
   - Verify internet connection
   - Check pandas/numpy compatibility

### 🎯 Next Steps After Generation
1. **Review** generated meditation recommendations
2. **Use** the TTS-ready script with your preferred text-to-speech system
3. **Practice** the recommended meditation
4. **Provide feedback** to improve future recommendations

---