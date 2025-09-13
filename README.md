# 🧘 AI-Powered Meditation Module

An intelligent meditation recommendation system that captures live audio/video input and generates personalized meditation scripts using advanced AI processing.

## ✨ Features

- 🎙️ **Live Audio Capture** - Record from microphone with automatic processing
- 🎥 **Live Video Capture** - Capture posture and visual cues from camera
- 🤖 **AI-Powered Analysis** - Multi-modal AI processing for personalized recommendations
- 🧘 **Smart Meditation Selection** - Clinical rule-based + ML hybrid recommendations
- 📝 **TTS-Ready Scripts** - Complete meditation scripts ready for text-to-speech
- 🔄 **End-to-End Pipeline** - From live capture to final meditation script

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Core dependencies
pip install librosa soundfile opencv-python transformers xgboost joblib pandas numpy

# Live capture dependencies  
python setup_live_capture.py
```

### 2. Run Live Meditation Session
```bash
python run_live_pipeline.py
```

**What happens:**
1. Captures your audio/video (30 seconds default)
2. Asks for diary entry about your current state
3. Processes everything through AI pipeline
4. Generates personalized meditation recommendation + script

### 3. Check Results
- **Recommendations**: `meditation_recommendations_output.json`
- **TTS Script**: `Core_engine/generated_meditation_script.json`

## 📋 Usage Options

### Interactive Live Session
```bash
python run_live_pipeline.py
# Follow prompts for audio/video capture and diary entry
```

### Manual Capture
```bash
# Capture 60 seconds of audio + video
python live_capture.py --mode both --duration 60 --create-librispeech

# Run processing pipeline
python run_full_pipeline.py
```

### Audio Only
```bash
python live_capture.py --mode audio --duration 30 --create-librispeech
python run_full_pipeline.py
```

## 🎯 Example Workflows

### For Anxiety Relief
```
Input: "Feeling anxious about work meeting"
Audio: Detects tense breathing patterns
Video: Detects tense posture
Output: Breathing meditation with grounding techniques
```

### For Sleep Preparation  
```
Input: "Having trouble sleeping, mind racing"
Audio: Detects agitated speech patterns
Video: Detects restless posture
Output: Yoga Nidra or progressive muscle relaxation
```

### For Focus Enhancement
```
Input: "Need better concentration for studying"
Audio: Detects scattered attention in voice
Video: Detects fidgeting or poor posture
Output: Focused attention or mantra meditation
```

## 📁 Project Structure

```
meditation_module/
├── live_capture.py              # Live audio/video capture
├── run_live_pipeline.py         # Complete interactive pipeline
├── run_full_pipeline.py         # Core pipeline processor
├── setup_live_capture.py        # Dependency installer
│
├── preprocessing_unit/          # Data preprocessing
│   ├── audio_preprocessor.py
│   ├── video_preprocessor.py
│   └── user_preprocessor.py
│
├── Encoders/                    # AI feature extraction
│   ├── audio_encoder.py
│   ├── vision_encoder.py
│   └── user_profile_encoder.py
│
├── Core_engine/                 # AI decision making
│   ├── meditation_selector.py   # Smart recommendations
│   ├── meditation_script_generator.py
│   └── fusion.py               # Multi-modal fusion
│
├── preprocess_input/           # Input data directory
├── preprocess_output/          # Processed data
├── encoder_output/             # AI embeddings
└── docs/                       # Detailed documentation
```

## 🔧 Advanced Usage

### List Available Devices
```bash
python live_capture.py --list-devices
```

### Custom Duration & Devices
```bash
python live_capture.py --mode both --duration 120 --audio-device 1 --video-device 0
```

### Debug Individual Components
```bash
python Core_engine/meditation_selector.py --rule-only
python preprocessing_unit/audio_preprocessor.py --input [path] --output [path]
```

## 📊 Output Format

### Meditation Recommendations
```json
{
  "recommendations": [
    {
      "meditation_type": "Mindful Breathing Meditation",
      "confidence": 0.92,
      "rationale": "Recommended for anxiety relief",
      "source": "rule_base_disorder"
    }
  ],
  "total_candidates": 5,
  "method": "hybrid"
}
```

### Generated Script
```json
{
  "meditation_type": "Body Scan Meditation", 
  "instructions": "Lie down comfortably...",
  "script": "Find a comfortable position... [complete TTS script]",
  "duration_minutes": "8-10",
  "format": "TTS-ready"
}
```

## 🛠️ Requirements

- **Python 3.11+**
- **Hardware**: Microphone and camera for live capture
- **Storage**: ~500MB for dependencies and model files
- **Internet**: Required for initial setup and AI script generation

### Core Dependencies
- `librosa` - Audio processing
- `opencv-python` - Video processing  
- `transformers` - AI models
- `sounddevice` - Live audio capture
- `pandas` - Data processing

## 🐛 Troubleshooting

### Audio/Video Issues
```bash
# Check available devices
python live_capture.py --list-devices

# Test with different device indices
python live_capture.py --audio-device 0 --video-device 1
```

### Pipeline Fails
- Check all dependencies are installed
- Verify input files exist in `preprocess_input/`
- Check terminal output for specific error messages
- Increase timeout for large files

### Script Generation Issues
- Verify Gemini API key in `Core_engine/meditation_script_generator.py`
- Check internet connection
- Update pandas/numpy: `pip install --upgrade pandas numpy`

## 📚 Documentation

- **Complete Guide**: `docs/Meditation_Module_Component_Documentation.md`
- **API References**: Individual module docstrings
- **Examples**: `docs/` directory

## 🎯 What's Next?

After running the pipeline:
1. Review your personalized recommendations in `meditation_recommendations_output.json`
2. Use the TTS script from `Core_engine/generated_meditation_script.json` with your preferred text-to-speech software
3. Practice the recommended meditation
4. The system learns from your usage patterns for better future recommendations

---

## 🤝 Contributing

This meditation module is designed to be modular and extensible. Each component can be run independently for testing and development.

**Happy Meditating! 🧘‍♀️✨**
