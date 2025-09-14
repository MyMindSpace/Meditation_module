"""
Run the full meditation module pipeline from diary entry to meditation script.

Steps:
1. Take diary entry input (JSON file)
2. Run diary preprocessor
3. Run all preprocess
ors (audio, diagnosis, user feedback, video)
4. Run encoders
5. Run core engine modules (fusion, decision, selector)
6. Generate meditation script
"""
import subprocess
import sys
import os

# Paths
DIARY_INPUT = "preprocess_input/diary_entry.json"
USER_FEEDBACK_INPUT = "preprocess_input/user_feedback.json"
DIAGNOSIS_INPUT = "preprocess_input/diagnosis_data.json"

# Auto-detect audio and video inputs (prioritize live captures)
import glob
from pathlib import Path

# Check for live audio captures first
live_audio_dirs = glob.glob("preprocess_input/LibriSpeech_Live*")
if live_audio_dirs:
    AUDIO_INPUT = live_audio_dirs[-1]  # Use most recent
    print(f"Using live audio: {AUDIO_INPUT}")
else:
    AUDIO_INPUT = "preprocess_input/LibriSpeech"
    print(f"Using default audio: {AUDIO_INPUT}")

# Check for live video captures first
live_videos = glob.glob("preprocess_input/live_video_*.mp4")
if live_videos:
    VIDEO_INPUT = live_videos[-1]  # Use most recent
    print(f"Using live video: {VIDEO_INPUT}")
else:
    # Fallback to any mp4 file in preprocess_input
    fallback_videos = glob.glob("preprocess_input/*.mp4")
    if fallback_videos:
        VIDEO_INPUT = fallback_videos[0]
        print(f"Using fallback video: {VIDEO_INPUT}")
    else:
        VIDEO_INPUT = "preprocess_input/20250903_125923.mp4"
        print(f"Using default video: {VIDEO_INPUT}")

PREPROCESS_OUTPUT = "preprocess_output/"
ENCODER_OUTPUT = "encoder_output/"
FUSED_DECISION = "preprocess_output/fused_decision.json"
MEDITATION_CSV = "Core_engine/meditation.csv"
SCRIPT_OUTPUT = "Core_engine/generated_meditation_script.json"


# Prompt user for diary entry and write to diary_entry.json
import json
diary_text = input("Enter your diary entry (describe your mood, thoughts, or experience):\n>")
import datetime
diary_entry = {
    "user_id": "user_1",
    "timestamp": datetime.datetime.now().isoformat(),
    "entry": diary_text
}
with open(DIARY_INPUT, "w", encoding="utf-8") as f:
    json.dump([diary_entry], f, indent=2, ensure_ascii=False)
print(f"Diary entry saved to {DIARY_INPUT}\n")

# Helper to run a python script
def run_script(script, args=None):
    cmd = [sys.executable, script]
    if args:
        cmd += args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Script {script} failed.")

# 1. Diary preprocessor
run_script("preprocessing_unit/user_preprocessor.py", ["--input", DIARY_INPUT, "--output", os.path.join(PREPROCESS_OUTPUT, "user_feedback_processed.json")])

# 2. Other preprocessors
run_script("preprocessing_unit/audio_preprocessor.py", ["--input", AUDIO_INPUT, "--output", os.path.join(PREPROCESS_OUTPUT, "audio_features")])
run_script("preprocessing_unit/diagnosis_processor.py", ["--input", DIAGNOSIS_INPUT, "--output", os.path.join(PREPROCESS_OUTPUT, "diagnosis_processed.json")])
# Process video - handle both single files and directories
if os.path.isfile(VIDEO_INPUT):
    # Single video file - create temp directory structure
    temp_video_dir = os.path.join("preprocess_input", "temp_video_dir")
    os.makedirs(temp_video_dir, exist_ok=True)
    
    # Copy video file to temp directory
    import shutil
    temp_video_path = os.path.join(temp_video_dir, os.path.basename(VIDEO_INPUT))
    shutil.copy2(VIDEO_INPUT, temp_video_path)
    
    run_script("preprocessing_unit/video_preprocessor.py", ["--input", temp_video_dir, "--output", os.path.join(PREPROCESS_OUTPUT, "video_frames")])
    
    # Clean up temp directory
    shutil.rmtree(temp_video_dir, ignore_errors=True)
else:
    # Directory input
    run_script("preprocessing_unit/video_preprocessor.py", ["--input", VIDEO_INPUT, "--output", os.path.join(PREPROCESS_OUTPUT, "video_frames")])

# 3. Encoders
run_script("Encoders/user_profile_encoder.py", ["--input", os.path.join(PREPROCESS_OUTPUT, "user_feedback_processed.json"), "--output", os.path.join(ENCODER_OUTPUT, "user_profiles_encoded.json")])
run_script("Encoders/audio_encoder.py", ["--input", os.path.join(PREPROCESS_OUTPUT, "audio_features"), "--output", os.path.join(ENCODER_OUTPUT, "audio_encoded.json")])
run_script("Encoders/diagnosis_encoder.py", ["--input", os.path.join(PREPROCESS_OUTPUT, "diagnosis_processed.json"), "--output", os.path.join(ENCODER_OUTPUT, "diagnosis_encoded.json")])
run_script("Encoders/vision_encoder.py", ["--input", os.path.join(PREPROCESS_OUTPUT, "video_frames"), "--output", os.path.join(ENCODER_OUTPUT, "vision_encoded.json"), "--include-live-posture"])

# 4. Core engine modules
run_script("Core_engine/pipeline.py")
run_script("Core_engine/meditation_selector.py")
run_script("Core_engine/fusion.py")
run_script("Core_engine/decision_manager.py")

# 5. Generate meditation script
run_script("Core_engine/meditation_script_generator.py")

print("\nPipeline complete! See meditation script in:", SCRIPT_OUTPUT)
