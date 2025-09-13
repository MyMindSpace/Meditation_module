"""
Run the full meditation module pipeline with pre-defined test diary entries.

Usage: python test_pipeline.py <test_case_name>
Where test_case_name is one of: depression, adhd, anxiety, sleep, trauma
"""
import subprocess
import sys
import os
import json
import datetime

def run_pipeline_with_diary(diary_text, test_case_name):
    """Run the full pipeline with a given diary entry."""
    
    # Paths
    DIARY_INPUT = "preprocess_input/diary_entry.json"
    USER_FEEDBACK_INPUT = "preprocess_input/user_feedback.json"
    DIAGNOSIS_INPUT = "preprocess_input/diagnosis_data.json"
    AUDIO_INPUT = "preprocess_input/LibriSpeech"
    VIDEO_INPUT = "preprocess_input/20250903_125923.mp4"

    PREPROCESS_OUTPUT = "preprocess_output/"
    ENCODER_OUTPUT = "encoder_output/"
    FUSED_DECISION = "preprocess_output/fused_decision.json"
    MEDITATION_CSV = "Core_engine/meditation.csv"
    SCRIPT_OUTPUT = f"Core_engine/generated_meditation_script_{test_case_name}.json"

    print(f"\n{'='*60}")
    print(f"RUNNING TEST CASE: {test_case_name.upper()}")
    print(f"{'='*60}")
    print(f"Diary Entry: {diary_text}")
    print(f"{'='*60}\n")

    # Create diary entry JSON
    diary_entry = {
        "user_id": f"test_user_{test_case_name}",
        "timestamp": datetime.datetime.now().isoformat(),
        "entry": diary_text
    }
    with open(DIARY_INPUT, "w", encoding="utf-8") as f:
        json.dump([diary_entry], f, indent=2, ensure_ascii=False)
    print(f"✓ Diary entry saved to {DIARY_INPUT}")

    # Helper to run a python script
    def run_script(script, args=None, description=""):
        cmd = [sys.executable, script]
        if args:
            cmd += args
        print(f"\n🔄 {description}: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(f"✓ Output: {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        return True

    try:
        # 1. Diary preprocessor
        if not run_script("preprocessing_unit/user_preprocessor.py", 
                         ["--input", DIARY_INPUT, "--output", os.path.join(PREPROCESS_OUTPUT, "user_feedback_processed.json")],
                         "Running diary preprocessor"):
            return False

        # 2. Other preprocessors
        if not run_script("preprocessing_unit/audio_preprocessor.py", 
                         ["--input", AUDIO_INPUT, "--output", os.path.join(PREPROCESS_OUTPUT, "audio_features")],
                         "Running audio preprocessor"):
            return False
            
        if not run_script("preprocessing_unit/diagnosis_processor.py", 
                         ["--input", DIAGNOSIS_INPUT, "--output", os.path.join(PREPROCESS_OUTPUT, "diagnosis_processed.json")],
                         "Running diagnosis preprocessor"):
            return False
            
        if not run_script("preprocessing_unit/video_preprocessor.py", 
                         ["--input", VIDEO_INPUT, "--output", os.path.join(PREPROCESS_OUTPUT, "video_frames")],
                         "Running video preprocessor"):
            return False

        # 3. Encoders
        if not run_script("Encoders/user_profile_encoder.py", 
                         ["--input", os.path.join(PREPROCESS_OUTPUT, "user_feedback_processed.json"), 
                          "--output", os.path.join(ENCODER_OUTPUT, "user_profiles_encoded.json")],
                         "Running user profile encoder"):
            return False
            
        if not run_script("Encoders/audio_encoder.py", 
                         ["--input", os.path.join(PREPROCESS_OUTPUT, "audio_features"), 
                          "--output", os.path.join(ENCODER_OUTPUT, "audio_encoded.json")],
                         "Running audio encoder"):
            return False
            
        if not run_script("Encoders/diagnosis_encoder.py", 
                         ["--input", os.path.join(PREPROCESS_OUTPUT, "diagnosis_processed.json"), 
                          "--output", os.path.join(ENCODER_OUTPUT, "diagnosis_encoded.json")],
                         "Running diagnosis encoder"):
            return False
            
        if not run_script("Encoders/vision_encoder.py", 
                         ["--input", os.path.join(PREPROCESS_OUTPUT, "video_frames"), 
                          "--output", os.path.join(ENCODER_OUTPUT, "vision_encoded.json")],
                         "Running vision encoder"):
            return False

        # 4. Core engine modules
        if not run_script("Core_engine/fusion.py", description="Running fusion module"):
            return False
            
        if not run_script("Core_engine/decision_manager.py", description="Running decision manager"):
            return False
            
        if not run_script("Core_engine/meditation_selector.py", ["--rule-only"], description="Running meditation selector"):
            return False

        # 4.5. Bridge script to update fused decision with selected meditation
        if not run_script("Core_engine/bridge_selector_to_script.py", description="Updating fused decision with meditation selection"):
            return False

        # 5. Generate meditation script
        if not run_script("Core_engine/meditation_script_generator.py", description="Running meditation script generator"):
            return False

        print(f"\n🎉 Pipeline completed successfully for {test_case_name} test case!")
        print(f"📋 Check outputs in:")
        print(f"   - Meditation recommendations: meditation_recommendations_output.json")
        print(f"   - Generated meditation script: Core_engine/generated_meditation_script.json")
        print(f"   - Processed data: {PREPROCESS_OUTPUT}")
        print(f"   - Encoded data: {ENCODER_OUTPUT}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_pipeline.py <test_case_name>")
        print("Available test cases: depression, adhd, anxiety, sleep, trauma")
        return
    
    test_case = sys.argv[1].lower()
    
    # Read the appropriate diary file
    diary_file = f"test_cases/diary_{test_case}.txt"
    
    if not os.path.exists(diary_file):
        print(f"❌ Test case file not found: {diary_file}")
        print("Available test cases: depression, adhd, anxiety, sleep, trauma")
        return
    
    with open(diary_file, 'r', encoding='utf-8') as f:
        diary_text = f.read().strip()
    
    success = run_pipeline_with_diary(diary_text, test_case)
    
    if success:
        print(f"\n✅ Test case '{test_case}' completed successfully!")
    else:
        print(f"\n❌ Test case '{test_case}' failed!")

if __name__ == "__main__":
    main()
