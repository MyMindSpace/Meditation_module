"""
Live Input Pipeline for Meditation Module

This script captures live audio/video input from the user and then runs
the full meditation pipeline with the captured data.
"""

import json
import subprocess
import sys
import datetime
from pathlib import Path

def run_live_pipeline():
    """Run the complete pipeline with live input capture"""
    
    print("🎙️ Welcome to the Live Meditation Pipeline! 🎥")
    print("This will capture your audio/video and generate personalized meditation recommendations.")
    print()
    
    # Get user preferences
    print("Choose capture mode:")
    print("1. Audio + Video (recommended)")
    print("2. Audio only") 
    print("3. Skip capture (use existing files)")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    # Get recording duration
    if choice in ['1', '2']:
        duration = input("Recording duration in seconds (default: 30): ").strip()
        if not duration:
            duration = "30"
        
        try:
            duration = int(duration)
        except ValueError:
            duration = 30
            print(f"Invalid duration, using default: {duration} seconds")
    
    # Capture live input
    audio_file = None
    video_file = None
    
    if choice == "1":  # Audio + Video
        print(f"\\n🎬 Starting audio + video capture for {duration} seconds...")
        try:
            result = subprocess.run([
                sys.executable, "live_capture.py", 
                "--mode", "both",
                "--duration", str(duration),
                "--create-librispeech"
            ], capture_output=True, text=True, check=True)
            print("✅ Audio and video captured successfully!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during capture: {e}")
            print("Error output:", e.stderr)
            return False
            
    elif choice == "2":  # Audio only
        print(f"\\n🎙️ Starting audio capture for {duration} seconds...")
        try:
            result = subprocess.run([
                sys.executable, "live_capture.py", 
                "--mode", "audio",
                "--duration", str(duration),
                "--create-librispeech"
            ], capture_output=True, text=True, check=True)
            print("✅ Audio captured successfully!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during capture: {e}")
            print("Error output:", e.stderr)
            return False
    
    # Get diary entry
    print("\\n📝 Please provide your diary entry:")
    print("Describe your current mood, thoughts, or meditation goals...")
    diary_text = input("> ")
    
    if not diary_text.strip():
        diary_text = "I want to practice meditation and improve my well-being."
    
    # Save diary entry
    diary_entry = {
        "user_id": "live_user",
        "timestamp": datetime.datetime.now().isoformat(),
        "entry": diary_text
    }
    
    diary_path = "preprocess_input/diary_entry.json"
    with open(diary_path, "w", encoding="utf-8") as f:
        json.dump([diary_entry], f, indent=2, ensure_ascii=False)
    
    print(f"✅ Diary entry saved to {diary_path}")
    
    # Run the full pipeline
    print("\\n🔄 Running the meditation pipeline...")
    print("This may take a few minutes as it processes audio, video, and generates your script...")
    try:
        # Run pipeline with empty input (since we already saved diary entry)
        result = subprocess.run([
            sys.executable, "run_full_pipeline.py"
        ], input="\\n", capture_output=True, text=True, timeout=600)  # 10 minute timeout for processing
        
        if result.returncode == 0:
            print("✅ Pipeline completed successfully!")
        else:
            print("⚠️ Pipeline completed with some warnings")
            
        print("\\n📋 Pipeline Output:")
        print(result.stdout)
        
        if result.stderr:
            print("\\n⚠️ Pipeline Warnings/Errors:")
            print(result.stderr)
        
        # Check for output files
        output_files = [
            ("meditation_recommendations_output.json", "Meditation Recommendations"),
            ("Core_engine/generated_meditation_script.json", "Generated Meditation Script"),
            ("preprocess_output/fused_decision.json", "Fusion Decision"),
            ("encoder_output/audio_encoded.json", "Audio Encoding"),
            ("encoder_output/vision_encoded.json", "Video Encoding")
        ]
        
        print("\\n📂 Generated Files:")
        files_found = 0
        for file_path, description in output_files:
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                print(f"✅ {description}: {file_path} ({file_size} bytes)")
                files_found += 1
            else:
                print(f"❌ {description}: {file_path} (not found)")
        
        # Show meditation script if available
        script_path = "Core_engine/generated_meditation_script.json"
        if Path(script_path).exists():
            print("\\n🧘 Your Personalized Meditation Script:")
            print("=" * 50)
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    script_data = json.load(f)
                print(f"Type: {script_data.get('meditation_type', 'Unknown')}")
                print(f"Duration: {script_data.get('duration_minutes', 'Unknown')} minutes")
                print(f"Generated: {script_data.get('generated_at', 'Unknown')}")
                print("\\nInstructions:")
                print(script_data.get('instructions', 'No instructions available'))
                print("\\n[Script is ready for Text-to-Speech conversion]")
            except Exception as e:
                print(f"Error reading script: {e}")
            print("=" * 50)
        
        return files_found >= 2  # Consider success if at least 2 key files are generated
        
    except subprocess.TimeoutExpired:
        print("❌ Pipeline timed out (took longer than 10 minutes)")
        print("This might indicate an issue with audio/video processing.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed with error: {e}")
        print("Error output:", e.stderr)
        return False


def main():
    print("=" * 60)
    print("🧘 LIVE MEDITATION PIPELINE 🧘")
    print("=" * 60)
    
    # Check if live capture dependencies are available
    try:
        import sounddevice
        import soundfile
        import cv2
    except ImportError as e:
        print("❌ Missing dependencies for live capture!")
        print(f"Error: {e}")
        print("\\nPlease install the required packages:")
        print("pip install -r live_capture_requirements.txt")
        return 1
    
    try:
        success = run_live_pipeline()
        
        if success:
            print("\\n🎉 Success! Your personalized meditation session is ready!")
            print("\\nNext steps:")
            print("1. Check 'meditation_recommendations_output.json' for recommendations")
            print("2. Check 'Core_engine/generated_meditation_script.json' for your custom script")
            print("3. Follow the meditation guidance provided")
        else:
            print("\\n❌ Pipeline failed. Please check the error messages above.")
            
    except KeyboardInterrupt:
        print("\\n\\n👋 Pipeline interrupted by user. Goodbye!")
        return 1
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
