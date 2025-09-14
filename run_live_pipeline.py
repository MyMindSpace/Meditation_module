"""
Live Input Pipeline for Meditation Module

This script captures live audio/video input from the user and then runs
the full meditation pipeline with the captured data.
"""

import json
import subprocess
import sys
import datetime
import os
from pathlib import Path

def run_live_pipeline():
    """Run the complete pipeline with live input capture and posture detection"""
    
    print("Welcome to the Enhanced Live Meditation Pipeline!")
    print("This will capture your audio/video with real-time posture feedback and generate personalized meditation recommendations.")
    print()
    
    # Get user preferences
    print("Choose capture mode:")
    print("1. Audio + Video with Posture Feedback (recommended)")
    print("2. Audio + Video without Posture Feedback") 
    print("3. Audio only") 
    print("4. Skip capture (use existing files)")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    # Get recording duration
    if choice in ['1', '2', '3']:
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
    posture_file = None
    
    if choice == "1":  # Audio + Video with Posture Feedback
        print(f"\\nStarting enhanced meditation session with posture feedback for {duration} seconds...")
        print("You'll see real-time posture analysis and feedback during the session.")
        try:
            result = subprocess.run([
                sys.executable, "live_capture_with_posture.py", 
                "--mode", "both",
                "--duration", str(duration),
                "--create-librispeech"
            ], capture_output=True, text=True, check=False)  # Don't raise exception on non-zero exit
            
            # Check if the session actually completed successfully by looking for success indicators
            if "Capture complete!" in result.stdout or "Posture data saved:" in result.stdout:
                print("Enhanced session completed successfully!")
                print(result.stdout)
            else:
                print(f"Session may have had issues. Exit code: {result.returncode}")
                print("Output:", result.stdout)
                if result.stderr:
                    print("Error output:", result.stderr)
                # Don't return False here - let it continue to see if files were created
                
        except Exception as e:
            print(f"Error during enhanced capture: {e}")
            return False
            
    elif choice == "2":  # Audio + Video without Posture Feedback
        print(f"\\nStarting audio + video capture for {duration} seconds...")
        try:
            result = subprocess.run([
                sys.executable, "live_capture_with_posture.py", 
                "--mode", "both",
                "--duration", str(duration),
                "--create-librispeech",
                "--no-posture"
            ], capture_output=True, text=True, check=False)
            
            if "Capture complete!" in result.stdout or "Video saved:" in result.stdout:
                print("Audio and video captured successfully!")
                print(result.stdout)
            else:
                print(f"Session may have had issues. Exit code: {result.returncode}")
                print("Output:", result.stdout)
                if result.stderr:
                    print("Error output:", result.stderr)
                    
        except Exception as e:
            print(f"Error during capture: {e}")
            return False
            
    elif choice == "3":  # Audio only
        print(f"\\nStarting audio capture for {duration} seconds...")
        try:
            result = subprocess.run([
                sys.executable, "live_capture.py", 
                "--mode", "audio",
                "--duration", str(duration),
                "--create-librispeech"
            ], capture_output=True, text=True, check=False)
            
            if "Capture complete!" in result.stdout or "Audio saved:" in result.stdout:
                print("Audio captured successfully!")
                print(result.stdout)
            else:
                print(f"Session may have had issues. Exit code: {result.returncode}")
                print("Output:", result.stdout)
                if result.stderr:
                    print("Error output:", result.stderr)
                    
        except Exception as e:
            print(f"Error during capture: {e}")
            return False
    
    # Check if capture was successful by looking for created files
    if choice in ['1', '2', '3']:
        import glob
        audio_files = glob.glob("preprocess_input/live_audio_*.wav")
        video_files = glob.glob("preprocess_input/live_video_*.mp4")
        
        if choice in ['1', '2'] and not video_files:
            print("Warning: No video files found. The capture may have failed.")
        if choice in ['1', '2', '3'] and not audio_files:
            print("Warning: No audio files found. The capture may have failed.")
        
        if choice == '1':
            posture_files = glob.glob("preprocess_input/posture_data_*.json")
            if not posture_files:
                print("Warning: No posture data files found.")
            else:
                print(f"Found {len(posture_files)} posture data files")
                
                # Convert posture data to scores format for pipeline
                try:
                    result = subprocess.run([
                        sys.executable, "convert_posture_data.py", "--all"
                    ], capture_output=True, text=True, check=True)
                    print("Posture data converted to posture_scores.json format")
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Could not convert posture data: {e}")
                except Exception as e:
                    print(f"Warning: Error converting posture data: {e}")
    
    # Get diary entry
    print("\\nPlease provide your diary entry:")
    print("Describe your current mood, thoughts, or meditation goals...")
    diary_text = input("> ")
    
    if not diary_text.strip():
        diary_text = "I want to practice meditation and improve my well-being."
    
    # Check for posture data from enhanced session
    posture_summary = None
    if choice == "1":  # Enhanced session with posture feedback
        # Look for the most recent posture data file
        import glob
        posture_files = glob.glob("preprocess_input/posture_data_*.json")
        if posture_files:
            latest_posture_file = max(posture_files, key=os.path.getctime)
            try:
                with open(latest_posture_file, 'r', encoding='utf-8') as f:
                    posture_data = json.load(f)
                    posture_summary = {
                        "average_posture_score": posture_data.get("posture_statistics", {}).get("average_score", 0.0),
                        "detection_rate": posture_data.get("posture_statistics", {}).get("detection_rate", 0.0),
                        "session_quality": "excellent" if posture_data.get("posture_statistics", {}).get("average_score", 0.0) >= 0.85 else
                                         "good" if posture_data.get("posture_statistics", {}).get("average_score", 0.0) >= 0.70 else
                                         "fair" if posture_data.get("posture_statistics", {}).get("average_score", 0.0) >= 0.55 else "poor"
                    }
                print(f"Posture analysis data loaded from session")
            except Exception as e:
                print(f"Could not load posture data: {e}")
    
    # Save diary entry with posture context
    diary_entry = {
        "user_id": "live_user",
        "timestamp": datetime.datetime.now().isoformat(),
        "entry": diary_text,
        "session_type": "enhanced_with_posture" if choice == "1" else "standard",
        "posture_analysis": posture_summary
    }
    
    diary_path = "preprocess_input/diary_entry.json"
    with open(diary_path, "w", encoding="utf-8") as f:
        json.dump([diary_entry], f, indent=2, ensure_ascii=False)
    
    print(f"Diary entry saved to {diary_path}")
    
    # Show posture summary if available
    if posture_summary:
        print(f"\\nYour Posture Session Summary:")
        print(f"   - Average Posture Score: {posture_summary['average_posture_score']:.1%}")
        print(f"   - Detection Rate: {posture_summary['detection_rate']:.1%}")
        print(f"   - Overall Quality: {posture_summary['session_quality'].title()}")
        print("   - This data will be used to personalize your meditation recommendations")
    
    # Run the full pipeline
    print("\\nRunning the meditation pipeline...")
    print("This may take a few minutes as it processes audio, video, and generates your script...")
    try:
        # Run pipeline with empty input (since we already saved diary entry)
        result = subprocess.run([
            sys.executable, "run_full_pipeline.py"
        ], input="\\n", capture_output=True, text=True, timeout=600)  # 10 minute timeout for processing
        
        if result.returncode == 0:
            print("Pipeline completed successfully!")
        else:
            print("Pipeline completed with some warnings")
            
        print("\\nPipeline Output:")
        print(result.stdout)
        
        if result.stderr:
            print("\\nPipeline Warnings/Errors:")
            print(result.stderr)
        
        # Check for output files
        output_files = [
            ("meditation_recommendations_output.json", "Meditation Recommendations"),
            ("Core_engine/generated_meditation_script.json", "Generated Meditation Script"),
            ("preprocess_output/fused_decision.json", "Fusion Decision"),
            ("encoder_output/audio_encoded.json", "Audio Encoding"),
            ("encoder_output/vision_encoded.json", "Video Encoding")
        ]
        
        # Add posture data file if available
        if choice == "1":  # Enhanced session
            posture_files = glob.glob("preprocess_input/posture_data_*.json")
            if posture_files:
                latest_posture_file = max(posture_files, key=os.path.getctime)
                output_files.append((latest_posture_file, "Posture Analysis Data"))
        
        print("\\nGenerated Files:")
        files_found = 0
        for file_path, description in output_files:
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                print(f"✓ {description}: {file_path} ({file_size} bytes)")
                files_found += 1
            else:
                print(f"✗ {description}: {file_path} (not found)")
        
        # Show meditation script if available
        script_path = "Core_engine/generated_meditation_script.json"
        if Path(script_path).exists():
            print("\\nYour Personalized Meditation Script:")
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
        print("Pipeline timed out (took longer than 10 minutes)")
        print("This might indicate an issue with audio/video processing.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed with error: {e}")
        print("Error output:", e.stderr)
        return False


def main():
    print("=" * 60)
    print("LIVE MEDITATION PIPELINE")
    print("=" * 60)
    
    # Check if live capture dependencies are available
    try:
        import sounddevice
        import soundfile
        import cv2
    except ImportError as e:
        print("Missing dependencies for live capture!")
        print(f"Error: {e}")
        print("\\nPlease install the required packages:")
        print("pip install -r live_capture_requirements.txt")
        return 1
    
    try:
        success = run_live_pipeline()
        
        if success:
            print("\\nSuccess! Your personalized meditation session is ready!")
            print("\\nNext steps:")
            print("1. Check 'meditation_recommendations_output.json' for recommendations")
            print("2. Check 'Core_engine/generated_meditation_script.json' for your custom script")
            print("3. Follow the meditation guidance provided")
        else:
            print("\\nPipeline failed. Please check the error messages above.")
            
    except KeyboardInterrupt:
        print("\\n\\nPipeline interrupted by user. Goodbye!")
        return 1
    except Exception as e:
        print(f"\\nUnexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
