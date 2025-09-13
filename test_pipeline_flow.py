"""
Quick test to verify the complete live pipeline flow
"""

import json
import subprocess
import sys
from pathlib import Path

def test_live_pipeline():
    """Test the complete flow with minimal setup"""
    
    print("🧪 Testing Live Pipeline Flow")
    print("=" * 40)
    
    # Create a test diary entry
    diary_path = "preprocess_input/diary_entry.json"
    test_diary = [{
        "user_id": "test_user",
        "timestamp": "2025-09-14T10:00:00",
        "entry": "I'm feeling anxious and need grounding meditation"
    }]
    
    with open(diary_path, 'w', encoding='utf-8') as f:
        json.dump(test_diary, f, indent=2)
    print(f"✅ Created test diary: {diary_path}")
    
    # Check if we have existing audio/video files
    audio_paths = list(Path("preprocess_input").glob("LibriSpeech*"))
    video_paths = list(Path("preprocess_input").glob("*.mp4"))
    
    print(f"📁 Found audio paths: {len(audio_paths)}")
    print(f"📁 Found video paths: {len(video_paths)}")
    
    # Run the pipeline
    print("\\n🔄 Running pipeline...")
    try:
        result = subprocess.run([
            sys.executable, "run_full_pipeline.py"
        ], input="\\n", capture_output=True, text=True, timeout=300)
        
        print("📋 Pipeline stdout:")
        print(result.stdout)
        
        if result.stderr:
            print("\\n⚠️ Pipeline stderr:")
            print(result.stderr)
        
        # Check outputs
        outputs = [
            "meditation_recommendations_output.json",
            "Core_engine/generated_meditation_script.json"
        ]
        
        print("\\n📂 Checking outputs:")
        for output in outputs:
            if Path(output).exists():
                size = Path(output).stat().st_size
                print(f"✅ {output} ({size} bytes)")
                
                # Show content preview
                if output.endswith('.json'):
                    try:
                        with open(output, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if 'meditation_type' in data:
                            print(f"   → Meditation Type: {data['meditation_type']}")
                        if 'recommendations' in data:
                            print(f"   → Recommendations: {len(data['recommendations'])}")
                    except Exception as e:
                        print(f"   → Error reading: {e}")
            else:
                print(f"❌ {output} (missing)")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_live_pipeline()
    print(f"\\n{'✅ SUCCESS' if success else '❌ FAILED'}")
