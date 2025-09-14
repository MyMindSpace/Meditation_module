#!/usr/bin/env python3
"""
Test script for posture detection integration
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    
    try:
        import sounddevice as sd
        print("✅ SoundDevice imported successfully")
    except ImportError as e:
        print(f"❌ SoundDevice import failed: {e}")
        return False
    
    try:
        import soundfile as sf
        print("✅ SoundFile imported successfully")
    except ImportError as e:
        print(f"❌ SoundFile import failed: {e}")
        return False
    
    return True

def test_posture_detector():
    """Test if the posture detector can be initialized"""
    print("\n🧘 Testing posture detector initialization...")
    
    try:
        from working_posture_detection import WorkingPostureDetector, PostureConfig, PostureThresholds
        
        config = PostureConfig()
        thresholds = PostureThresholds()
        detector = WorkingPostureDetector(config, thresholds)
        
        print("✅ Posture detector initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Posture detector initialization failed: {e}")
        return False

def test_camera_access():
    """Test if camera can be accessed"""
    print("\n📹 Testing camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera access successful")
                cap.release()
                return True
            else:
                print("❌ Could not read from camera")
                cap.release()
                return False
        else:
            print("❌ Could not open camera")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_audio_devices():
    """Test if audio devices are available"""
    print("\n🎙️ Testing audio devices...")
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if input_devices:
            print(f"✅ Found {len(input_devices)} audio input devices")
            for i, device in enumerate(input_devices[:3]):  # Show first 3
                print(f"   {i}: {device['name']}")
            return True
        else:
            print("❌ No audio input devices found")
            return False
            
    except Exception as e:
        print(f"❌ Audio device test failed: {e}")
        return False

def test_file_structure():
    """Test if required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "working_posture_detection.py",
        "live_capture_with_posture.py",
        "run_live_pipeline.py",
        "live_capture.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🧘 POSTURE DETECTION INTEGRATION TEST")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_file_structure),
        ("Posture Detector Test", test_posture_detector),
        ("Camera Access Test", test_camera_access),
        ("Audio Devices Test", test_audio_devices)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The posture detection integration is ready to use.")
        print("\n🚀 You can now run:")
        print("   python run_live_pipeline.py")
        print("   Choose option 1 for enhanced meditation session with posture feedback")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the issues above.")
        print("\n💡 Common solutions:")
        print("   • Install missing packages: pip install opencv-python mediapipe sounddevice soundfile")
        print("   • Check camera permissions")
        print("   • Ensure audio devices are connected")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())
