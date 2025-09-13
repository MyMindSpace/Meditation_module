"""
Setup script for Live Capture dependencies
"""

import subprocess
import sys

def install_dependencies():
    """Install required packages for live audio/video capture"""
    
    packages = [
        "sounddevice>=0.4.6",
        "soundfile>=0.12.1", 
        "opencv-python>=4.8.0"
    ]
    
    print("🔧 Installing live capture dependencies...")
    print("This may take a few minutes...")
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("\\n🎉 All dependencies installed successfully!")
    print("\\nYou can now run:")
    print("python run_live_pipeline.py")
    
    return True

def test_imports():
    """Test if all required packages can be imported"""
    
    print("\\n🔍 Testing imports...")
    
    packages = [
        ("sounddevice", "sounddevice"),
        ("soundfile", "soundfile"), 
        ("opencv-python", "cv2")
    ]
    
    all_good = True
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name} - OK")
        except ImportError:
            print(f"❌ {package_name} - FAILED")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("=" * 50)
    print("🎙️ LIVE CAPTURE SETUP 🎥")
    print("=" * 50)
    
    if install_dependencies():
        if test_imports():
            print("\\n🚀 Setup complete! Ready for live capture.")
        else:
            print("\\n⚠️ Some packages failed to import. Please check the installation.")
    else:
        print("\\n❌ Setup failed. Please check your internet connection and try again.")
