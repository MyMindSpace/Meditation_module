# live_meditation_session.py
import time
import numpy as np
from live_posture_detection import LiveCameraPostureDetector, PostureCalibrator
from session_manager import SessionManager, SessionConfig, RealTimeData

def run_live_posture_session():
    """Complete live posture detection setup"""
    # Initialize live detector
    live_detector = LiveCameraPostureDetector(camera_id=0, target_fps=15)
    
    # Start detection
    live_detector.start_live_detection()
    
    # Optional: Calibrate for user
    print("Starting calibration...")
    calibrator = PostureCalibrator(live_detector)
    calibrator.start_calibration(duration_seconds=30)
    
    # Create session with live integration
    session_config = SessionConfig(
        session_id=f"live_session_{int(time.time())}",
        user_id="live_user",
        meditation_type="mindfulness",
        planned_duration=600,  # 10 minutes
        enable_posture_detection=True,
        posture_correction_interval=20.0
    )
    
    manager = SessionManager()
    session_id = manager.create_session(session_config)
    manager.start_session(session_id)
    
    try:
        # Live processing loop
        while True:
            posture_result = live_detector.get_latest_posture()
            
            if posture_result:
                # Apply calibration
                calibrated_score = calibrator.get_calibrated_score(
                    posture_result['posture_score']
                )
                
                # Create real-time data
                rt_data = RealTimeData(
                    timestamp=posture_result['timestamp'],
                    video_frame=None,  # Frame not needed after processing
                    posture_score=calibrated_score,
                    engagement_level=0.8
                )
                
                # Process in session
                manager.process_real_time_data(session_id, rt_data)
                
                # Display feedback
                if calibrated_score < 0.6:
                    print(f"⚠️  Posture Score: {calibrated_score:.2f} - Please adjust your posture")
                else:
                    print(f"✅ Good posture: {calibrated_score:.2f}")
                    
            time.sleep(0.1)  # 10Hz update rate for user feedback
            
    except KeyboardInterrupt:
        print("Stopping session...")
    finally:
        live_detector.stop_detection()
        manager.end_session(session_id, user_rating=4.0)

if __name__ == "__main__":
    run_live_posture_session()
