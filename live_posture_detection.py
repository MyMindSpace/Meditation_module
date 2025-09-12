# live_posture_detection.py
import cv2
import threading
from queue import Queue
import time
import numpy as np
from Encoders.vision_encoder import VisionEncoder
from Core_engine.Posture_detector import PostureDetector

class LiveCameraPostureDetector:
    def __init__(self, camera_id=0, target_fps=30):
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=10)
        self.is_running = False
        
        # Initialize components
        self.vision_encoder = VisionEncoder(use_torch=False)  # Faster for real-time
        self.posture_detector = PostureDetector(use_lstm=False)  # Use heuristics for speed
        
        # Camera setup
        self.cap = None
        self.camera_thread = None
        self.processing_thread = None
        
    def start_live_detection(self):
        """Start live posture detection"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
        # Set camera properties for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        self.is_running = True
        
        # Start threads
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        
        self.camera_thread.start()
        self.processing_thread.start()
        
    def _camera_loop(self):
        """Continuous camera capture"""
        frame_time = 1.0 / self.target_fps
        
        while self.is_running:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if ret:
                # Add frame to queue (non-blocking)
                try:
                    self.frame_queue.put(frame, block=False)
                except:
                    # Drop frame if queue is full
                    pass
                    
            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def _processing_loop(self):
        """Process frames for posture detection"""
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                # Process frame through vision encoder
                vision_result = self.vision_encoder.process_frame(frame)
                
                if vision_result['pose_detected']:
                    # Create proper record for posture detector
                    record = {
                        "embeddings": {"pose": vision_result['pose_embedding']}
                    }
                    
                    # Get posture analysis
                    posture_result = self.posture_detector.process_record(record)
                    
                    # Add timestamp and frame info
                    result = {
                        'timestamp': time.time(),
                        'posture_score': posture_result['posture_score'],
                        'keypoints': vision_result['pose_keypoints'],
                        'debug_info': posture_result.get('debug', {}),
                        'frame_shape': frame.shape
                    }
                    
                    # Store result
                    try:
                        self.result_queue.put(result, block=False)
                    except:
                        # Remove oldest result if queue is full
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put(result, block=False)
                        except:
                            pass
                            
            except Exception as e:
                print(f"Processing error: {e}")
                continue
                
    def get_latest_posture(self):
        """Get the most recent posture analysis"""
        try:
            return self.result_queue.get_nowait()
        except:
            return None
            
    def stop_detection(self):
        """Stop live detection"""
        self.is_running = False
        
        if self.camera_thread:
            self.camera_thread.join(timeout=2.0)
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            
        if self.cap:
            self.cap.release()

class PostureCalibrator:
    def __init__(self, detector):
        self.detector = detector
        self.calibration_frames = []
        self.baseline_scores = {}
        
    def start_calibration(self, duration_seconds=30):
        """Collect baseline posture data"""
        print("Starting posture calibration. Please sit in your preferred meditation position.")
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            # Get frame from camera
            result = self.detector.get_latest_posture()
            if result:
                self.calibration_frames.append(result)
                
        self._compute_baseline()
        
    def _compute_baseline(self):
        """Compute user's baseline posture metrics"""
        if not self.calibration_frames:
            return
            
        scores = [frame['posture_score'] for frame in self.calibration_frames]
        
        self.baseline_scores = {
            'mean_posture': np.mean(scores),
            'std_posture': np.std(scores),
            'good_threshold': np.percentile(scores, 75),
            'poor_threshold': np.percentile(scores, 25)
        }
        
    def get_calibrated_score(self, raw_score):
        """Convert raw score to calibrated score"""
        if not self.baseline_scores:
            return raw_score
            
        baseline = self.baseline_scores['mean_posture']
        good_threshold = self.baseline_scores['good_threshold']
        poor_threshold = self.baseline_scores['poor_threshold']
        
        if raw_score >= good_threshold:
            return min(1.0, 0.8 + 0.2 * (raw_score - good_threshold) / (1.0 - good_threshold))
        elif raw_score <= poor_threshold:
            return max(0.0, 0.2 * raw_score / poor_threshold)
        else:
            return 0.2 + 0.6 * (raw_score - poor_threshold) / (good_threshold - poor_threshold)
