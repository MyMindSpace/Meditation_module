#!/usr/bin/env python3
"""
Working Enhanced Posture Detection System
Combines the best features with proper error handling
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

@dataclass
class PostureConfig:
    """Configuration for posture detection"""
    show_skeleton: bool = True
    show_score: bool = True
    show_corrections: bool = True
    show_detailed_metrics: bool = True
    smoothing_window: int = 10
    confidence_threshold: float = 0.7

@dataclass
class PostureThresholds:
    """Scientific posture analysis thresholds"""
    excellent_threshold: float = 0.85
    good_threshold: float = 0.70
    fair_threshold: float = 0.55
    poor_threshold: float = 0.40
    max_shoulder_slope: float = 0.05  # 5% of frame height
    max_head_forward: float = 0.08    # 8% of frame width

class WorkingPostureDetector:
    """Working enhanced posture detector with proper error handling"""
    
    def __init__(self, config: PostureConfig = None, thresholds: PostureThresholds = None):
        self.config = config or PostureConfig()
        self.thresholds = thresholds or PostureThresholds()
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Data structures
        self.score_history = deque(maxlen=self.config.smoothing_window)
        self.metrics_history = deque(maxlen=self.config.smoothing_window)
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = deque(maxlen=30)
        
        # Feedback messages
        self.feedback_messages = {
            'excellent': "Excellent Posture! 🌟",
            'good': "Good Posture ✓",
            'fair': "Fair Posture - Minor Adjustments",
            'poor': "Poor Posture - Major Corrections Needed ❌",
            'no_detection': "Position yourself in view"
        }
        
        print("✅ Working Posture Detector initialized successfully!")
    
    def calculate_posture_metrics(self, landmarks) -> Dict[str, float]:
        """Calculate comprehensive posture metrics"""
        metrics = {}
        
        try:
            # Extract key landmarks
            nose = landmarks.landmark[0]
            left_shoulder = landmarks.landmark[11]
            right_shoulder = landmarks.landmark[12]
            
            # Check if hips are available
            hips_available = False
            if (len(landmarks.landmark) > 24 and 
                landmarks.landmark[23].visibility > 0.3 and 
                landmarks.landmark[24].visibility > 0.3):
                left_hip = landmarks.landmark[23]
                right_hip = landmarks.landmark[24]
                hips_available = True
            else:
                # Estimate hip positions
                shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
                estimated_hip_y = shoulder_center_y + 0.2  # 20% of frame height below shoulders
                left_hip = type('obj', (object,), {'x': left_shoulder.x, 'y': estimated_hip_y, 'visibility': 0.5})()
                right_hip = type('obj', (object,), {'x': right_shoulder.x, 'y': estimated_hip_y, 'visibility': 0.5})()
            
            # 1. Shoulder Alignment Analysis
            shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            
            metrics['shoulder_slope'] = shoulder_slope
            metrics['shoulder_alignment_score'] = max(0, 1 - (shoulder_slope / self.thresholds.max_shoulder_slope))
            
            # 2. Head Position Analysis
            head_forward_distance = abs(nose.x - shoulder_center_x)
            head_vertical_offset = nose.y - shoulder_center_y
            
            metrics['head_forward_distance'] = head_forward_distance
            metrics['head_vertical_offset'] = head_vertical_offset
            metrics['head_position_score'] = max(0, 1 - (head_forward_distance / self.thresholds.max_head_forward))
            
            # 3. Spinal Alignment Analysis
            spine_center_top = shoulder_center_x
            spine_center_bottom = (left_hip.x + right_hip.x) / 2
            spine_deviation = abs(spine_center_top - spine_center_bottom)
            
            metrics['spine_deviation'] = spine_deviation
            metrics['spine_alignment_score'] = max(0, 1 - (spine_deviation / 0.1))  # 10% of frame width
            
            # 4. Hip Alignment Analysis
            if hips_available:
                hip_slope = abs(left_hip.y - right_hip.y)
                metrics['hip_slope'] = hip_slope
                metrics['hip_alignment_score'] = max(0, 1 - (hip_slope / self.thresholds.max_shoulder_slope))
            else:
                metrics['hip_slope'] = shoulder_slope
                metrics['hip_alignment_score'] = metrics['shoulder_alignment_score'] * 0.8
            
            # 5. Overall Symmetry Analysis
            left_side_center = (left_shoulder.x + left_hip.x) / 2
            right_side_center = (right_shoulder.x + right_hip.x) / 2
            body_symmetry = abs(left_side_center - right_side_center)
            
            metrics['body_symmetry'] = body_symmetry
            metrics['symmetry_score'] = max(0, 1 - (body_symmetry / 0.15))  # 15% of frame width
            
            # 6. Angle Calculations
            shoulder_angle = math.degrees(math.atan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            ))
            metrics['shoulder_angle'] = abs(shoulder_angle)
            metrics['shoulder_angle_score'] = max(0, 1 - (abs(shoulder_angle) / 90))
            
            # 7. Confidence metrics
            avg_visibility = (nose.visibility + left_shoulder.visibility + right_shoulder.visibility) / 3
            metrics['detection_confidence'] = avg_visibility
            metrics['confidence_weight'] = min(1.0, avg_visibility / 0.8)
            
            # 8. Detection mode
            metrics['hips_available'] = hips_available
            metrics['detection_mode'] = 'upper_body' if not hips_available else 'full_body'
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Return default metrics
            metrics = {
                'shoulder_alignment_score': 0.5,
                'head_position_score': 0.5,
                'spine_alignment_score': 0.5,
                'hip_alignment_score': 0.5,
                'symmetry_score': 0.5,
                'shoulder_angle_score': 0.5,
                'detection_confidence': 0.5,
                'confidence_weight': 0.5,
                'hips_available': False,
                'detection_mode': 'upper_body'
            }
        
        return metrics
    
    def calculate_posture_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall posture score from metrics"""
        try:
            # Define weights for different aspects
            weights = {
                'shoulder_alignment_score': 0.30,
                'head_position_score': 0.25,
                'spine_alignment_score': 0.20,
                'hip_alignment_score': 0.15,
                'symmetry_score': 0.10
            }
            
            # Calculate weighted score
            weighted_score = sum(metrics.get(key, 0.5) * weight for key, weight in weights.items())
            
            # Apply confidence weighting
            confidence_weight = metrics.get('confidence_weight', 0.5)
            final_score = weighted_score * confidence_weight
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            print(f"Error calculating posture score: {e}")
            return 0.5
    
    def get_feedback_level(self, score: float) -> str:
        """Get feedback level based on score"""
        if score >= self.thresholds.excellent_threshold:
            return 'excellent'
        elif score >= self.thresholds.good_threshold:
            return 'good'
        elif score >= self.thresholds.fair_threshold:
            return 'fair'
        else:
            return 'poor'
    
    def get_feedback_color(self, level: str) -> Tuple[int, int, int]:
        """Get color based on feedback level"""
        colors = {
            'excellent': (0, 255, 255),  # Cyan
            'good': (0, 255, 0),         # Green
            'fair': (0, 165, 255),       # Orange
            'poor': (0, 0, 255),         # Red
            'no_detection': (100, 100, 100)
        }
        return colors.get(level, (255, 255, 255))
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process frame and return annotated frame with feedback"""
        start_time = time.time()
        self.frame_count += 1
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        annotated_frame = frame.copy()
        feedback_data = {}
        
        if results.pose_landmarks:
            # Draw pose skeleton
            if self.config.show_skeleton:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=2
                    )
                )
            
            # Calculate metrics and score
            metrics = self.calculate_posture_metrics(results.pose_landmarks)
            posture_score = self.calculate_posture_score(metrics)
            
            # Update history for smoothing
            self.score_history.append(posture_score)
            self.metrics_history.append(metrics)
            
            # Calculate smoothed score
            smoothed_score = np.mean(list(self.score_history)) if self.score_history else posture_score
            
            # Get feedback
            feedback_level = self.get_feedback_level(smoothed_score)
            message = self.feedback_messages.get(feedback_level, "")
            color = self.get_feedback_color(feedback_level)
            
            # Draw feedback
            self.draw_feedback(annotated_frame, smoothed_score, message, color, metrics)
            
            feedback_data = {
                'posture_score': smoothed_score,
                'raw_score': posture_score,
                'feedback_level': feedback_level,
                'message': message,
                'landmarks_detected': True,
                'metrics': metrics
            }
        else:
            # No pose detected
            self.draw_no_pose_message(annotated_frame)
            feedback_data = {
                'posture_score': 0.0,
                'feedback_level': 'no_detection',
                'message': 'Position yourself in view',
                'landmarks_detected': False
            }
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return annotated_frame, feedback_data
    
    def draw_feedback(self, frame: np.ndarray, score: float, message: str, color: Tuple[int, int, int], metrics: Dict[str, float]):
        """Draw comprehensive feedback on frame"""
        h, w, _ = frame.shape
        
        # Draw posture score bar
        if self.config.show_score:
            self.draw_score_bar(frame, score, color)
        
        # Draw feedback message
        self.draw_feedback_message(frame, message, color)
        
        # Draw detailed metrics
        if self.config.show_detailed_metrics:
            self.draw_detailed_metrics(frame, metrics)
        
        # Draw detection mode indicator
        self.draw_detection_mode(frame, metrics)
        
        # Draw performance info
        self.draw_performance_info(frame)
    
    def draw_score_bar(self, frame: np.ndarray, score: float, color: Tuple[int, int, int]):
        """Draw posture score bar"""
        h, w, _ = frame.shape
        
        bar_width = 250
        bar_height = 25
        bar_x = w - bar_width - 20
        bar_y = 30
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (40, 40, 40), -1)
        
        # Score fill
        fill_width = int(bar_width * score)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
        
        # Score text
        score_text = f"Posture: {score:.1%}"
        cv2.putText(frame, score_text, (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def draw_feedback_message(self, frame: np.ndarray, message: str, color: Tuple[int, int, int]):
        """Draw feedback message"""
        h, w, _ = frame.shape
        
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 80
        
        # Message background
        padding = 15
        cv2.rectangle(frame, (text_x - padding, text_y - 30), (text_x + text_size[0] + padding, text_y + 10), color, -1)
        cv2.rectangle(frame, (text_x - padding, text_y - 30), (text_x + text_size[0] + padding, text_y + 10), (255, 255, 255), 2)
        
        # Message text
        cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def draw_detailed_metrics(self, frame: np.ndarray, metrics: Dict[str, float]):
        """Draw detailed posture metrics"""
        h, w, _ = frame.shape
        
        # Position in top-left
        x_start = 20
        y_start = 80
        
        # Background box
        box_width = 280
        box_height = 120
        cv2.rectangle(frame, (x_start - 5, y_start - 25), (x_start + box_width, y_start + box_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (x_start - 5, y_start - 25), (x_start + box_width, y_start + box_height), (100, 100, 100), 1)
        
        # Title
        cv2.putText(frame, "Detailed Analysis", (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Key metrics
        key_metrics = [
            ('shoulder_alignment_score', 'Shoulders'),
            ('head_position_score', 'Head'),
            ('spine_alignment_score', 'Spine'),
            ('hip_alignment_score', 'Hips')
        ]
        
        y_offset = y_start + 25
        for metric_key, display_name in key_metrics:
            if metric_key in metrics:
                score = metrics[metric_key]
                color = self.get_feedback_color(self.get_feedback_level(score))
                
                # Metric name
                cv2.putText(frame, display_name, (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Score
                score_text = f"{score:.2f}"
                cv2.putText(frame, score_text, (x_start + 120, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Mini progress bar
                bar_start = (x_start + 160, y_offset - 8)
                bar_end = (x_start + 250, y_offset - 2)
                self.draw_mini_progress_bar(frame, bar_start, bar_end, score, color)
                
                y_offset += 20
    
    def draw_mini_progress_bar(self, frame: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], progress: float, color: Tuple[int, int, int]):
        """Draw a mini progress bar"""
        x1, y1 = start
        x2, y2 = end
        
        # Background
        cv2.rectangle(frame, start, end, (30, 30, 30), -1)
        
        # Progress fill
        progress_width = int((x2 - x1) * progress)
        if progress_width > 0:
            cv2.rectangle(frame, start, (x1 + progress_width, y2), color, -1)
        
        # Border
        cv2.rectangle(frame, start, end, (100, 100, 100), 1)
    
    def draw_detection_mode(self, frame: np.ndarray, metrics: Dict[str, float]):
        """Draw detection mode indicator"""
        h, w, _ = frame.shape
        
        # Position in top-left
        mode_x, mode_y = 20, 20
        
        # Get detection mode
        detection_mode = metrics.get('detection_mode', 'unknown')
        hips_available = metrics.get('hips_available', False)
        
        # Color based on mode
        if detection_mode == 'full_body':
            color = (0, 255, 0)  # Green
            text = "FULL BODY"
        elif detection_mode == 'upper_body':
            color = (0, 255, 255)  # Yellow
            text = "UPPER BODY"
        else:
            color = (0, 165, 255)  # Orange
            text = "UNKNOWN"
        
        # Draw background
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (mode_x - 5, mode_y - 20), (mode_x + text_size[0] + 5, mode_y + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (mode_x - 5, mode_y - 20), (mode_x + text_size[0] + 5, mode_y + 5), color, 2)
        
        # Draw text
        cv2.putText(frame, text, (mode_x, mode_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add hips status
        hips_text = "Hips: ✓" if hips_available else "Hips: Est."
        hips_color = (0, 255, 0) if hips_available else (0, 255, 255)
        cv2.putText(frame, hips_text, (mode_x, mode_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, hips_color, 1)
    
    def draw_performance_info(self, frame: np.ndarray):
        """Draw performance information"""
        h, w, _ = frame.shape
        
        # FPS indicator
        if self.processing_times:
            avg_processing_time = np.mean(list(self.processing_times))
            fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Frame count
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, frame_text, (w - 100, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def draw_no_pose_message(self, frame: np.ndarray):
        """Draw no pose detected message"""
        h, w, _ = frame.shape
        message = "Please position yourself in view"
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        
        cv2.rectangle(frame, (text_x - 10, text_y - 25), (text_x + text_size[0] + 10, text_y + 5), (0, 0, 200), -1)
        cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def main():
    """Main function to run the working posture detection"""
    print("🚀 Working Enhanced Posture Detection System")
    print("📊 Features:")
    print("   • Scientific posture analysis with weighted metrics")
    print("   • Upper body focused detection (works when sitting)")
    print("   • Real-time stability tracking")
    print("   • Detailed visual feedback")
    print("   • Performance monitoring")
    print("   • Robust error handling")
    print("\n🎯 Press 'q' to quit, 's' to screenshot")
    
    # Initialize detector
    config = PostureConfig(
        show_skeleton=True,
        show_score=True,
        show_corrections=True,
        show_detailed_metrics=True,
        smoothing_window=10,
        confidence_threshold=0.7
    )
    
    thresholds = PostureThresholds(
        excellent_threshold=0.85,
        good_threshold=0.70,
        fair_threshold=0.55,
        poor_threshold=0.40
    )
    
    detector = WorkingPostureDetector(config, thresholds)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            annotated_frame, feedback_data = detector.process_frame(frame)
            
            # Add session info
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            h, w, _ = annotated_frame.shape
            info_text = f"Session: {int(elapsed_time//60):02d}:{int(elapsed_time%60):02d} | FPS: {fps:.1f}"
            cv2.putText(annotated_frame, info_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show detection status
            if feedback_data['landmarks_detected']:
                status_text = f"Detection: Active | Score: {feedback_data['posture_score']:.2f}"
                cv2.putText(annotated_frame, status_text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(annotated_frame, "Detection: Inactive", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Display frame
            cv2.imshow('Working Enhanced Posture Detection', annotated_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quitting session...")
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"working_posture_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Session interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ System cleanup completed")

if __name__ == "__main__":
    main()
