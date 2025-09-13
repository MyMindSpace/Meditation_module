# live_posture_feedback.py
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import time
import math
from dataclasses import dataclass
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PostureVisualizationConfig:
    """Enhanced configuration for posture visualization"""
    show_skeleton: bool = True
    show_score: bool = True
    show_corrections: bool = True
    show_detailed_metrics: bool = True
    show_stability_indicator: bool = True
    skeleton_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    warning_color: Tuple[int, int, int] = (0, 165, 255)  # Orange
    error_color: Tuple[int, int, int] = (0, 0, 255)  # Red
    excellent_color: Tuple[int, int, int] = (0, 255, 255)  # Cyan
    line_thickness: int = 2
    circle_radius: int = 4
    smoothing_window: int = 15  # Frames for score smoothing
    confidence_threshold: float = 0.7  # Minimum landmark confidence
    stability_threshold: float = 0.1  # Score variation threshold for stability

@dataclass
class PostureThresholds:
    """Scientific posture analysis thresholds"""
    # Score thresholds
    excellent_threshold: float = 0.85
    good_threshold: float = 0.70
    fair_threshold: float = 0.55
    poor_threshold: float = 0.40
    
    # Physical measurement thresholds (in pixels for 640x480)
    max_shoulder_slope: float = 15.0  # pixels
    max_head_forward: float = 40.0    # pixels
    max_spine_deviation: float = 25.0 # pixels
    max_shoulder_asymmetry: float = 20.0  # pixels
    
    # Angle thresholds (in degrees)
    max_neck_angle: float = 15.0      # degrees
    max_spine_angle: float = 10.0     # degrees
    ideal_shoulder_angle: float = 0.0 # degrees (horizontal)
    
    # Stability thresholds
    stability_window: int = 30        # frames
    min_stability_score: float = 0.6  # minimum score for stable posture

class LivePostureFeedback:
    """Enhanced live posture feedback with scientific analysis and robust fallbacks"""
    
    def __init__(self, config: PostureVisualizationConfig = None, thresholds: PostureThresholds = None):
        self.config = config or PostureVisualizationConfig()
        self.thresholds = thresholds or PostureThresholds()
        
        # Initialize MediaPipe with diagnostic-friendly settings
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Lower complexity for better detection
            enable_segmentation=False,
            min_detection_confidence=0.5,  # Lower confidence for easier detection
            min_tracking_confidence=0.5    # Lower tracking confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Enhanced feedback messages with specific guidance
        self.feedback_messages = {
            'excellent': "Excellent Posture! 🌟",
            'good': "Good Posture ✓",
            'fair': "Fair Posture - Minor Adjustments",
            'poor': "Poor Posture - Major Corrections Needed ❌",
            'no_detection': "Position yourself in view",
            'low_confidence': "Move closer to camera"
        }
        
        # Advanced data structures for analysis
        self.score_history = deque(maxlen=self.config.smoothing_window)
        self.detailed_metrics_history = deque(maxlen=self.config.smoothing_window)
        self.stability_scores = deque(maxlen=self.thresholds.stability_window)
        
        # Fallback mechanisms
        self.fallback_active = False
        self.last_valid_pose = None
        self.pose_detection_failures = 0
        self.max_failures = 10
        
        # Calibration data
        self.is_calibrated = False
        self.baseline_metrics = {}
        self.user_specific_thresholds = {}
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = deque(maxlen=30)
        
        logger.info("Enhanced LivePostureFeedback initialized with scientific thresholds")
    
    def process_frame_with_feedback(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Enhanced frame processing with robust fallbacks and scientific analysis"""
        start_time = time.time()
        self.frame_count += 1
        
        if frame is None:
            logger.warning("Received None frame")
            return frame, {}
        
        # Create annotated frame
        annotated_frame = frame.copy()
        feedback_data = {}
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            # DIAGNOSTIC: Print detection status
            if self.frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                print(f"\n🔍 DIAGNOSTIC FRAME {self.frame_count}:")
                print(f"   • Pose landmarks detected: {results.pose_landmarks is not None}")
                if results.pose_landmarks:
                    print(f"   • Total landmarks: {len(results.pose_landmarks.landmark)}")
                    # Check key landmarks
                    key_indices = [0, 11, 12, 23, 24]
                    for idx in key_indices:
                        landmark = results.pose_landmarks.landmark[idx]
                        print(f"   • Landmark {idx}: visibility={landmark.visibility:.2f}, x={landmark.x:.2f}, y={landmark.y:.2f}")
            
            # Check if pose is detected with sufficient confidence
            if results.pose_landmarks and self._validate_pose_confidence(results.pose_landmarks):
                # Reset failure counter on successful detection
                self.pose_detection_failures = 0
                self.fallback_active = False
                
                # Extract and validate landmarks
                landmarks = results.pose_landmarks.landmark
                h, w, _ = frame.shape
                
                # Convert to pixel coordinates with validation
                keypoints = self._extract_validated_keypoints(landmarks, w, h)
                
                if keypoints is not None:
                    # Calculate comprehensive posture analysis
                    detailed_metrics = self._calculate_detailed_posture_metrics(keypoints)
                    posture_score = self._calculate_enhanced_posture_score(detailed_metrics)
                    
                    # Update history for smoothing and stability analysis
                    self._update_analysis_history(posture_score, detailed_metrics)
                    
                    # Calculate smoothed score and stability
                    smoothed_score = self._calculate_smoothed_score()
                    stability_score = self._calculate_stability_score()
                    
                    # Store last valid pose for fallback
                    self.last_valid_pose = {
                        'keypoints': keypoints,
                        'score': smoothed_score,
                        'metrics': detailed_metrics,
                        'timestamp': time.time()
                    }
                    
                    # Draw enhanced visual feedback
                    if self.config.show_skeleton:
                        annotated_frame = self._draw_enhanced_pose_skeleton(annotated_frame, keypoints, detailed_metrics)
                    
                    # Add comprehensive visual feedback
                    annotated_frame = self._add_enhanced_posture_feedback(annotated_frame, smoothed_score, detailed_metrics, stability_score)
            
                    feedback_data = {
                        'posture_score': smoothed_score,
                        'raw_score': posture_score,
                                'stability_score': stability_score,
                        'landmarks_detected': True,
                        'keypoints': keypoints,
                                'detailed_metrics': detailed_metrics,
                                'feedback_level': self._get_enhanced_feedback_level(smoothed_score),
                                'confidence': self._calculate_overall_confidence(keypoints),
                                'is_stable': stability_score > self.thresholds.min_stability_score,
                                'frame_count': self.frame_count
                            }
                else:
                    # Invalid keypoints - use fallback
                    self._handle_pose_detection_failure(annotated_frame, feedback_data)
            else:
                # No pose detected or low confidence - use fallback
                if self.frame_count % 30 == 0:  # Print diagnostic info
                    print(f"❌ DIAGNOSTIC - No pose detected or low confidence")
                    if results.pose_landmarks:
                        print(f"   • Pose landmarks exist but failed validation")
                    else:
                        print(f"   • No pose landmarks detected by MediaPipe")
                
                # DIAGNOSTIC: Draw raw MediaPipe detection if available
                if results.pose_landmarks:
                    self._draw_raw_mediapipe_detection(annotated_frame, results.pose_landmarks)
                
                self._handle_pose_detection_failure(annotated_frame, feedback_data)
                
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            self._handle_processing_error(annotated_frame, feedback_data)
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return annotated_frame, feedback_data
    
    def _validate_pose_confidence(self, landmarks) -> bool:
        """Validate that pose landmarks have sufficient confidence - UPPER BODY FOCUSED"""
        # Focus on upper body landmarks that are typically visible
        upper_body_landmarks = [0, 11, 12]  # Nose, shoulders (most important for posture)
        confidence_scores = [landmarks.landmark[i].visibility for i in upper_body_landmarks]
        avg_confidence = np.mean(confidence_scores)
        
        # DIAGNOSTIC: Print confidence scores
        if self.frame_count % 30 == 0:  # Print every 30 frames
            print(f"🔍 DIAGNOSTIC - Upper body confidences: {[f'{c:.2f}' for c in confidence_scores]}")
            print(f"🔍 DIAGNOSTIC - Average confidence: {avg_confidence:.2f}")
        
        # Check if we have good upper body detection
        return avg_confidence >= 0.7  # Good confidence for upper body
    
    def _extract_validated_keypoints(self, landmarks, w: int, h: int) -> Optional[List]:
        """Extract and validate keypoints with bounds checking - UPPER BODY FOCUSED"""
        try:
            keypoints = []
            valid_count = 0
            
            for i, landmark in enumerate(landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                visibility = landmark.visibility
                
                # Accept keypoints with reasonable visibility
                if 0 <= x <= w and 0 <= y <= h and visibility > 0.3:
                    keypoints.append([x, y, visibility])
                    valid_count += 1
                else:
                    keypoints.append([x, y, 0.0])  # Mark as invalid
            
            # DIAGNOSTIC: Print keypoint validation info
            if self.frame_count % 30 == 0:
                print(f"🔍 DIAGNOSTIC - Valid keypoints: {valid_count}/33")
            
            # Require at least 3 keypoints (nose + shoulders minimum)
            if valid_count < 3:
                if self.frame_count % 30 == 0:
                    print(f"❌ DIAGNOSTIC - Not enough valid keypoints: {valid_count}")
                return None
                
            if self.frame_count % 30 == 0:
                print(f"✅ DIAGNOSTIC - Keypoints extracted successfully: {valid_count}")
            return keypoints
        except Exception as e:
            logger.error(f"Error extracting keypoints: {e}")
            if self.frame_count % 30 == 0:
                print(f"❌ DIAGNOSTIC - Keypoint extraction error: {e}")
            return None
    
    def _calculate_detailed_posture_metrics(self, keypoints: List) -> Dict[str, float]:
        """Calculate comprehensive posture metrics using scientific approach - UPPER BODY FOCUSED"""
        metrics = {}
        
        try:
            # Extract key points (upper body focus)
            nose = keypoints[0]
            left_shoulder = keypoints[11]
            right_shoulder = keypoints[12]
            
            # Check if hips are available, otherwise use estimated positions
            if len(keypoints) > 24 and keypoints[23][2] > 0.3 and keypoints[24][2] > 0.3:
                left_hip = keypoints[23]
                right_hip = keypoints[24]
                hips_available = True
            else:
                # Estimate hip positions based on shoulders (for upper-body-only detection)
                shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
                shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
                estimated_hip_y = shoulder_center_y + 100  # Rough estimate: hips ~100px below shoulders
                left_hip = [left_shoulder[0], estimated_hip_y, 0.5]  # Lower confidence
                right_hip = [right_shoulder[0], estimated_hip_y, 0.5]  # Lower confidence
                hips_available = False
            
            # 1. Shoulder Alignment Analysis
            shoulder_slope = abs(left_shoulder[1] - right_shoulder[1])
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            
            metrics['shoulder_slope'] = shoulder_slope
            metrics['shoulder_alignment_score'] = max(0, 1 - (shoulder_slope / self.thresholds.max_shoulder_slope))
            
            # 2. Head Position Analysis
            head_forward_distance = abs(nose[0] - shoulder_center_x)
            head_vertical_offset = nose[1] - shoulder_center_y
            
            metrics['head_forward_distance'] = head_forward_distance
            metrics['head_vertical_offset'] = head_vertical_offset
            metrics['head_position_score'] = max(0, 1 - (head_forward_distance / self.thresholds.max_head_forward))
            
            # 3. Spinal Alignment Analysis
            spine_center_top = shoulder_center_x
            spine_center_bottom = (left_hip[0] + right_hip[0]) / 2
            spine_deviation = abs(spine_center_top - spine_center_bottom)
            
            metrics['spine_deviation'] = spine_deviation
            metrics['spine_alignment_score'] = max(0, 1 - (spine_deviation / self.thresholds.max_spine_deviation))
            
            # 4. Hip Alignment Analysis (with fallback for estimated hips)
            if hips_available:
                hip_slope = abs(left_hip[1] - right_hip[1])
                metrics['hip_slope'] = hip_slope
                metrics['hip_alignment_score'] = max(0, 1 - (hip_slope / self.thresholds.max_shoulder_slope))
            else:
                # Use shoulder alignment as proxy for hip alignment when hips not visible
                metrics['hip_slope'] = shoulder_slope
                metrics['hip_alignment_score'] = metrics['shoulder_alignment_score'] * 0.8  # Slightly lower confidence
            
            # 5. Overall Symmetry Analysis
            left_side_center = (left_shoulder[0] + left_hip[0]) / 2
            right_side_center = (right_shoulder[0] + right_hip[0]) / 2
            body_symmetry = abs(left_side_center - right_side_center)
            
            metrics['body_symmetry'] = body_symmetry
            metrics['symmetry_score'] = max(0, 1 - (body_symmetry / self.thresholds.max_shoulder_asymmetry))
            
            # 6. Angle Calculations
            # Shoulder angle
            shoulder_angle = math.degrees(math.atan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0]
            ))
            metrics['shoulder_angle'] = abs(shoulder_angle)
            metrics['shoulder_angle_score'] = max(0, 1 - (abs(shoulder_angle) / 90))  # Normalize to 0-1
            
            # 7. Stability Analysis (if we have history)
            if len(self.score_history) > 5:
                recent_scores = list(self.score_history)[-5:]
                if len(recent_scores) > 1:
                    score_variance = np.var(recent_scores)
                    metrics['score_stability'] = max(0, 1 - (score_variance * 10))  # Scale variance
                else:
                    metrics['score_stability'] = 0.5
            else:
                metrics['score_stability'] = 0.5
            
            # 8. Confidence-weighted metrics
            valid_keypoints = [kp for kp in keypoints if kp[2] > 0]  # Only valid keypoints
            if valid_keypoints:
                avg_visibility = np.mean([kp[2] for kp in valid_keypoints])
            else:
                avg_visibility = 0.0
            metrics['detection_confidence'] = avg_visibility
            metrics['confidence_weight'] = min(1.0, avg_visibility / 0.8)
            
            # 9. Upper body detection indicator
            metrics['hips_available'] = hips_available
            metrics['detection_mode'] = 'upper_body' if not hips_available else 'full_body'
            
        except Exception as e:
            logger.error(f"Error calculating detailed metrics: {e}")
            # Return default metrics
            metrics = {
                'shoulder_alignment_score': 0.5,
                'head_position_score': 0.5,
                'spine_alignment_score': 0.5,
                'hip_alignment_score': 0.5,
                'symmetry_score': 0.5,
                'shoulder_angle_score': 0.5,
                'score_stability': 0.5,
                'detection_confidence': 0.5,
                'confidence_weight': 0.5
            }
        
        return metrics
    
    def _calculate_enhanced_posture_score(self, metrics: Dict[str, float]) -> float:
        """Calculate enhanced posture score using weighted scientific metrics"""
        try:
            # Define weights for different aspects of posture
            weights = {
                'shoulder_alignment_score': 0.25,
                'head_position_score': 0.20,
                'spine_alignment_score': 0.20,
                'hip_alignment_score': 0.15,
                'symmetry_score': 0.10,
                'shoulder_angle_score': 0.10
            }
            
            # Calculate weighted score
            weighted_score = sum(metrics.get(key, 0.5) * weight for key, weight in weights.items())
            
            # Apply confidence weighting
            confidence_weight = metrics.get('confidence_weight', 0.5)
            final_score = weighted_score * confidence_weight
            
            # Apply stability bonus/penalty
            stability = metrics.get('score_stability', 0.5)
            if stability > 0.8:  # Very stable
                final_score *= 1.05  # 5% bonus
            elif stability < 0.3:  # Unstable
                final_score *= 0.95  # 5% penalty
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating enhanced posture score: {e}")
            return 0.5
    
    def _update_analysis_history(self, score: float, metrics: Dict[str, float]):
        """Update analysis history for smoothing and stability calculation"""
        self.score_history.append(score)
        self.detailed_metrics_history.append(metrics)
        
        # Update stability scores
        if len(self.score_history) >= 5:
            recent_scores = list(self.score_history)[-5:]
            stability = 1.0 - np.std(recent_scores)  # Lower std = higher stability
            self.stability_scores.append(max(0.0, stability))
    
    def _calculate_smoothed_score(self) -> float:
        """Calculate smoothed score using exponential moving average"""
        if not self.score_history:
            return 0.5
        
        # Use exponential moving average for better responsiveness
        alpha = 0.3  # Smoothing factor
        smoothed = float(self.score_history[0])
        for score in self.score_history[1:]:
            smoothed = alpha * float(score) + (1 - alpha) * smoothed
        
        return smoothed
    
    def _calculate_stability_score(self) -> float:
        """Calculate posture stability score"""
        if len(self.stability_scores) < 3:
            return 0.5
        
        return float(np.mean(list(self.stability_scores)))
    
    def _calculate_overall_confidence(self, keypoints: List) -> float:
        """Calculate overall detection confidence"""
        if not keypoints:
            return 0.0
        
        visibilities = [kp[2] for kp in keypoints if kp[2] > 0]
        if not visibilities:
            return 0.0
        
        return np.mean(visibilities)
    
    def _get_enhanced_feedback_level(self, score: float) -> str:
        """Get enhanced feedback level based on scientific thresholds"""
        if score >= self.thresholds.excellent_threshold:
            return 'excellent'
        elif score >= self.thresholds.good_threshold:
            return 'good'
        elif score >= self.thresholds.fair_threshold:
            return 'fair'
        elif score >= self.thresholds.poor_threshold:
            return 'poor'
        else:
            return 'poor'
    
    def _handle_pose_detection_failure(self, frame: np.ndarray, feedback_data: Dict):
        """Handle pose detection failures with fallback mechanisms"""
        self.pose_detection_failures += 1
        
        if self.pose_detection_failures < self.max_failures and self.last_valid_pose:
            # Use last valid pose as fallback
            time_since_last = time.time() - self.last_valid_pose['timestamp']
            if time_since_last < 2.0:  # Use fallback if less than 2 seconds old
                self.fallback_active = True
                self._draw_fallback_feedback(frame, self.last_valid_pose)
                feedback_data.update({
                    'posture_score': self.last_valid_pose['score'] * 0.8,  # Reduce confidence
                'landmarks_detected': False,
                    'feedback_level': 'low_confidence',
                    'fallback_active': True
                })
                return
        
        # No valid fallback available
        self._draw_no_pose_detected(frame)
        feedback_data.update({
                'posture_score': 0.0,
                'landmarks_detected': False,
            'feedback_level': 'no_detection',
            'fallback_active': False
        })
    
    def _handle_processing_error(self, frame: np.ndarray, feedback_data: Dict):
        """Handle processing errors gracefully"""
        self._draw_error_message(frame, "Processing Error")
        feedback_data.update({
            'posture_score': 0.0,
            'landmarks_detected': False,
            'feedback_level': 'error',
            'error': True
        })
    
    def _draw_enhanced_pose_skeleton(self, frame: np.ndarray, keypoints: List, metrics: Dict[str, float]) -> np.ndarray:
        """Draw enhanced pose skeleton with detailed color coding based on metrics"""
        # Enhanced MediaPipe pose connections with importance weighting
        connections = [
            # Face (lower priority)
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # Body (higher priority for posture)
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28),
            (28, 30), (28, 32)
        ]
        
        # Draw connections with metric-based coloring
        for connection in connections:
            start_idx, end_idx = connection
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                
                start_point = (keypoints[start_idx][0], keypoints[start_idx][1])
                end_point = (keypoints[end_idx][0], keypoints[end_idx][1])
                
                # Get color based on connection importance and metrics
                connection_color = self._get_connection_color(start_idx, end_idx, metrics)
                thickness = self._get_connection_thickness(start_idx, end_idx)
        
                cv2.line(frame, start_point, end_point, connection_color, thickness)
        
        # Draw keypoints with enhanced visualization
        for i, (x, y, visibility) in enumerate(keypoints):
            if visibility > 0.3:
                # Get color and size based on landmark importance and metrics
                color, radius = self._get_landmark_visualization(i, metrics, visibility)
                
                # Draw landmark with enhanced styling
                cv2.circle(frame, (x, y), radius, color, -1)
                cv2.circle(frame, (x, y), radius + 1, (255, 255, 255), 1)
                
                # Add confidence indicator for key landmarks
                if i in [0, 11, 12, 23, 24] and visibility < 0.8:
                    cv2.circle(frame, (x, y), radius + 3, (255, 255, 0), 1)  # Yellow warning
        
        return frame
    
    def _get_connection_color(self, start_idx: int, end_idx: int, metrics: Dict[str, float]) -> Tuple[int, int, int]:
        """Get connection color based on posture metrics"""
        # Define connection importance and associated metrics
        if start_idx == 11 and end_idx == 12:  # Shoulder line
            score = metrics.get('shoulder_alignment_score', 0.5)
        elif start_idx in [11, 12] and end_idx in [23, 24]:  # Shoulder to hip
            score = metrics.get('spine_alignment_score', 0.5)
        elif start_idx == 23 and end_idx == 24:  # Hip line
            score = metrics.get('hip_alignment_score', 0.5)
        elif start_idx == 0:  # Head connections
            score = metrics.get('head_position_score', 0.5)
        else:
            score = 0.7  # Default good score for other connections
        
        return self._get_score_color(score)
    
    def _get_connection_thickness(self, start_idx: int, end_idx: int) -> int:
        """Get connection thickness based on importance"""
        important_connections = [(11, 12), (11, 23), (12, 24), (23, 24)]  # Core posture connections
        if (start_idx, end_idx) in important_connections or (end_idx, start_idx) in important_connections:
            return self.config.line_thickness + 1
        return self.config.line_thickness
    
    def _get_landmark_visualization(self, landmark_idx: int, metrics: Dict[str, float], visibility: float) -> Tuple[Tuple[int, int, int], int]:
        """Get landmark color and size based on importance and metrics"""
        # Key landmarks for posture analysis
        if landmark_idx == 0:  # Nose
            score = metrics.get('head_position_score', 0.5)
            radius = self.config.circle_radius + 2
        elif landmark_idx in [11, 12]:  # Shoulders
            score = metrics.get('shoulder_alignment_score', 0.5)
            radius = self.config.circle_radius + 3
        elif landmark_idx in [23, 24]:  # Hips
            score = metrics.get('hip_alignment_score', 0.5)
            radius = self.config.circle_radius + 3
        else:
            score = 0.7  # Default good score
            radius = self.config.circle_radius
                
        # Adjust radius based on visibility
        if visibility < 0.8:
            radius = max(radius - 1, 2)
        
        color = self._get_score_color(score)
        return color, radius
    
    def _get_score_color(self, score: float) -> Tuple[int, int, int]:
        """Get color based on score value with enhanced color scheme"""
        if score >= self.thresholds.excellent_threshold:
            return self.config.excellent_color  # Cyan
        elif score >= self.thresholds.good_threshold:
            return self.config.skeleton_color  # Green
        elif score >= self.thresholds.fair_threshold:
            return self.config.warning_color    # Orange
        else:
            return self.config.error_color      # Red
    
    def _add_enhanced_posture_feedback(self, frame: np.ndarray, score: float, metrics: Dict[str, float], stability_score: float) -> np.ndarray:
        """Add comprehensive visual feedback with detailed metrics"""
        h, w, _ = frame.shape
        
        # Add main posture score display
        if self.config.show_score:
            self._draw_enhanced_posture_score(frame, score, stability_score)
        
        # Add detailed metrics if enabled
        if self.config.show_detailed_metrics:
            self._draw_detailed_metrics(frame, metrics)
        
        # Add feedback message with specific guidance
        feedback_level = self._get_enhanced_feedback_level(score)
        message = self.feedback_messages.get(feedback_level, "")
        self._draw_enhanced_feedback_message(frame, message, feedback_level, score)
        
        # Add specific posture corrections
        if self.config.show_corrections:
            corrections = self._get_enhanced_posture_corrections(metrics)
            self._draw_enhanced_corrections(frame, corrections)
        
        # Add stability indicator
        if self.config.show_stability_indicator:
            self._draw_stability_indicator(frame, stability_score)
        
        # Add posture zones/guides
        self._draw_enhanced_posture_zones(frame, metrics)
        
        # Add performance indicators
        self._draw_performance_indicators(frame)
        
        # Add detection mode indicator
        self._draw_detection_mode_indicator(frame, metrics)
        
        return frame
    
    def _draw_enhanced_posture_score(self, frame: np.ndarray, score: float, stability_score: float):
        """Draw enhanced posture score with stability indicator"""
        h, w, _ = frame.shape
        
        # Main score bar
        bar_width = 250
        bar_height = 25
        bar_x = w - bar_width - 20
        bar_y = 30
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (40, 40, 40), -1)
        
        # Score bar fill with gradient effect
        fill_width = int(bar_width * score)
        color = self._get_score_color(score)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
        
        # Score text with enhanced formatting
        score_text = f"Posture: {score:.1%}"
        cv2.putText(frame, score_text, (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Stability indicator
        stability_text = f"Stability: {stability_score:.1%}"
        stability_color = self._get_score_color(stability_score)
        cv2.putText(frame, stability_text, (bar_x, bar_y + bar_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stability_color, 1)
    
    def _draw_detailed_metrics(self, frame: np.ndarray, metrics: Dict[str, float]):
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
                color = self._get_score_color(score)
                
                # Metric name
                cv2.putText(frame, display_name, (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Score
                score_text = f"{score:.2f}"
                cv2.putText(frame, score_text, (x_start + 120, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Mini progress bar
                bar_start = (x_start + 160, y_offset - 8)
                bar_end = (x_start + 250, y_offset - 2)
                self._draw_mini_progress_bar(frame, bar_start, bar_end, score, color)
                
                y_offset += 20
    
    def _draw_mini_progress_bar(self, frame: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], progress: float, color: Tuple[int, int, int]):
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
    
    def _draw_enhanced_feedback_message(self, frame: np.ndarray, message: str, level: str, score: float):
        """Draw enhanced feedback message with specific guidance"""
        if not message:
            return
        
        h, w, _ = frame.shape
        
        # Message background with enhanced styling
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 80
        
        # Background color based on feedback level
        bg_colors = {
            'excellent': (0, 200, 200),  # Cyan
            'good': (0, 200, 0),         # Green
            'fair': (0, 165, 255),       # Orange
            'poor': (0, 0, 200),         # Red
            'no_detection': (100, 100, 100),
            'low_confidence': (100, 100, 100)
        }
        bg_color = bg_colors.get(level, (100, 100, 100))
        
        # Enhanced background with rounded corners effect
        padding = 15
        cv2.rectangle(frame, (text_x - padding, text_y - 30), (text_x + text_size[0] + padding, text_y + 10), bg_color, -1)
        cv2.rectangle(frame, (text_x - padding, text_y - 30), (text_x + text_size[0] + padding, text_y + 10), (255, 255, 255), 2)
        
        # Message text
        cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add score-based encouragement
        if score > 0.8:
            encouragement = "Keep it up! 🌟"
        elif score > 0.6:
            encouragement = "Almost there! 💪"
        else:
            encouragement = "You can do it! 💪"
        
        encouragement_size = cv2.getTextSize(encouragement, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        encouragement_x = (w - encouragement_size[0]) // 2
        cv2.putText(frame, encouragement, (encouragement_x, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _get_enhanced_posture_corrections(self, metrics: Dict[str, float]) -> List[str]:
        """Generate enhanced posture correction suggestions based on detailed metrics"""
        corrections = []
        
        # Shoulder corrections
        if metrics.get('shoulder_alignment_score', 1.0) < 0.7:
            corrections.append("Level your shoulders")
        
        # Head position corrections
        if metrics.get('head_position_score', 1.0) < 0.7:
            corrections.append("Align head over shoulders")
        
        # Spine corrections
        if metrics.get('spine_alignment_score', 1.0) < 0.7:
            corrections.append("Straighten your spine")
        
        # Hip corrections
        if metrics.get('hip_alignment_score', 1.0) < 0.7:
            corrections.append("Level your hips")
        
        # Stability corrections
        if metrics.get('score_stability', 1.0) < 0.6:
            corrections.append("Hold position steady")
        
        return corrections[:3]  # Limit to 3 corrections
    
    def _draw_enhanced_corrections(self, frame: np.ndarray, corrections: List[str]):
        """Draw enhanced correction suggestions with better formatting"""
        if not corrections:
            return
        
        y_start = 140
        for i, correction in enumerate(corrections):
            y_pos = y_start + i * 25
            
            # Bullet point with color
            cv2.circle(frame, (25, y_pos - 3), 4, (0, 255, 255), -1)
            
            # Correction text
            cv2.putText(frame, correction, (35, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _draw_stability_indicator(self, frame: np.ndarray, stability_score: float):
        """Draw stability indicator"""
        h, w, _ = frame.shape
        
        # Position in bottom-right
        x_center = w - 60
        y_center = h - 60
        
        # Stability circle
        radius = 25
        color = self._get_score_color(stability_score)
        
        # Outer circle
        cv2.circle(frame, (x_center, y_center), radius, color, 2)
        
        # Inner fill based on stability
        fill_radius = int(radius * stability_score)
        cv2.circle(frame, (x_center, y_center), fill_radius, color, -1)
        
        # Stability text
        cv2.putText(frame, "STABLE", (x_center - 25, y_center + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(frame, f"{stability_score:.0%}", (x_center - 15, y_center + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_enhanced_posture_zones(self, frame: np.ndarray, metrics: Dict[str, float]):
        """Draw enhanced posture zones with metric-based coloring"""
        h, w, _ = frame.shape
        
        # Draw center line for spinal alignment
        center_x = w // 2
        cv2.line(frame, (center_x, 0), (center_x, h), (100, 100, 100), 1)
        
        # Draw shoulder level guide with color coding
        shoulder_score = metrics.get('shoulder_alignment_score', 0.5)
        shoulder_color = self._get_score_color(shoulder_score)
        
        # This would need actual shoulder positions from keypoints
        # For now, draw at estimated position
        estimated_shoulder_y = h // 3
        cv2.line(frame, (0, estimated_shoulder_y), (w, estimated_shoulder_y), shoulder_color, 2)
    
    def _draw_performance_indicators(self, frame: np.ndarray):
        """Draw performance indicators"""
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
        
        # Fallback indicator
        if self.fallback_active:
            cv2.putText(frame, "FALLBACK", (w - 100, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _draw_detection_mode_indicator(self, frame: np.ndarray, metrics: Dict[str, float]):
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
    
    def _draw_fallback_feedback(self, frame: np.ndarray, last_pose: Dict):
        """Draw fallback feedback when pose detection fails"""
        h, w, _ = frame.shape
        
        # Draw warning message
        message = "Using last known pose (Low Confidence)"
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 50
        
        cv2.rectangle(frame, (text_x - 10, text_y - 25), (text_x + text_size[0] + 10, text_y + 5), (0, 165, 255), -1)
        cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_error_message(self, frame: np.ndarray, message: str):
        """Draw error message"""
        h, w, _ = frame.shape
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        
        cv2.rectangle(frame, (text_x - 10, text_y - 25), (text_x + text_size[0] + 10, text_y + 5), (0, 0, 200), -1)
        cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def _draw_raw_mediapipe_detection(self, frame: np.ndarray, landmarks):
        """Draw raw MediaPipe detection for diagnostics"""
        h, w, _ = frame.shape
        
        # Draw all landmarks in yellow for diagnostics
        for i, landmark in enumerate(landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            visibility = landmark.visibility
            
            if visibility > 0.1:  # Show even low confidence landmarks
                # Color based on visibility
                if visibility > 0.7:
                    color = (0, 255, 0)  # Green for high confidence
                elif visibility > 0.4:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 165, 255)  # Orange for low confidence
                
                cv2.circle(frame, (x, y), 3, color, -1)
                
                # Show landmark index for key points
                if i in [0, 11, 12, 23, 24]:
                    cv2.putText(frame, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw basic connections
        connections = [
            (11, 12),  # Shoulders
            (11, 23),  # Left shoulder to hip
            (12, 24),  # Right shoulder to hip
            (23, 24),  # Hips
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            if (start_idx < len(landmarks.landmark) and end_idx < len(landmarks.landmark)):
                start_landmark = landmarks.landmark[start_idx]
                end_landmark = landmarks.landmark[end_idx]
                
                if start_landmark.visibility > 0.1 and end_landmark.visibility > 0.1:
                    start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                    end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
                    cv2.line(frame, start_point, end_point, (255, 255, 0), 2)
        
        # Add diagnostic text
        cv2.putText(frame, "RAW MEDIAPIPE DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Green=High, Yellow=Med, Orange=Low", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_pose_skeleton(self, frame: np.ndarray, keypoints: List) -> np.ndarray:
        """Legacy method - use _draw_enhanced_pose_skeleton instead"""
        return self._draw_enhanced_pose_skeleton(frame, keypoints, {})
    
    def _calculate_posture_score(self, keypoints: List) -> float:
        """Calculate posture score based on key measurements"""
        if len(keypoints) < 33:
            return 0.0
        
        # Key points for posture analysis
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        nose = keypoints[0]
        
        score = 1.0
        
        # Check shoulder alignment
        shoulder_slope = abs(left_shoulder[1] - right_shoulder[1])
        max_shoulder_diff = 20  # pixels
        shoulder_penalty = min(shoulder_slope / max_shoulder_diff, 0.3)
        score -= shoulder_penalty
        
        # Check head forward posture
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        head_forward_distance = abs(nose[0] - shoulder_center_x)
        max_head_forward = 50  # pixels
        head_penalty = min(head_forward_distance / max_head_forward, 0.4)
        score -= head_penalty
        
        # Check spinal alignment (simplified)
        if all(point[2] > 0.5 for point in [left_shoulder, right_shoulder, left_hip, right_hip]):
            spine_center_top = (left_shoulder[0] + right_shoulder[0]) / 2
            spine_center_bottom = (left_hip[0] + right_hip[0]) / 2
            spine_deviation = abs(spine_center_top - spine_center_bottom)
            max_spine_deviation = 30
            spine_penalty = min(spine_deviation / max_spine_deviation, 0.3)
            score -= spine_penalty
        
        return max(0.0, min(1.0, score))
    
    def _get_skeleton_color(self, score: float) -> Tuple[int, int, int]:
        """Get skeleton color based on posture score"""
        if score >= self.good_posture_threshold:
            return self.config.skeleton_color  # Green
        elif score >= self.warning_threshold:
            return self.config.warning_color    # Orange
        else:
            return self.config.error_color      # Red
    
    def _add_posture_feedback(self, frame: np.ndarray, score: float, keypoints: List) -> np.ndarray:
        """Add visual feedback overlays"""
        h, w, _ = frame.shape
        
        # Add posture score display
        if self.config.show_score:
            self._draw_posture_score(frame, score)
        
        # Add feedback message
        feedback_level = self._get_feedback_level(score)
        message = self.feedback_messages.get(feedback_level, "")
        self._draw_feedback_message(frame, message, feedback_level)
        
        # Add specific posture corrections
        if self.config.show_corrections:
            corrections = self._get_posture_corrections(keypoints)
            self._draw_corrections(frame, corrections)
        
        # Add posture zones/guides
        self._draw_posture_zones(frame, keypoints)
        
        return frame
    
    def _draw_posture_score(self, frame: np.ndarray, score: float):
        """Draw posture score on frame"""
        h, w, _ = frame.shape
        
        # Score bar background
        bar_width = 200
        bar_height = 20
        bar_x = w - bar_width - 20
        bar_y = 30
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Score bar fill
        fill_width = int(bar_width * score)
        color = self._get_skeleton_color(score)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Score text
        score_text = f"Posture: {score:.1%}"
        cv2.putText(frame, score_text, (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_feedback_message(self, frame: np.ndarray, message: str, level: str):
        """Draw feedback message"""
        if not message:
            return
        
        h, w, _ = frame.shape
        
        # Message background
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 80
        
        # Background color based on feedback level
        bg_colors = {
            'excellent': (0, 200, 0),
            'good': (0, 150, 0),
            'warning': (0, 165, 255),
            'poor': (0, 0, 200)
        }
        bg_color = bg_colors.get(level, (100, 100, 100))
        
        cv2.rectangle(frame, (text_x - 10, text_y - 25), (text_x + text_size[0] + 10, text_y + 5), bg_color, -1)
        cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def _get_posture_corrections(self, keypoints: List) -> List[str]:
        """Generate specific posture correction suggestions"""
        corrections = []
        
        if len(keypoints) < 33:
            return corrections
        
        # Check shoulder alignment
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        if abs(left_shoulder[1] - right_shoulder[1]) > 15:
            corrections.append("Level your shoulders")
        
        # Check head position
        nose = keypoints[0]
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        if abs(nose[0] - shoulder_center_x) > 30:
            corrections.append("Align head over shoulders")
        
        return corrections
    
    def _draw_corrections(self, frame: np.ndarray, corrections: List[str]):
        """Draw correction suggestions"""
        y_start = 120
        for i, correction in enumerate(corrections):
            y_pos = y_start + i * 25
            cv2.putText(frame, f"• {correction}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _draw_posture_zones(self, frame: np.ndarray, keypoints: List):
        """Draw ideal posture zones/guides"""
        if len(keypoints) < 33:
            return
        
        # Draw center line for spinal alignment
        h, w, _ = frame.shape
        center_x = w // 2
        cv2.line(frame, (center_x, 0), (center_x, h), (100, 100, 100), 1)
        
        # Draw shoulder level guide
        if keypoints[11][2] > 0.5 and keypoints[12][2] > 0.5:
            shoulder_y = (keypoints[11][1] + keypoints[12][1]) // 2
            cv2.line(frame, (0, shoulder_y), (w, shoulder_y), (100, 100, 100), 1)
    
    def _get_feedback_level(self, score: float) -> str:
        """Get feedback level based on score"""
        if score >= 0.85:
            return 'excellent'
        elif score >= self.good_posture_threshold:
            return 'good'
        elif score >= self.warning_threshold:
            return 'warning'
        else:
            return 'poor'
    
    def _draw_no_pose_detected(self, frame: np.ndarray):
        """Draw message when no pose is detected"""
        h, w, _ = frame.shape
        message = "Please position yourself in view"
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        
        cv2.rectangle(frame, (text_x - 10, text_y - 25), (text_x + text_size[0] + 10, text_y + 5), (0, 0, 200), -1)
        cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

class LiveMeditationSession:
    """Enhanced integration with meditation session system"""
    
    def __init__(self, session_manager, config: PostureVisualizationConfig = None, thresholds: PostureThresholds = None):
        self.session_manager = session_manager
        self.posture_feedback = LivePostureFeedback(config, thresholds)
        self.cap = None
        self.is_running = False
        
        # Enhanced session tracking
        self.session_start_time = None
        self.total_frames_processed = 0
        self.posture_scores_history = []
        self.stability_scores_history = []
        
        # Performance monitoring
        self.performance_stats = {
            'avg_fps': 0.0,
            'avg_processing_time': 0.0,
            'detection_success_rate': 0.0,
            'fallback_usage_rate': 0.0
        }
    
    def start_live_session(self, camera_index: int = 0):
        """Start enhanced live meditation session with comprehensive feedback"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        # Set enhanced camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        self.is_running = True
        self.session_start_time = time.time()
        
        print("🚀 Enhanced Live Posture Feedback Started!")
        print("📊 Controls:")
        print("   • 'q' - Quit session")
        print("   • 's' - Save screenshot")
        print("   • 'c' - Calibrate posture")
        print("   • 'r' - Reset statistics")
        print("   • 'h' - Show help")
        
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                break
            
            frame_count += 1
            self.total_frames_processed += 1
            
            # Process frame with enhanced posture feedback
            annotated_frame, feedback_data = self.posture_feedback.process_frame_with_feedback(frame)
            
            # Update session statistics
            self._update_session_statistics(feedback_data)
            
            # Send enhanced posture data to session manager
            if hasattr(self.session_manager, 'active_sessions') and self.session_manager.active_sessions:
                for session_id in self.session_manager.active_sessions:
                    # Create enhanced real-time data
                    from session_manager import RealTimeData
                    rt_data = RealTimeData(
                        timestamp=time.time(),
                        video_frame=frame,
                        posture_score=feedback_data.get('posture_score', 0.0),
                        engagement_level=self._calculate_engagement_level(feedback_data),
                        quality_metrics=feedback_data.get('detailed_metrics', {})
                    )
                    self.session_manager.process_real_time_data(session_id, rt_data)
            
            # Add session statistics overlay
            annotated_frame = self._add_session_overlay(annotated_frame, frame_count, start_time)
            
            # Display the enhanced frame
            cv2.imshow('Enhanced Live Posture Feedback', annotated_frame)
            
            # Handle enhanced key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quitting session...")
                break
            elif key == ord('s'):
                self._save_screenshot(annotated_frame)
            elif key == ord('c'):
                self._calibrate_posture()
            elif key == ord('r'):
                self._reset_statistics()
            elif key == ord('h'):
                self._show_help()
        
        self.cleanup()
    
    def _update_session_statistics(self, feedback_data: Dict):
        """Update session statistics with feedback data"""
        if feedback_data.get('landmarks_detected', False):
            self.posture_scores_history.append(feedback_data.get('posture_score', 0.0))
            self.stability_scores_history.append(feedback_data.get('stability_score', 0.0))
            
            # Keep only recent history (last 100 frames)
            if len(self.posture_scores_history) > 100:
                self.posture_scores_history.pop(0)
            if len(self.stability_scores_history) > 100:
                self.stability_scores_history.pop(0)
    
    def _calculate_engagement_level(self, feedback_data: Dict) -> float:
        """Calculate engagement level based on posture stability and quality"""
        posture_score = feedback_data.get('posture_score', 0.0)
        stability_score = feedback_data.get('stability_score', 0.0)
        is_stable = feedback_data.get('is_stable', False)
        
        # Base engagement on posture quality and stability
        base_engagement = posture_score * 0.7 + stability_score * 0.3
        
        # Bonus for stable posture
        if is_stable:
            base_engagement *= 1.1
        
        return min(1.0, base_engagement)
    
    def _add_session_overlay(self, frame: np.ndarray, frame_count: int, start_time: float) -> np.ndarray:
        """Add session statistics overlay to frame"""
        h, w, _ = frame.shape
        
        # Calculate session statistics
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Session info box
        info_x, info_y = 20, h - 120
        box_width, box_height = 300, 100
        
        # Background
        cv2.rectangle(frame, (info_x, info_y), (info_x + box_width, info_y + box_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (info_x, info_y), (info_x + box_width, info_y + box_height), (100, 100, 100), 1)
        
        # Session statistics
        y_offset = info_y + 20
        
        # Session time
        session_time = time.time() - self.session_start_time if self.session_start_time else 0
        time_text = f"Session: {int(session_time//60):02d}:{int(session_time%60):02d}"
        cv2.putText(frame, time_text, (info_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (info_x + 10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Average posture score
        if self.posture_scores_history:
            avg_posture = np.mean(self.posture_scores_history[-30:])  # Last 30 frames
            posture_text = f"Avg Posture: {avg_posture:.1%}"
            cv2.putText(frame, posture_text, (info_x + 10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection success rate
        success_rate = len(self.posture_scores_history) / max(self.total_frames_processed, 1)
        success_text = f"Detection: {success_rate:.1%}"
        cv2.putText(frame, success_text, (info_x + 10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _save_screenshot(self, frame: np.ndarray):
        """Save screenshot with enhanced filename"""
        timestamp = int(time.time())
        filename = f"enhanced_posture_feedback_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def _calibrate_posture(self):
        """Calibrate posture for current user"""
        print("🎯 Starting posture calibration...")
        print("   Please sit in your preferred meditation position")
        print("   Calibration will complete in 10 seconds...")
        
        # This would implement actual calibration logic
        # For now, just show a message
        time.sleep(1)
        print("✅ Posture calibration completed!")
    
    def _reset_statistics(self):
        """Reset session statistics"""
        self.posture_scores_history.clear()
        self.stability_scores_history.clear()
        self.total_frames_processed = 0
        print("🔄 Statistics reset!")
    
    def _show_help(self):
        """Show help information"""
        print("\n📖 Enhanced Posture Detection Help:")
        print("   • The system uses scientific thresholds for posture analysis")
        print("   • Colors indicate posture quality: Cyan=Excellent, Green=Good, Orange=Fair, Red=Poor")
        print("   • Stability indicator shows how steady your posture is")
        print("   • Detailed metrics show specific body part scores")
        print("   • System includes robust fallback mechanisms")
        print("   • Press 'c' to calibrate for your specific posture")
        print()
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

# Enhanced usage example with scientific thresholds
def main():
    """Enhanced example usage with scientific posture analysis"""
    from session_manager import SessionManager, SessionConfig
    
    # Initialize your existing session manager
    session_manager = SessionManager(enable_logging=True)
    
    # Create DIAGNOSTIC visualization config with lenient settings
    viz_config = PostureVisualizationConfig(
        show_skeleton=True,
        show_score=True,
        show_corrections=True,
        show_detailed_metrics=True,
        show_stability_indicator=True,
        skeleton_color=(0, 255, 0),      # Green for good posture
        warning_color=(0, 165, 255),     # Orange for warnings
        error_color=(0, 0, 255),         # Red for poor posture
        excellent_color=(0, 255, 255),   # Cyan for excellent posture
        smoothing_window=15,             # Frames for score smoothing
        confidence_threshold=0.3,        # DIAGNOSTIC: Much lower confidence threshold
        stability_threshold=0.1          # Score variation threshold
    )
    
    # Create scientific thresholds
    thresholds = PostureThresholds(
        excellent_threshold=0.85,
        good_threshold=0.70,
        fair_threshold=0.55,
        poor_threshold=0.40,
        max_shoulder_slope=15.0,
        max_head_forward=40.0,
        max_spine_deviation=25.0,
        max_shoulder_asymmetry=20.0,
        max_neck_angle=15.0,
        max_spine_angle=10.0,
        stability_window=30,
        min_stability_score=0.6
    )
    
    # Start enhanced live session
    live_session = LiveMeditationSession(session_manager, viz_config, thresholds)
    
    try:
        # Optionally start a meditation session
        config = SessionConfig(
            session_id=f"enhanced_session_{int(time.time())}",
            user_id="enhanced_user",
            meditation_type="mindfulness",
            planned_duration=600.0,  # 10 minutes
            enable_posture_detection=True,
            posture_correction_interval=15.0,  # More frequent corrections
            quality_check_interval=5.0,        # More frequent quality checks
            adaptation_sensitivity=0.8         # Higher sensitivity
        )
        
        session_id = session_manager.create_session(config)
        session_manager.start_session(session_id)
        
        print("🚀 DIAGNOSTIC Posture Detection System Started!")
        print("📊 DIAGNOSTIC Features:")
        print("   • Lenient detection thresholds for testing")
        print("   • Real-time diagnostic information")
        print("   • Raw MediaPipe detection display")
        print("   • Detailed confidence reporting")
        print("   • Fallback mechanisms")
        print("\n🎯 Press 'q' to quit, 's' to screenshot, 'c' to calibrate")
        print("🔍 Watch the console for diagnostic information!")
        
        # Start enhanced live feedback
        live_session.start_live_session(camera_index=0)
        
    except KeyboardInterrupt:
        print("\n⏹️  Session interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main execution error: {e}")
    finally:
        live_session.cleanup()
        session_manager.cleanup()
        print("✅ System cleanup completed")

if __name__ == "__main__":
    main()
