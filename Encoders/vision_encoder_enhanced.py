
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Tuple, Any, Optional
import time
import logging
from scipy.spatial.distance import euclidean
import math

class VisionEncoder:
    """Enhanced vision encoder with comprehensive pose analysis and visual feedback"""

    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.logger = logging.getLogger(__name__)

        # Visual feedback settings
        self.feedback_colors = {
            'excellent': (0, 255, 0),     # Green
            'good': (0, 255, 255),        # Yellow
            'needs_attention': (0, 165, 255),  # Orange
            'poor': (0, 0, 255),          # Red
            'neutral': (255, 255, 255)    # White
        }

        # Font settings for text overlay
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame and extract comprehensive pose features"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if not results.pose_landmarks:
                return self._generate_empty_result(frame)

            # Extract enhanced pose features
            pose_features = self._extract_enhanced_pose_features(frame, results)

            # Compute pose embedding with new measurements
            pose_embedding = self._compute_pose_embedding(results.pose_landmarks.landmark)

            return {
                'pose_embedding': pose_embedding,
                'pose_features': pose_features,
                'landmarks': results.pose_landmarks,
                'segmentation_mask': results.segmentation_mask,
                'processed_frame': frame,
                'timestamp': time.time()
            }

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return self._generate_error_result(frame, str(e))

    def render_posture_feedback(self, frame: np.ndarray, posture_data: Dict[str, Any]) -> np.ndarray:
        """Render detailed visual feedback on the frame"""
        try:
            feedback_frame = frame.copy()

            # Draw pose landmarks if available
            if 'landmarks' in posture_data and posture_data['landmarks']:
                self._draw_enhanced_pose_landmarks(feedback_frame, posture_data['landmarks'])

            # Draw posture score
            if 'posture_score' in posture_data:
                self._draw_posture_score(feedback_frame, posture_data['posture_score'])

            # Draw body part scores
            if 'body_part_scores' in posture_data:
                self._draw_body_part_feedback(feedback_frame, posture_data['body_part_scores'])

            # Draw recommendations
            if 'recommendations' in posture_data:
                self._draw_recommendations(feedback_frame, posture_data['recommendations'])

            # Draw stability indicator
            if 'stability_score' in posture_data:
                self._draw_stability_indicator(feedback_frame, posture_data['stability_score'])

            # Draw detailed metrics overlay
            if 'detailed_metrics' in posture_data:
                self._draw_metrics_overlay(feedback_frame, posture_data['detailed_metrics'])

            return feedback_frame

        except Exception as e:
            self.logger.error(f"Feedback rendering error: {e}")
            return frame  # Return original frame on error

    def _extract_enhanced_pose_features(self, frame: np.ndarray, results) -> Dict[str, Any]:
        """Extract comprehensive pose features including new measurements"""
        features = {}

        if not results.pose_landmarks:
            return features

        landmarks = results.pose_landmarks.landmark

        # Convert landmarks to pixel coordinates
        h, w, _ = frame.shape
        landmark_coords = []
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z
            landmark_coords.append([x, y, z])

        landmark_array = np.array(landmark_coords)

        # Extract detailed measurements
        features['shoulder_alignment'] = self._calculate_shoulder_alignment(landmark_array)
        features['spine_straightness'] = self._calculate_spine_straightness(landmark_array)
        features['head_stability'] = self._calculate_head_stability(landmark_array)
        features['breathing_space'] = self._calculate_breathing_space(landmark_array)
        features['forward_head_posture'] = self._calculate_forward_head_posture(landmark_array)
        features['shoulder_elevation'] = self._calculate_shoulder_elevation(landmark_array)
        features['hip_alignment'] = self._calculate_hip_alignment(landmark_array)
        features['overall_symmetry'] = self._calculate_overall_symmetry(landmark_array)

        # Calculate pose confidence
        features['pose_confidence'] = self._calculate_pose_confidence(landmarks)

        # Extract key angles
        features['key_angles'] = self._extract_key_angles(landmark_array)

        return features

    def _compute_pose_embedding(self, keypoints) -> List[float]:
        """Compute pose embedding with enhanced measurements"""
        embedding = []

        try:
            # Convert landmarks to array
            coords = []
            for landmark in keypoints:
                coords.append([landmark.x, landmark.y, landmark.z])
            landmark_array = np.array(coords)

            # Key landmark indices (MediaPipe pose landmarks)
            nose = 0
            left_shoulder = 11
            right_shoulder = 12
            left_elbow = 13
            right_elbow = 14
            left_hip = 23
            right_hip = 24
            left_knee = 25
            right_knee = 26

            # Basic distances and angles
            shoulder_width = euclidean(landmark_array[left_shoulder][:2], landmark_array[right_shoulder][:2])
            hip_width = euclidean(landmark_array[left_hip][:2], landmark_array[right_hip][:2])

            # Shoulder alignment
            shoulder_height_diff = abs(landmark_array[left_shoulder][1] - landmark_array[right_shoulder][1])
            shoulder_alignment = 1.0 - min(1.0, shoulder_height_diff / shoulder_width) if shoulder_width > 0 else 0.5

            # Spine straightness
            shoulder_center = (landmark_array[left_shoulder] + landmark_array[right_shoulder]) / 2
            hip_center = (landmark_array[left_hip] + landmark_array[right_hip]) / 2
            spine_vector = shoulder_center - hip_center
            spine_angle = math.degrees(math.atan2(abs(spine_vector[0]), abs(spine_vector[1])))
            spine_straightness = max(0, 1 - (spine_angle / 30.0))  # Normalize by 30 degrees

            # Head stability (distance from nose to shoulder center)
            head_shoulder_distance = euclidean(landmark_array[nose][:2], shoulder_center[:2])
            head_stability = min(1.0, head_shoulder_distance / shoulder_width) if shoulder_width > 0 else 0.5

            # Breathing space (shoulder width relative to hip width)
            breathing_space = min(1.0, shoulder_width / hip_width) if hip_width > 0 else 0.5

            # Build embedding vector (23 elements to match expected size)
            embedding = [
                shoulder_alignment,
                spine_straightness, 
                head_stability,
                breathing_space,
                shoulder_width / 100.0,  # Normalize
                hip_width / 100.0,  # Normalize
                landmark_array[nose][0] / 100.0,  # Nose X
                landmark_array[nose][1] / 100.0,  # Nose Y
                landmark_array[left_shoulder][0] / 100.0,  # Left shoulder X
                landmark_array[left_shoulder][1] / 100.0,  # Left shoulder Y
                landmark_array[right_shoulder][0] / 100.0,  # Right shoulder X
                landmark_array[right_shoulder][1] / 100.0,  # Right shoulder Y
                landmark_array[left_hip][0] / 100.0,  # Left hip X
                landmark_array[left_hip][1] / 100.0,  # Left hip Y
                landmark_array[right_hip][0] / 100.0,  # Right hip X
                landmark_array[right_hip][1] / 100.0,  # Right hip Y
                # Additional computed features
                abs(landmark_array[left_shoulder][1] - landmark_array[right_shoulder][1]) / 100.0,  # Shoulder height diff
                abs(landmark_array[left_hip][1] - landmark_array[right_hip][1]) / 100.0,  # Hip height diff
                spine_angle / 180.0,  # Spine angle normalized
                head_shoulder_distance / 100.0,  # Head-shoulder distance
                # Fill remaining slots with key measurements
                (landmark_array[left_shoulder][1] + landmark_array[right_shoulder][1]) / 200.0,  # Avg shoulder height
                (landmark_array[left_hip][1] + landmark_array[right_hip][1]) / 200.0,  # Avg hip height
                min(1.0, max(0.0, breathing_space))  # Clamped breathing space
            ]

            # Ensure we have exactly 23 elements
            while len(embedding) < 23:
                embedding.append(0.5)  # Neutral values
            embedding = embedding[:23]  # Truncate if too long

            return embedding

        except Exception as e:
            self.logger.error(f"Pose embedding computation error: {e}")
            return [0.5] * 23  # Return neutral embedding

    def _calculate_shoulder_alignment(self, landmarks: np.ndarray) -> float:
        """Calculate shoulder alignment score"""
        try:
            left_shoulder = landmarks[11]  # Left shoulder
            right_shoulder = landmarks[12]  # Right shoulder

            height_diff = abs(left_shoulder[1] - right_shoulder[1])
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])

            if shoulder_width > 0:
                alignment_score = 1.0 - min(1.0, height_diff / shoulder_width)
            else:
                alignment_score = 0.5

            return alignment_score

        except Exception:
            return 0.5

    def _calculate_spine_straightness(self, landmarks: np.ndarray) -> float:
        """Calculate spine straightness"""
        try:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2

            spine_vector = shoulder_center - hip_center
            angle = math.degrees(math.atan2(abs(spine_vector[0]), abs(spine_vector[1])))

            # Score based on how close to vertical (0 degrees)
            straightness = max(0, 1 - (angle / 30.0))
            return straightness

        except Exception:
            return 0.5

    def _calculate_head_stability(self, landmarks: np.ndarray) -> float:
        """Calculate head stability"""
        try:
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            shoulder_center = (left_shoulder + right_shoulder) / 2
            head_shoulder_distance = euclidean(nose[:2], shoulder_center[:2])
            shoulder_width = euclidean(left_shoulder[:2], right_shoulder[:2])

            if shoulder_width > 0:
                stability = min(1.0, head_shoulder_distance / shoulder_width)
            else:
                stability = 0.5

            return stability

        except Exception:
            return 0.5

    def _calculate_breathing_space(self, landmarks: np.ndarray) -> float:
        """Calculate breathing space (chest openness)"""
        try:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            shoulder_width = euclidean(left_shoulder[:2], right_shoulder[:2])

            # Use relative measurement (could be enhanced with baseline)
            if hasattr(self, 'baseline_shoulder_width'):
                breathing_space = min(1.0, shoulder_width / self.baseline_shoulder_width)
            else:
                self.baseline_shoulder_width = shoulder_width
                breathing_space = 0.8  # Default good score

            return breathing_space

        except Exception:
            return 0.5

    def _calculate_forward_head_posture(self, landmarks: np.ndarray) -> float:
        """Calculate forward head posture"""
        try:
            nose = landmarks[0]
            left_ear = landmarks[7] if len(landmarks) > 7 else landmarks[0]  # Fallback to nose
            right_ear = landmarks[8] if len(landmarks) > 8 else landmarks[0]

            ear_center = (left_ear + right_ear) / 2
            forward_distance = abs(nose[0] - ear_center[0])

            # Score based on minimal forward distance
            max_acceptable_distance = 50  # pixels
            score = max(0, 1 - (forward_distance / max_acceptable_distance))

            return score

        except Exception:
            return 0.5

    def _calculate_shoulder_elevation(self, landmarks: np.ndarray) -> float:
        """Calculate shoulder elevation (tension indicator)"""
        try:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_ear = landmarks[7] if len(landmarks) > 7 else landmarks[0]
            right_ear = landmarks[8] if len(landmarks) > 8 else landmarks[0]

            # Calculate ear-shoulder distances
            left_distance = euclidean(left_shoulder[:2], left_ear[:2])
            right_distance = euclidean(right_shoulder[:2], right_ear[:2])
            avg_distance = (left_distance + right_distance) / 2

            # Normalize (larger distance = less elevation = better score)
            if hasattr(self, 'baseline_ear_shoulder_distance'):
                score = min(1.0, avg_distance / self.baseline_ear_shoulder_distance)
            else:
                self.baseline_ear_shoulder_distance = avg_distance
                score = 0.8

            return score

        except Exception:
            return 0.5

    def _calculate_hip_alignment(self, landmarks: np.ndarray) -> float:
        """Calculate hip alignment"""
        try:
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            height_diff = abs(left_hip[1] - right_hip[1])
            hip_width = abs(left_hip[0] - right_hip[0])

            if hip_width > 0:
                alignment = 1.0 - min(1.0, height_diff / hip_width)
            else:
                alignment = 0.5

            return alignment

        except Exception:
            return 0.5

    def _calculate_overall_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate overall body symmetry"""
        try:
            # Key symmetrical pairs
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            # Calculate symmetry scores
            shoulder_symmetry = self._calculate_bilateral_symmetry(left_shoulder, right_shoulder)
            hip_symmetry = self._calculate_bilateral_symmetry(left_hip, right_hip)

            # Average symmetry
            overall_symmetry = (shoulder_symmetry + hip_symmetry) / 2

            return overall_symmetry

        except Exception:
            return 0.5

    def _calculate_bilateral_symmetry(self, left_point: np.ndarray, right_point: np.ndarray) -> float:
        """Calculate symmetry between bilateral points"""
        try:
            center_point = (left_point + right_point) / 2
            left_distance = euclidean(left_point, center_point)
            right_distance = euclidean(right_point, center_point)

            if max(left_distance, right_distance) > 0:
                symmetry = min(left_distance, right_distance) / max(left_distance, right_distance)
            else:
                symmetry = 1.0

            return symmetry

        except Exception:
            return 0.5

    def _calculate_pose_confidence(self, landmarks) -> float:
        """Calculate overall pose confidence"""
        try:
            confidences = [landmark.visibility for landmark in landmarks if hasattr(landmark, 'visibility')]
            if confidences:
                return sum(confidences) / len(confidences)
            else:
                return 0.8  # Default confidence
        except Exception:
            return 0.5

    def _extract_key_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Extract key body angles"""
        angles = {}

        try:
            # Neck angle
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            shoulder_center = (left_shoulder + right_shoulder) / 2

            neck_vector = nose - shoulder_center
            angles['neck_angle'] = math.degrees(math.atan2(neck_vector[1], neck_vector[0]))

            # Torso angle
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            hip_center = (left_hip + right_hip) / 2

            torso_vector = shoulder_center - hip_center
            angles['torso_angle'] = math.degrees(math.atan2(torso_vector[0], torso_vector[1]))

        except Exception as e:
            self.logger.warning(f"Angle extraction error: {e}")
            angles = {'neck_angle': 0.0, 'torso_angle': 0.0}

        return angles

    def _draw_enhanced_pose_landmarks(self, frame: np.ndarray, landmarks):
        """Draw enhanced pose landmarks with color coding"""
        try:
            # Draw standard pose connections
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Highlight key landmarks for meditation
            h, w, _ = frame.shape
            key_landmarks = [0, 11, 12, 23, 24]  # Nose, shoulders, hips

            for idx in key_landmarks:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # Green highlight

        except Exception as e:
            self.logger.warning(f"Landmark drawing error: {e}")

    def _draw_posture_score(self, frame: np.ndarray, score: float):
        """Draw overall posture score"""
        try:
            # Choose color based on score
            if score >= 0.8:
                color = self.feedback_colors['excellent']
            elif score >= 0.6:
                color = self.feedback_colors['good']
            elif score >= 0.4:
                color = self.feedback_colors['needs_attention']
            else:
                color = self.feedback_colors['poor']

            # Draw score text
            score_text = f"Posture Score: {score:.1%}"
            text_size = cv2.getTextSize(score_text, self.font, self.font_scale, self.font_thickness)[0]

            # Position in top-left corner
            x = 10
            y = 30

            # Draw background rectangle
            cv2.rectangle(frame, (x-5, y-text_size[1]-5), (x+text_size[0]+5, y+5), (0, 0, 0), -1)

            # Draw text
            cv2.putText(frame, score_text, (x, y), self.font, self.font_scale, color, self.font_thickness)

        except Exception as e:
            self.logger.warning(f"Score drawing error: {e}")

    def _draw_body_part_feedback(self, frame: np.ndarray, body_part_scores: Dict[str, float]):
        """Draw body part specific feedback"""
        try:
            h, w, _ = frame.shape
            y_offset = 70  # Start below posture score

            for part_name, score in body_part_scores.items():
                # Choose color based on score
                if score >= 0.7:
                    color = self.feedback_colors['excellent']
                elif score >= 0.5:
                    color = self.feedback_colors['good']
                else:
                    color = self.feedback_colors['needs_attention']

                # Format text
                score_text = f"{part_name.replace('_', ' ').title()}: {score:.1%}"

                # Draw text
                cv2.putText(frame, score_text, (10, y_offset), self.font, 0.5, color, 1)
                y_offset += 25

        except Exception as e:
            self.logger.warning(f"Body part feedback drawing error: {e}")

    def _draw_recommendations(self, frame: np.ndarray, recommendations: List[str]):
        """Draw posture recommendations"""
        try:
            if not recommendations:
                return

            h, w, _ = frame.shape

            # Position recommendations in bottom area
            y_start = h - 100

            # Draw background for recommendations
            cv2.rectangle(frame, (10, y_start - 30), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, y_start - 30), (w - 10, h - 10), (255, 255, 255), 2)

            # Draw recommendations
            for i, rec in enumerate(recommendations[:3]):  # Max 3 recommendations
                y_pos = y_start + (i * 25)
                # Truncate long recommendations
                if len(rec) > 50:
                    rec = rec[:47] + "..."

                cv2.putText(frame, rec, (15, y_pos), self.font, 0.5, 
                           self.feedback_colors['neutral'], 1)

        except Exception as e:
            self.logger.warning(f"Recommendations drawing error: {e}")

    def _draw_stability_indicator(self, frame: np.ndarray, stability_score: float):
        """Draw stability indicator"""
        try:
            h, w, _ = frame.shape

            # Position in top-right
            x = w - 150
            y = 30

            # Choose color
            if stability_score >= 0.8:
                color = self.feedback_colors['excellent']
            elif stability_score >= 0.6:
                color = self.feedback_colors['good']
            else:
                color = self.feedback_colors['needs_attention']

            # Draw stability indicator
            stability_text = f"Stability: {stability_score:.1%}"
            cv2.putText(frame, stability_text, (x, y), self.font, self.font_scale, color, self.font_thickness)

            # Draw stability bar
            bar_width = 100
            bar_height = 10
            bar_x = x
            bar_y = y + 10

            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

            # Filled bar
            fill_width = int(bar_width * stability_score)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)

        except Exception as e:
            self.logger.warning(f"Stability indicator drawing error: {e}")

    def _draw_metrics_overlay(self, frame: np.ndarray, metrics: Dict[str, float]):
        """Draw detailed metrics overlay"""
        try:
            h, w, _ = frame.shape

            # Position in right side
            x_start = w - 200
            y_start = 80

            # Draw background
            overlay_height = len(metrics) * 20 + 20
            cv2.rectangle(frame, (x_start - 10, y_start - 15), 
                         (w - 10, y_start + overlay_height), (0, 0, 0), -1)

            # Draw metrics
            for i, (metric_name, value) in enumerate(metrics.items()):
                y_pos = y_start + (i * 20)

                # Choose color based on value
                if value >= 0.7:
                    color = self.feedback_colors['excellent']
                elif value >= 0.5:
                    color = self.feedback_colors['good']
                else:
                    color = self.feedback_colors['needs_attention']

                # Format metric name
                display_name = metric_name.replace('_', ' ').title()[:12]  # Truncate long names
                metric_text = f"{display_name}: {value:.2f}"

                cv2.putText(frame, metric_text, (x_start, y_pos), self.font, 0.4, color, 1)

        except Exception as e:
            self.logger.warning(f"Metrics overlay drawing error: {e}")

    def _generate_empty_result(self, frame: np.ndarray) -> Dict[str, Any]:
        """Generate result when no pose is detected"""
        return {
            'pose_embedding': [0.0] * 23,
            'pose_features': {},
            'landmarks': None,
            'segmentation_mask': None,
            'processed_frame': frame,
            'timestamp': time.time(),
            'error': 'No pose detected'
        }

    def _generate_error_result(self, frame: np.ndarray, error_message: str) -> Dict[str, Any]:
        """Generate result when error occurs"""
        return {
            'pose_embedding': [0.0] * 23,
            'pose_features': {},
            'landmarks': None,
            'segmentation_mask': None,
            'processed_frame': frame,
            'timestamp': time.time(),
            'error': error_message
        }

    def get_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose keypoints from frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                h, w, _ = frame.shape
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    x = landmark.x * w
                    y = landmark.y * h
                    z = landmark.z
                    keypoints.append([x, y, z])

                return np.array(keypoints)

            return None

        except Exception as e:
            self.logger.error(f"Keypoint extraction error: {e}")
            return None

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.pose.close()
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")
