
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Tuple, Any, Optional
import time
import math
from scipy.spatial.distance import euclidean
from scipy.stats import zscore
import logging

class PostureDetector:
    """Enhanced posture detector with meditation-specific capabilities"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.session_history = []
        self.reference_posture = self._load_ideal_meditation_posture()
        self.stability_window = []
        self.stability_threshold = 0.02
        self.logger = logging.getLogger(__name__)

        # Meditation-specific weights
        self.meditation_weights = {
            'head_alignment': 0.25,
            'shoulder_level': 0.20,
            'spine_straightness': 0.30,
            'sitting_stability': 0.15,
            'breathing_space': 0.10
        }

    def _load_ideal_meditation_posture(self) -> Dict[str, float]:
        """Load ideal meditation posture reference points"""
        return {
            'head_forward_angle': 15.0,  # degrees
            'shoulder_levelness': 0.95,  # ratio
            'spine_curve_deviation': 10.0,  # degrees
            'hip_stability_threshold': 0.02,
            'chest_openness': 0.8
        }

    def detect_meditation_posture(self, frame: np.ndarray) -> Dict[str, Any]:
        """Enhanced posture detection specifically for meditation"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if not results.pose_landmarks:
                return self._generate_no_detection_result()

            landmarks = self._extract_landmarks_array(results.pose_landmarks)
            posture_metrics = self._assess_meditation_posture(landmarks)
            posture_score = self.calculate_meditation_posture_score(posture_metrics)
            recommendations = self.generate_contextual_feedback(posture_metrics, self.session_history)

            # Update stability tracking
            self._update_stability_tracking(landmarks)

            result = {
                'posture_score': posture_score,
                'detailed_metrics': posture_metrics,
                'recommendations': recommendations,
                'landmarks': landmarks,
                'body_part_scores': self._calculate_body_part_scores(posture_metrics),
                'stability_score': self._calculate_stability_score(),
                'timestamp': time.time()
            }

            # Add to session history
            self.session_history.append(result)
            if len(self.session_history) > 100:  # Keep last 100 measurements
                self.session_history.pop(0)

            return result

        except Exception as e:
            self.logger.error(f"Meditation posture detection error: {e}")
            return self._generate_error_result(str(e))

    def _assess_meditation_posture(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Comprehensive meditation posture assessment"""
        metrics = {}

        # Head alignment assessment
        metrics['head_alignment'] = self._assess_head_alignment(landmarks)

        # Shoulder levelness
        metrics['shoulder_level'] = self._assess_shoulder_levelness(landmarks)

        # Spine straightness
        metrics['spine_straightness'] = self._assess_spine_alignment(landmarks)

        # Sitting stability
        metrics['sitting_stability'] = self._assess_sitting_stability(landmarks)

        # Breathing space (chest openness)
        metrics['breathing_space'] = self._assess_breathing_space(landmarks)

        # Additional meditation-specific metrics
        metrics['forward_lean'] = self._assess_forward_lean(landmarks)
        metrics['shoulder_tension'] = self._assess_shoulder_tension(landmarks)
        metrics['neck_strain'] = self._assess_neck_strain(landmarks)

        return metrics

    def _assess_head_alignment(self, landmarks: np.ndarray) -> float:
        """Assess head alignment for meditation posture"""
        try:
            # Key landmarks for head assessment
            nose = landmarks[0]  # Nose tip
            left_ear = landmarks[7]  # Left ear
            right_ear = landmarks[8]  # Right ear
            neck = landmarks[0]  # Approximation of neck center

            # Calculate head tilt
            ear_midpoint = (left_ear + right_ear) / 2
            head_vector = nose - ear_midpoint

            # Calculate forward head posture
            forward_angle = math.degrees(math.atan2(abs(head_vector[2]), abs(head_vector[1])))

            # Score based on ideal angle (slight forward is natural)
            ideal_angle = self.reference_posture['head_forward_angle']
            deviation = abs(forward_angle - ideal_angle)

            # Convert to score (0-1, where 1 is perfect)
            score = max(0, 1 - (deviation / 45.0))  # Normalize by 45 degrees max deviation

            return score

        except Exception as e:
            self.logger.warning(f"Head alignment assessment error: {e}")
            return 0.5  # Neutral score on error

    def _assess_shoulder_levelness(self, landmarks: np.ndarray) -> float:
        """Assess shoulder levelness"""
        try:
            left_shoulder = landmarks[11]  # Left shoulder
            right_shoulder = landmarks[12]  # Right shoulder

            # Calculate height difference
            height_diff = abs(left_shoulder[1] - right_shoulder[1])
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])

            # Normalize by shoulder width
            levelness_ratio = 1 - (height_diff / shoulder_width) if shoulder_width > 0 else 0

            # Ensure ratio is between 0 and 1
            score = max(0, min(1, levelness_ratio))

            return score

        except Exception as e:
            self.logger.warning(f"Shoulder levelness assessment error: {e}")
            return 0.5

    def _assess_spine_alignment(self, landmarks: np.ndarray) -> float:
        """Assess spine straightness for meditation"""
        try:
            # Key spinal landmarks
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            # Calculate spine line
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2

            spine_vector = shoulder_center - hip_center

            # Calculate deviation from vertical
            vertical_deviation = math.degrees(math.atan2(abs(spine_vector[0]), abs(spine_vector[1])))

            # Score based on straightness
            max_acceptable_deviation = 15.0  # degrees
            score = max(0, 1 - (vertical_deviation / max_acceptable_deviation))

            return score

        except Exception as e:
            self.logger.warning(f"Spine alignment assessment error: {e}")
            return 0.5

    def _assess_sitting_stability(self, landmarks: np.ndarray) -> float:
        """Assess sitting stability"""
        try:
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            # Calculate hip stability over time
            current_hip_center = (left_hip + right_hip) / 2

            if len(self.stability_window) > 0:
                # Calculate movement from previous positions
                recent_positions = [pos for pos, _ in self.stability_window[-10:]]  # Last 10 positions
                if recent_positions:
                    movements = [euclidean(current_hip_center, pos) for pos in recent_positions]
                    avg_movement = sum(movements) / len(movements)

                    # Score based on stability (less movement = higher score)
                    max_movement = 0.05  # Threshold for stable sitting
                    score = max(0, 1 - (avg_movement / max_movement))
                    return score

            return 0.8  # Default good score for first measurements

        except Exception as e:
            self.logger.warning(f"Sitting stability assessment error: {e}")
            return 0.5

    def _assess_breathing_space(self, landmarks: np.ndarray) -> float:
        """Assess chest openness for proper breathing"""
        try:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            # Calculate shoulder width (proxy for chest openness)
            shoulder_width = euclidean(left_shoulder, right_shoulder)

            # Compare to reference measurement (requires calibration)
            # For now, use relative assessment
            if hasattr(self, 'reference_shoulder_width'):
                openness_ratio = shoulder_width / self.reference_shoulder_width
                score = min(1.0, max(0.0, openness_ratio))
            else:
                # First measurement becomes reference
                self.reference_shoulder_width = shoulder_width
                score = 0.8  # Assume good starting position

            return score

        except Exception as e:
            self.logger.warning(f"Breathing space assessment error: {e}")
            return 0.5

    def _assess_forward_lean(self, landmarks: np.ndarray) -> float:
        """Assess forward lean (slouching)"""
        try:
            head_center = landmarks[0]  # Nose as head reference
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            shoulder_center = (left_shoulder + right_shoulder) / 2

            # Calculate forward lean angle
            lean_vector = head_center - shoulder_center
            lean_angle = math.degrees(math.atan2(abs(lean_vector[2]), abs(lean_vector[1])))

            # Ideal slight forward lean
            ideal_lean = 10.0  # degrees
            deviation = abs(lean_angle - ideal_lean)

            score = max(0, 1 - (deviation / 30.0))  # Normalize by 30 degrees
            return score

        except Exception as e:
            self.logger.warning(f"Forward lean assessment error: {e}")
            return 0.5

    def _assess_shoulder_tension(self, landmarks: np.ndarray) -> float:
        """Assess shoulder tension by elevation"""
        try:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_ear = landmarks[7]
            right_ear = landmarks[8]

            # Calculate ear-to-shoulder distance as tension indicator
            left_distance = euclidean(left_shoulder, left_ear)
            right_distance = euclidean(right_shoulder, right_ear)
            avg_distance = (left_distance + right_distance) / 2

            # Higher distance typically indicates relaxed shoulders
            if hasattr(self, 'reference_shoulder_ear_distance'):
                tension_ratio = avg_distance / self.reference_shoulder_ear_distance
                score = min(1.0, max(0.0, tension_ratio))
            else:
                self.reference_shoulder_ear_distance = avg_distance
                score = 0.8

            return score

        except Exception as e:
            self.logger.warning(f"Shoulder tension assessment error: {e}")
            return 0.5

    def _assess_neck_strain(self, landmarks: np.ndarray) -> float:
        """Assess neck strain"""
        try:
            nose = landmarks[0]
            left_ear = landmarks[7]
            right_ear = landmarks[8]
            ear_center = (left_ear + right_ear) / 2

            # Calculate neck angle
            neck_vector = nose - ear_center
            neck_angle = math.degrees(math.atan2(abs(neck_vector[0]), abs(neck_vector[1])))

            # Score based on natural neck position
            ideal_neck_angle = 15.0  # degrees
            deviation = abs(neck_angle - ideal_neck_angle)

            score = max(0, 1 - (deviation / 25.0))
            return score

        except Exception as e:
            self.logger.warning(f"Neck strain assessment error: {e}")
            return 0.5

    def calculate_meditation_posture_score(self, metrics: Dict[str, float]) -> float:
        """Weighted scoring for meditation-specific posture"""
        weighted_score = 0.0
        total_weight = 0.0

        for metric_name, weight in self.meditation_weights.items():
            if metric_name in metrics:
                weighted_score += metrics[metric_name] * weight
                total_weight += weight

        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.5  # Default neutral score

        return max(0.0, min(1.0, final_score))

    def generate_contextual_feedback(self, current_posture: Dict[str, float], 
                                   session_history: List[Dict[str, Any]]) -> List[str]:
        """Generate personalized posture guidance"""
        recommendations = []

        # Analyze current posture issues
        for metric_name, score in current_posture.items():
            if score < 0.6:  # Threshold for needing improvement
                recommendation = self._generate_metric_recommendation(metric_name, score)
                if recommendation:
                    recommendations.append(recommendation)

        # Analyze trends if we have history
        if len(session_history) > 5:
            trends = self._analyze_posture_trends(session_history)
            trend_recommendations = self._generate_trend_recommendations(trends)
            recommendations.extend(trend_recommendations)

        # Prioritize recommendations
        return self._prioritize_recommendations(recommendations)[:3]  # Top 3 recommendations

    def _generate_metric_recommendation(self, metric_name: str, score: float) -> str:
        """Generate recommendation for specific metric"""
        recommendations_map = {
            'head_alignment': "Gently tuck your chin and align your head over your shoulders",
            'shoulder_level': "Relax your shoulders and ensure they're level - drop any tension",
            'spine_straightness': "Lengthen your spine, imagine a string pulling you up from the crown of your head",
            'sitting_stability': "Find your sitting bones and establish a stable, grounded base",
            'breathing_space': "Open your chest gently, allow space for natural breathing",
            'forward_lean': "Sit back slightly and engage your core to maintain upright posture",
            'shoulder_tension': "Take a deep breath and let your shoulders drop naturally",
            'neck_strain': "Soften your neck and find a natural, comfortable head position"
        }

        return recommendations_map.get(metric_name, f"Focus on improving your {metric_name}")

    def _analyze_posture_trends(self, history: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze posture trends over time"""
        trends = {}

        if len(history) < 5:
            return trends

        # Get recent scores
        recent_scores = history[-10:]  # Last 10 measurements

        for metric in self.meditation_weights.keys():
            scores = [h['detailed_metrics'].get(metric, 0.5) for h in recent_scores 
                     if 'detailed_metrics' in h and metric in h['detailed_metrics']]

            if len(scores) >= 3:
                # Calculate trend
                early_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
                late_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)

                if late_avg > early_avg + 0.1:
                    trends[metric] = "improving"
                elif late_avg < early_avg - 0.1:
                    trends[metric] = "declining"
                else:
                    trends[metric] = "stable"

        return trends

    def _generate_trend_recommendations(self, trends: Dict[str, str]) -> List[str]:
        """Generate recommendations based on trends"""
        recommendations = []

        declining_metrics = [metric for metric, trend in trends.items() if trend == "declining"]
        improving_metrics = [metric for metric, trend in trends.items() if trend == "improving"]

        if declining_metrics:
            recommendations.append(f"Pay attention to your {declining_metrics[0]} - it's been declining")

        if improving_metrics:
            recommendations.append(f"Great progress on your {improving_metrics[0]} - keep it up!")

        return recommendations

    def _prioritize_recommendations(self, recommendations: List[str]) -> List[str]:
        """Prioritize recommendations by importance"""
        # Simple prioritization - could be enhanced with ML
        priority_keywords = ['spine', 'head', 'shoulder', 'breathing']

        prioritized = []
        remaining = []

        for rec in recommendations:
            if any(keyword in rec.lower() for keyword in priority_keywords):
                prioritized.append(rec)
            else:
                remaining.append(rec)

        return prioritized + remaining

    def _calculate_body_part_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate scores for different body parts"""
        return {
            'head_neck': (metrics.get('head_alignment', 0.5) + metrics.get('neck_strain', 0.5)) / 2,
            'shoulders': (metrics.get('shoulder_level', 0.5) + metrics.get('shoulder_tension', 0.5)) / 2,
            'spine': metrics.get('spine_straightness', 0.5),
            'core': metrics.get('sitting_stability', 0.5),
            'chest': metrics.get('breathing_space', 0.5)
        }

    def _update_stability_tracking(self, landmarks: np.ndarray):
        """Update stability tracking window"""
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        hip_center = (left_hip + right_hip) / 2

        current_time = time.time()
        self.stability_window.append((hip_center, current_time))

        # Keep only last 5 seconds of data
        cutoff_time = current_time - 5.0
        self.stability_window = [(pos, t) for pos, t in self.stability_window if t > cutoff_time]

    def _calculate_stability_score(self) -> float:
        """Calculate overall stability score"""
        if len(self.stability_window) < 3:
            return 0.8  # Default good score for insufficient data

        positions = [pos for pos, _ in self.stability_window]
        movements = []

        for i in range(1, len(positions)):
            movement = euclidean(positions[i], positions[i-1])
            movements.append(movement)

        if movements:
            avg_movement = sum(movements) / len(movements)
            # Convert to score (less movement = higher score)
            max_stable_movement = 0.02
            score = max(0, 1 - (avg_movement / max_stable_movement))
            return min(1.0, score)

        return 0.8

    def _extract_landmarks_array(self, pose_landmarks) -> np.ndarray:
        """Extract landmarks as numpy array"""
        landmarks = []
        for landmark in pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)

    def _generate_no_detection_result(self) -> Dict[str, Any]:
        """Generate result when no pose is detected"""
        return {
            'posture_score': 0.0,
            'detailed_metrics': {},
            'recommendations': ["Please ensure you're visible in the camera frame"],
            'landmarks': None,
            'body_part_scores': {},
            'stability_score': 0.0,
            'timestamp': time.time(),
            'error': 'No pose detected'
        }

    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate result when error occurs"""
        return {
            'posture_score': 0.0,
            'detailed_metrics': {},
            'recommendations': ["Technical issue occurred, please try again"],
            'landmarks': None,
            'body_part_scores': {},
            'stability_score': 0.0,
            'timestamp': time.time(),
            'error': error_message
        }

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process a record for compatibility with existing system"""
        try:
            # Extract frame from embeddings if available
            if 'embeddings' in record and 'pose' in record['embeddings']:
                # This would need adaptation based on actual data format
                pose_data = record['embeddings']['pose']
                # Convert to frame format if needed
                # For now, return basic processing result

                return {
                    'posture_score': 0.7,  # Placeholder
                    'recommendations': ["Maintain good posture"],
                    'body_part_scores': {
                        'head_neck': 0.7,
                        'shoulders': 0.8,
                        'spine': 0.6,
                        'core': 0.7,
                        'chest': 0.8
                    }
                }

            return self._generate_no_detection_result()

        except Exception as e:
            return self._generate_error_result(str(e))

class EnhancedPostureDetector(PostureDetector):
    """Extended class with meditation-specific features"""

    def __init__(self, meditation_specific=True):
        super().__init__()
        self.meditation_specific = meditation_specific
        self.posture_calibration = {}
        self.session_start_time = None

    def start_session(self):
        """Start a new meditation session"""
        self.session_start_time = time.time()
        self.session_history.clear()
        self.stability_window.clear()

    def end_session(self) -> Dict[str, Any]:
        """End session and return summary"""
        if not self.session_history:
            return {'error': 'No session data available'}

        session_duration = time.time() - (self.session_start_time or time.time())

        # Calculate session averages
        avg_score = sum(h['posture_score'] for h in self.session_history) / len(self.session_history)

        # Calculate improvement trend
        if len(self.session_history) >= 10:
            early_scores = [h['posture_score'] for h in self.session_history[:5]]
            late_scores = [h['posture_score'] for h in self.session_history[-5:]]
            improvement = (sum(late_scores) / len(late_scores)) - (sum(early_scores) / len(early_scores))
        else:
            improvement = 0

        # Generate session summary
        summary = {
            'session_duration': session_duration,
            'average_posture_score': avg_score,
            'improvement_trend': improvement,
            'total_measurements': len(self.session_history),
            'common_recommendations': self._get_common_recommendations(),
            'best_score': max(h['posture_score'] for h in self.session_history),
            'stability_average': sum(h.get('stability_score', 0) for h in self.session_history) / len(self.session_history)
        }

        return summary

    def _get_common_recommendations(self) -> List[str]:
        """Get most common recommendations from session"""
        all_recommendations = []
        for history_item in self.session_history:
            all_recommendations.extend(history_item.get('recommendations', []))

        # Count frequency of similar recommendations
        recommendation_counts = {}
        for rec in all_recommendations:
            # Simplified grouping by first word
            key = rec.split()[0].lower() if rec else 'unknown'
            recommendation_counts[key] = recommendation_counts.get(key, 0) + 1

        # Return most common
        if recommendation_counts:
            most_common = max(recommendation_counts.items(), key=lambda x: x[1])
            return [f"Most frequent advice: focus on {most_common[0]}"]

        return []
