# Encoders/user_profile_encoder.py
"""
User Profile Encoder (UE)

Processes user data to extract:
- Preference embeddings
- Progress vectors
- Behavioral patterns
- Session history analysis
"""

import argparse
import json
import math
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional imports with fallbacks
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class UserProfileEncoder:
    """
    User Profile Encoder for meditation module
    Extracts comprehensive user embeddings from profile and session data
    """
    
    def __init__(self,
                 embedding_dim: int = 64,
                 lookback_days: int = 30,
                 min_sessions: int = 3):
        
        self.embedding_dim = embedding_dim
        self.lookback_days = lookback_days
        self.min_sessions = min_sessions
        
        # Initialize scalers
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            self.min_max_scaler = MinMaxScaler()
            self.pca = PCA(n_components=min(32, embedding_dim))
        else:
            self.scaler = None
            self.min_max_scaler = None
            self.pca = None
        
        # Meditation type categories for preference analysis
        self.meditation_categories = {
            'mindfulness': ['mindfulness meditation', 'mindfulness', 'awareness', 'present moment'],
            'breathing': ['breathwork', 'breath awareness', 'breathing', 'pranayama'],
            'body': ['body scan', 'progressive relaxation', 'body awareness', 'somatic'],
            'movement': ['movement meditation', 'walking meditation', 'yoga', 'tai chi'],
            'visualization': ['visualization', 'guided imagery', 'mental imagery'],
            'loving_kindness': ['loving-kindness', 'compassion', 'metta', 'self-compassion'],
            'mantra': ['mantra meditation', 'chanting', 'repetition'],
            'sound': ['sound meditation', 'singing bowls', 'music therapy'],
            'focus': ['focused attention', 'concentration', 'single-pointed focus'],
            'open_awareness': ['open monitoring', 'choiceless awareness', 'open awareness']
        }
        
        # Session quality indicators
        self.quality_indicators = [
            'completion_rate', 'posture_score', 'engagement_level', 
            'feedback_rating', 'duration_adherence', 'consistency'
        ]
        
        # Behavioral pattern categories
        self.behavior_patterns = [
            'morning_preference', 'evening_preference', 'consistency_score',
            'duration_preference', 'difficulty_preference', 'social_vs_solo'
        ]
    
    def extract_basic_demographics(self, profile_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract basic demographic and preference features"""
        features = {}
        
        # Age group (encoded)
        age = profile_data.get('age', 0)
        features['age_normalized'] = min(age / 100.0, 1.0) if age > 0 else 0.5
        features['age_group_young'] = 1.0 if 18 <= age <= 30 else 0.0
        features['age_group_middle'] = 1.0 if 31 <= age <= 50 else 0.0
        features['age_group_senior'] = 1.0 if age > 50 else 0.0
        
        # Experience level
        experience = profile_data.get('meditation_experience', 'beginner').lower()
        features['exp_beginner'] = 1.0 if experience == 'beginner' else 0.0
        features['exp_intermediate'] = 1.0 if experience == 'intermediate' else 0.0
        features['exp_advanced'] = 1.0 if experience == 'advanced' else 0.0
        
        # Goals and motivations
        goals = profile_data.get('goals', [])
        if isinstance(goals, str):
            goals = [goals]
        
        goal_categories = {
            'stress_reduction': ['stress', 'anxiety', 'calm', 'relax'],
            'focus_improvement': ['focus', 'concentration', 'attention', 'clarity'],
            'sleep_improvement': ['sleep', 'insomnia', 'rest', 'bedtime'],
            'emotional_regulation': ['emotion', 'mood', 'anger', 'depression'],
            'spiritual_growth': ['spiritual', 'enlightenment', 'awareness', 'growth'],
            'health_wellness': ['health', 'wellness', 'healing', 'recovery']
        }
        
        for category, keywords in goal_categories.items():
            features[f'goal_{category}'] = 0.0
            for goal in goals:
                if any(keyword in str(goal).lower() for keyword in keywords):
                    features[f'goal_{category}'] = 1.0
                    break
        
        # Availability and schedule preferences
        availability = profile_data.get('availability', {})
        if isinstance(availability, dict):
            features['morning_available'] = 1.0 if availability.get('morning', False) else 0.0
            features['afternoon_available'] = 1.0 if availability.get('afternoon', False) else 0.0
            features['evening_available'] = 1.0 if availability.get('evening', False) else 0.0
            features['flexible_schedule'] = sum([features['morning_available'], 
                                               features['afternoon_available'], 
                                               features['evening_available']]) / 3.0
        else:
            features['morning_available'] = features['afternoon_available'] = features['evening_available'] = 0.5
            features['flexible_schedule'] = 0.5
        
        # Preferred duration
        pref_duration = profile_data.get('preferred_duration_minutes', 15)
        features['duration_normalized'] = min(pref_duration / 60.0, 1.0)  # Normalize to [0,1]
        features['short_sessions'] = 1.0 if pref_duration <= 10 else 0.0
        features['medium_sessions'] = 1.0 if 11 <= pref_duration <= 25 else 0.0
        features['long_sessions'] = 1.0 if pref_duration > 25 else 0.0
        
        return features
    
    def analyze_session_history(self, sessions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze user's session history for behavioral patterns"""
        if not sessions:
            return {f'session_{indicator}': 0.0 for indicator in self.quality_indicators + self.behavior_patterns}
        
        features = {}
        
        # Filter recent sessions
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        recent_sessions = []
        
        for session in sessions:
            session_date = session.get('date', session.get('timestamp', ''))
            if session_date:
                try:
                    if isinstance(session_date, str):
                        # Try different date formats
                        for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d', '%m/%d/%Y']:
                            try:
                                parsed_date = datetime.strptime(session_date.split()[0], fmt)
                                if parsed_date >= cutoff_date:
                                    recent_sessions.append(session)
                                break
                            except ValueError:
                                continue
                except:
                    recent_sessions.append(session)  # Include if date parsing fails
            else:
                recent_sessions.append(session)  # Include if no date
        
        if not recent_sessions:
            recent_sessions = sessions[-min(10, len(sessions)):]  # Use last 10 sessions as fallback
        
        # Session completion analysis
        completed_sessions = [s for s in recent_sessions if s.get('completed', True)]
        features['completion_rate'] = len(completed_sessions) / len(recent_sessions)
        
        # Posture quality analysis
        posture_scores = [s.get('posture_score', 0.5) for s in recent_sessions if 'posture_score' in s]
        features['avg_posture_score'] = np.mean(posture_scores) if posture_scores else 0.5
        features['posture_consistency'] = 1.0 - np.std(posture_scores) if len(posture_scores) > 1 else 0.5
        
        # Engagement analysis
        engagement_scores = [s.get('engagement_level', 0.5) for s in recent_sessions if 'engagement_level' in s]
        features['avg_engagement'] = np.mean(engagement_scores) if engagement_scores else 0.5
        
        # Feedback analysis
        ratings = [s.get('feedback_rating', 3.0) for s in recent_sessions if 'feedback_rating' in s]
        features['avg_rating'] = np.mean(ratings) / 5.0 if ratings else 0.6  # Normalize to [0,1]
        features['rating_consistency'] = 1.0 - (np.std(ratings) / 5.0) if len(ratings) > 1 else 0.5
        
        # Duration adherence
        planned_durations = [s.get('planned_duration', 15) for s in recent_sessions]
        actual_durations = [s.get('actual_duration', s.get('planned_duration', 15)) for s in recent_sessions]
        
        if planned_durations and actual_durations:
            adherence_ratios = [min(actual/planned, 2.0) for actual, planned in zip(actual_durations, planned_durations) if planned > 0]
            features['duration_adherence'] = np.mean([min(ratio, 1.0) for ratio in adherence_ratios]) if adherence_ratios else 0.5
        else:
            features['duration_adherence'] = 0.5
        
        # Time preference analysis
        session_times = []
        for session in recent_sessions:
            timestamp = session.get('timestamp', session.get('date', ''))
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        # Extract time from timestamp
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%H:%M:%S', '%H:%M']:
                            try:
                                dt = datetime.strptime(timestamp, fmt)
                                session_times.append(dt.hour)
                                break
                            except ValueError:
                                continue
                except:
                    pass
        
        if session_times:
            morning_sessions = sum(1 for h in session_times if 5 <= h < 12)
            afternoon_sessions = sum(1 for h in session_times if 12 <= h < 18)
            evening_sessions = sum(1 for h in session_times if 18 <= h <= 23 or 0 <= h < 5)
            
            total = len(session_times)
            features['morning_preference'] = morning_sessions / total
            features['afternoon_preference'] = afternoon_sessions / total
            features['evening_preference'] = evening_sessions / total
        else:
            features['morning_preference'] = features['afternoon_preference'] = features['evening_preference'] = 0.33
        
        # Consistency analysis (regularity of sessions)
        if len(recent_sessions) >= 3:
            session_dates = []
            for session in recent_sessions:
                date_str = session.get('date', session.get('timestamp', ''))
                if date_str:
                    try:
                        for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S']:
                            try:
                                date_obj = datetime.strptime(date_str.split()[0], fmt)
                                session_dates.append(date_obj)
                                break
                            except ValueError:
                                continue
                    except:
                        pass
            
            if len(session_dates) >= 3:
                session_dates.sort()
                intervals = [(session_dates[i+1] - session_dates[i]).days for i in range(len(session_dates)-1)]
                avg_interval = np.mean(intervals)
                interval_std = np.std(intervals)
                
                # Consistency score: higher when intervals are more regular
                features['consistency_score'] = 1.0 / (1.0 + interval_std / max(avg_interval, 1.0))
            else:
                features['consistency_score'] = 0.3  # Low consistency for irregular sessions
        else:
            features['consistency_score'] = 0.1  # Very low for insufficient data
        
        # Preferred meditation types
        meditation_types = [s.get('meditation_type', '').lower() for s in recent_sessions if s.get('meditation_type')]
        type_preferences = {}
        
        if meditation_types:
            type_counts = Counter(meditation_types)
            total_sessions = len(meditation_types)
            
            for category, keywords in self.meditation_categories.items():
                category_count = 0
                for med_type in meditation_types:
                    if any(keyword in med_type for keyword in keywords):
                        category_count += 1
                type_preferences[f'pref_{category}'] = category_count / total_sessions
        else:
            for category in self.meditation_categories.keys():
                type_preferences[f'pref_{category}'] = 0.1  # Small default preference
        
        features.update(type_preferences)
        
        return features
    
    def calculate_progress_metrics(self, sessions: List[Dict[str, Any]], profile_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate user progress and improvement metrics"""
        if len(sessions) < self.min_sessions:
            return {
                'overall_progress': 0.0,
                'posture_improvement': 0.0,
                'consistency_trend': 0.0,
                'engagement_trend': 0.0,
                'duration_progress': 0.0,
                'goal_achievement': 0.0
            }
        
        # Sort sessions by date/timestamp
        sorted_sessions = sorted(sessions, key=lambda x: x.get('date', x.get('timestamp', '0')))
        
        metrics = {}
        
        # Posture improvement over time
        posture_scores = [s.get('posture_score', 0.5) for s in sorted_sessions if 'posture_score' in s]
        if len(posture_scores) >= 3:
            # Linear trend analysis
            x = np.arange(len(posture_scores))
            slope = np.polyfit(x, posture_scores, 1)[0] if len(posture_scores) > 1 else 0.0
            metrics['posture_improvement'] = max(0.0, min(1.0, slope + 0.5))  # Normalize to [0,1]
        else:
            metrics['posture_improvement'] = 0.5
        
        # Engagement trend
        engagement_scores = [s.get('engagement_level', 0.5) for s in sorted_sessions if 'engagement_level' in s]
        if len(engagement_scores) >= 3:
            x = np.arange(len(engagement_scores))
            slope = np.polyfit(x, engagement_scores, 1)[0] if len(engagement_scores) > 1 else 0.0
            metrics['engagement_trend'] = max(0.0, min(1.0, slope + 0.5))
        else:
            metrics['engagement_trend'] = 0.5
        
        # Consistency trend (sessions per week over time)
        if len(sorted_sessions) >= 4:
            # Group sessions by week
            weekly_counts = defaultdict(int)
            for session in sorted_sessions:
                date_str = session.get('date', session.get('timestamp', ''))
                if date_str:
                    try:
                        date_obj = datetime.strptime(date_str.split()[0], '%Y-%m-%d')
                        week = date_obj.isocalendar()[1]  # ISO week number
                        weekly_counts[week] += 1
                    except:
                        pass
            
            if len(weekly_counts) >= 3:
                weekly_values = list(weekly_counts.values())
                x = np.arange(len(weekly_values))
                slope = np.polyfit(x, weekly_values, 1)[0] if len(weekly_values) > 1 else 0.0
                metrics['consistency_trend'] = max(0.0, min(1.0, (slope + 2.0) / 4.0))  # Normalize
            else:
                metrics['consistency_trend'] = 0.5
        else:
            metrics['consistency_trend'] = 0.3
        
        # Duration progress (ability to maintain longer sessions)
        planned_durations = [s.get('planned_duration', 15) for s in sorted_sessions]
        actual_durations = [s.get('actual_duration', planned) for s, planned in zip(sorted_sessions, planned_durations)]
        
        if len(actual_durations) >= 3:
            x = np.arange(len(actual_durations))
            slope = np.polyfit(x, actual_durations, 1)[0] if len(actual_durations) > 1 else 0.0
            metrics['duration_progress'] = max(0.0, min(1.0, (slope + 10.0) / 20.0))  # Normalize
        else:
            metrics['duration_progress'] = 0.5
        
        # Goal achievement estimation (based on improvement in relevant metrics)
        goals = profile_data.get('goals', [])
        goal_progress = []
        
        for goal in goals if isinstance(goals, list) else [goals]:
            goal_str = str(goal).lower()
            if any(keyword in goal_str for keyword in ['stress', 'anxiety', 'calm']):
                # For stress/anxiety goals, look at engagement and consistency
                progress = (metrics.get('engagement_trend', 0.5) + metrics.get('consistency_trend', 0.5)) / 2.0
                goal_progress.append(progress)
            elif any(keyword in goal_str for keyword in ['focus', 'attention', 'concentration']):
                # For focus goals, emphasize posture and duration
                progress = (metrics.get('posture_improvement', 0.5) + metrics.get('duration_progress', 0.5)) / 2.0
                goal_progress.append(progress)
            elif any(keyword in goal_str for keyword in ['sleep', 'rest']):
                # For sleep goals, consistency is key
                progress = metrics.get('consistency_trend', 0.5)
                goal_progress.append(progress)
            else:
                # General progress
                progress = np.mean(list(metrics.values()))
                goal_progress.append(progress)
        
        metrics['goal_achievement'] = np.mean(goal_progress) if goal_progress else 0.5
        
        # Overall progress score
        metrics['overall_progress'] = np.mean([
            metrics['posture_improvement'],
            metrics['engagement_trend'],
            metrics['consistency_trend'],
            metrics['duration_progress'],
            metrics['goal_achievement']
        ])
        
        return metrics
    
    def identify_behavioral_patterns(self, sessions: List[Dict[str, Any]], profile_data: Dict[str, Any]) -> Dict[str, float]:
        """Identify key behavioral patterns in user data"""
        patterns = {}
        
        if not sessions:
            return {pattern: 0.0 for pattern in self.behavior_patterns}
        
        # Analyze session timing patterns
        session_hours = []
        session_days = []
        
        for session in sessions:
            timestamp = session.get('timestamp', session.get('date', ''))
            if timestamp:
                try:
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d']:
                        try:
                            dt = datetime.strptime(timestamp, fmt)
                            session_hours.append(dt.hour)
                            session_days.append(dt.weekday())  # 0=Monday, 6=Sunday
                            break
                        except ValueError:
                            continue
                except:
                    pass
        
        if session_hours:
            # Time of day preferences
            morning_count = sum(1 for h in session_hours if 5 <= h < 12)
            evening_count = sum(1 for h in session_hours if 18 <= h <= 23 or 0 <= h < 5)
            
            patterns['morning_preference'] = morning_count / len(session_hours)
            patterns['evening_preference'] = evening_count / len(session_hours)
            
            # Time consistency (prefer same times?)
            hour_std = np.std(session_hours)
            patterns['time_consistency'] = 1.0 / (1.0 + hour_std / 12.0)  # Normalize
        else:
            patterns['morning_preference'] = patterns['evening_preference'] = 0.33
            patterns['time_consistency'] = 0.5
        
        if session_days:
            # Day of week consistency
            day_counts = Counter(session_days)
            # Check if user prefers weekdays vs weekends
            weekday_count = sum(count for day, count in day_counts.items() if day < 5)
            weekend_count = sum(count for day, count in day_counts.items() if day >= 5)
            
            patterns['weekday_preference'] = weekday_count / len(session_days) if session_days else 0.5
            patterns['weekend_preference'] = weekend_count / len(session_days) if session_days else 0.5
        else:
            patterns['weekday_preference'] = patterns['weekend_preference'] = 0.5
        
        # Duration consistency pattern
        durations = [s.get('actual_duration', s.get('planned_duration', 15)) for s in sessions]
        if durations:
            duration_std = np.std(durations)
            patterns['duration_consistency'] = 1.0 / (1.0 + duration_std / np.mean(durations))
            
            # Preferred duration range
            avg_duration = np.mean(durations)
            patterns['short_duration_preference'] = 1.0 if avg_duration <= 10 else 0.0
            patterns['medium_duration_preference'] = 1.0 if 10 < avg_duration <= 25 else 0.0
            patterns['long_duration_preference'] = 1.0 if avg_duration > 25 else 0.0
        else:
            patterns['duration_consistency'] = 0.5
            patterns['short_duration_preference'] = patterns['medium_duration_preference'] = patterns['long_duration_preference'] = 0.33
        
        # Completion pattern
        completion_rates = []
        if len(sessions) >= 5:
            # Look at completion rate over time windows
            window_size = max(3, len(sessions) // 3)
            for i in range(len(sessions) - window_size + 1):
                window_sessions = sessions[i:i+window_size]
                completed = sum(1 for s in window_sessions if s.get('completed', True))
                completion_rates.append(completed / window_size)
        
        if completion_rates:
            patterns['completion_consistency'] = 1.0 - np.std(completion_rates)
            patterns['high_completion_tendency'] = 1.0 if np.mean(completion_rates) > 0.8 else 0.0
        else:
            patterns['completion_consistency'] = 0.5
            patterns['high_completion_tendency'] = 0.5
        
        # Feedback pattern analysis
        ratings = [s.get('feedback_rating', 3.0) for s in sessions if 'feedback_rating' in s]
        if ratings:
            rating_std = np.std(ratings)
            patterns['rating_consistency'] = 1.0 / (1.0 + rating_std)
            patterns['positive_feedback_tendency'] = 1.0 if np.mean(ratings) > 3.5 else 0.0
        else:
            patterns['rating_consistency'] = 0.5
            patterns['positive_feedback_tendency'] = 0.5
        
        return patterns
    
    def create_user_embedding(self, profile_data: Dict[str, Any], sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive user profile embedding"""
        # Extract different feature sets
        demographic_features = self.extract_basic_demographics(profile_data)
        session_features = self.analyze_session_history(sessions)
        progress_metrics = self.calculate_progress_metrics(sessions, profile_data)
        behavioral_patterns = self.identify_behavioral_patterns(sessions, profile_data)
        
        # Combine all features
        all_features = {}
        all_features.update(demographic_features)
        all_features.update(session_features)
        all_features.update(progress_metrics)
        all_features.update(behavioral_patterns)
        
        # Convert to array and normalize
        feature_vector = np.array(list(all_features.values()), dtype=np.float32)
        
        # Handle NaN and inf values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Apply scaling if available
        if self.scaler is not None and len(feature_vector) > 10:
            try:
                # Fit scaler on current data (in production, this would be pre-fitted)
                scaled_features = self.scaler.fit_transform(feature_vector.reshape(1, -1))
                feature_vector = scaled_features.flatten()
            except Exception as e:
                print(f"Scaling failed: {e}")
                # Fallback to manual normalization
                feature_vector = np.clip(feature_vector, 0.0, 1.0)
        else:
            # Manual normalization to [0, 1] range
            feature_vector = np.clip(feature_vector, 0.0, 1.0)
        
        # Apply PCA for dimensionality reduction if needed
        if self.pca is not None and len(feature_vector) > self.embedding_dim:
            try:
                reduced_features = self.pca.fit_transform(feature_vector.reshape(1, -1))
                final_embedding = reduced_features.flatten()
            except Exception as e:
                print(f"PCA failed: {e}")
                # Truncate to desired size
                final_embedding = feature_vector[:self.embedding_dim]
        else:
            # Pad or truncate to desired embedding size
            if len(feature_vector) < self.embedding_dim:
                # Pad with zeros
                final_embedding = np.pad(feature_vector, (0, self.embedding_dim - len(feature_vector)))
            else:
                # Truncate
                final_embedding = feature_vector[:self.embedding_dim]
        
        # Create user segments/clusters
        user_segment = self._determine_user_segment(all_features)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(sessions, profile_data)
        
        return {
            'embedding': final_embedding.tolist(),
            'feature_breakdown': {
                'demographics': demographic_features,
                'session_analysis': session_features,
                'progress_metrics': progress_metrics,
                'behavioral_patterns': behavioral_patterns
            },
            'user_segment': user_segment,
            'confidence_scores': confidence_scores,
            'total_sessions': len(sessions),
            'profile_completeness': self._calculate_profile_completeness(profile_data),
            'recommendations': self._generate_user_recommendations(all_features, sessions)
        }
    
    def _determine_user_segment(self, features: Dict[str, float]) -> str:
        """Determine user segment based on key features"""
        # Simple rule-based segmentation (could be replaced with ML clustering)
        
        # Check experience level
        if features.get('exp_beginner', 0) > 0:
            if features.get('consistency_score', 0) > 0.7:
                return 'committed_beginner'
            else:
                return 'casual_beginner'
        elif features.get('exp_intermediate', 0) > 0:
            if features.get('overall_progress', 0) > 0.6:
                return 'progressing_intermediate'
            else:
                return 'stable_intermediate'
        elif features.get('exp_advanced', 0) > 0:
            return 'advanced_practitioner'
        
        # Fallback based on behavior
        consistency = features.get('consistency_score', 0)
        engagement = features.get('avg_engagement', 0)
        
        if consistency > 0.7 and engagement > 0.7:
            return 'dedicated_user'
        elif consistency > 0.4 or engagement > 0.4:
            return 'regular_user'
        else:
            return 'occasional_user'
    
    def _calculate_confidence_scores(self, sessions: List[Dict[str, Any]], profile_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence in different aspects of the user model"""
        confidence = {}
        
        # Data quantity confidence
        confidence['data_quantity'] = min(1.0, len(sessions) / 20.0)  # Full confidence at 20+ sessions
        
        # Profile completeness confidence
        profile_fields = ['age', 'meditation_experience', 'goals', 'availability', 'preferred_duration_minutes']
        completed_fields = sum(1 for field in profile_fields if profile_data.get(field) is not None)
        confidence['profile_completeness'] = completed_fields / len(profile_fields)
        
        # Recency confidence (more recent data = higher confidence)
        if sessions:
            latest_session = max(sessions, key=lambda x: x.get('date', x.get('timestamp', '0')))
            latest_date_str = latest_session.get('date', latest_session.get('timestamp', ''))
            
            try:
                latest_date = datetime.strptime(latest_date_str.split()[0], '%Y-%m-%d')
                days_since = (datetime.now() - latest_date).days
                confidence['recency'] = max(0.0, 1.0 - days_since / 60.0)  # Decay over 60 days
            except:
                confidence['recency'] = 0.5
        else:
            confidence['recency'] = 0.0
        
        # Consistency confidence (consistent users are more predictable)
        consistency_score = 0.0
        for session in sessions:
            if 'posture_score' in session and 'engagement_level' in session:
                consistency_score += 1.0
        
        confidence['data_consistency'] = consistency_score / len(sessions) if sessions else 0.0
        
        # Overall confidence
        confidence['overall'] = np.mean(list(confidence.values()))
        
        return confidence
    
    def _calculate_profile_completeness(self, profile_data: Dict[str, Any]) -> float:
        """Calculate how complete the user profile is"""
        required_fields = [
            'age', 'meditation_experience', 'goals', 'availability', 
            'preferred_duration_minutes', 'timezone'
        ]
        
        optional_fields = [
            'name', 'occupation', 'health_conditions', 'medications',
            'previous_meditation_types', 'preferences'
        ]
        
        required_score = sum(1 for field in required_fields if profile_data.get(field) is not None) / len(required_fields)
        optional_score = sum(1 for field in optional_fields if profile_data.get(field) is not None) / len(optional_fields)
        
        return 0.8 * required_score + 0.2 * optional_score
    
    def _generate_user_recommendations(self, features: Dict[str, float], sessions: List[Dict[str, Any]]) -> List[str]:
        """Generate personalized recommendations based on user profile"""
        recommendations = []
        
        # Consistency recommendations
        if features.get('consistency_score', 0) < 0.4:
            recommendations.append("Try to establish a regular meditation schedule")
        
        # Duration recommendations
        avg_completion = features.get('completion_rate', 1.0)
        if avg_completion < 0.7:
            recommendations.append("Consider shorter sessions to improve completion rate")
        
        # Posture recommendations
        if features.get('avg_posture_score', 0.5) < 0.6:
            recommendations.append("Focus on improving posture during meditation")
        
        # Time-based recommendations
        if features.get('morning_preference', 0) > 0.6:
            recommendations.append("Morning sessions work well for you - keep it up!")
        elif features.get('evening_preference', 0) > 0.6:
            recommendations.append("Evening sessions suit your schedule well")
        
        # Progress-based recommendations
        if features.get('overall_progress', 0) > 0.7:
            recommendations.append("Great progress! Consider exploring advanced techniques")
        elif features.get('overall_progress', 0) < 0.3:
            recommendations.append("Focus on building a consistent foundation")
        
        # Goal-specific recommendations
        if features.get('goal_stress_reduction', 0) > 0:
            recommendations.append("Incorporate more breathing exercises for stress relief")
        
        if features.get('goal_focus_improvement', 0) > 0:
            recommendations.append("Try focused attention meditations to improve concentration")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def process_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process complete user data package"""
        profile_data = user_data.get('profile', {})
        sessions = user_data.get('sessions', [])
        
        if not isinstance(sessions, list):
            sessions = []
        
        # Create embedding and analysis
        result = self.create_user_embedding(profile_data, sessions)
        
        # Add metadata
        result.update({
            'user_id': user_data.get('user_id', 'unknown'),
            'created_at': datetime.now().isoformat(),
            'model_version': '1.0',
            'embedding_dimensions': self.embedding_dim
        })
        
        return result
    
    def _empty_result(self, user_id: str = 'unknown', error: str = None) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'embedding': [0.0] * self.embedding_dim,
            'feature_breakdown': {
                'demographics': {},
                'session_analysis': {},
                'progress_metrics': {},
                'behavioral_patterns': {}
            },
            'user_segment': 'unknown',
            'confidence_scores': {'overall': 0.0},
            'total_sessions': 0,
            'profile_completeness': 0.0,
            'recommendations': ['Complete your profile to get personalized recommendations'],
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'model_version': '1.0',
            'embedding_dimensions': self.embedding_dim,
            'error': error
        }


def main():
    """CLI interface for user profile encoder"""
    parser = argparse.ArgumentParser(description="User Profile Encoder for meditation module")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSON file with user data")
    parser.add_argument("--output", type=str, default="preprocess_output/user_profile_encoded.json",
                       help="Output JSON file")
    parser.add_argument("--embedding-dim", type=int, default=64,
                       help="Embedding dimensionality")
    parser.add_argument("--lookback-days", type=int, default=30,
                       help="Days to look back for session analysis")
    
    args = parser.parse_args()
    
    # Initialize encoder
    encoder = UserProfileEncoder(
        embedding_dim=args.embedding_dim,
        lookback_days=args.lookback_days
    )
    
    # Load input data
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with input_path.open("r", encoding="utf-8") as f:
        input_data = json.load(f)
    
    # Process data
    if isinstance(input_data, list):
        results = []
        for user_data in input_data:
            result = encoder.process_user_data(user_data)
            results.append(result)
    else:
        results = [encoder.process_user_data(input_data)]
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"User profile encoding complete. Processed {len(results)} users.")
    print(f"Results saved to {output_path}")
    
    # Summary statistics
    if results:
        segments = [r.get('user_segment', 'unknown') for r in results]
        avg_confidence = np.mean([r.get('confidence_scores', {}).get('overall', 0) for r in results])
        avg_completeness = np.mean([r.get('profile_completeness', 0) for r in results])
        
        print(f"\nSummary:")
        print(f"User segments: {dict(zip(*np.unique(segments, return_counts=True)))}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Average profile completeness: {avg_completeness:.3f}")


if __name__ == "__main__":
    main()