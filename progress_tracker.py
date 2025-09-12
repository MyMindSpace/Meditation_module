# progress_tracker.py
"""
Progress Tracker

Tracks user progress and learning patterns:
- Session history analysis
- Progress metrics calculation
- Personalized insights generation
- Adaptive goal setting
"""

import json
import math
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np


class ProgressMetric(Enum):
    """Types of progress metrics"""
    POSTURE_IMPROVEMENT = "posture_improvement"
    CONSISTENCY = "consistency"
    DURATION_TOLERANCE = "duration_tolerance"
    ENGAGEMENT = "engagement"
    FOCUS_QUALITY = "focus_quality"
    STRESS_REDUCTION = "stress_reduction"
    SKILL_MASTERY = "skill_mastery"


@dataclass
class ProgressDataPoint:
    """Single progress measurement"""
    timestamp: datetime
    session_id: str
    metric_type: ProgressMetric
    value: float
    session_duration: float
    meditation_type: str
    user_rating: Optional[float] = None
    notes: str = ""


@dataclass
class ProgressTrend:
    """Progress trend analysis"""
    metric: ProgressMetric
    current_value: float
    trend_direction: str  # 'improving', 'stable', 'declining'
    trend_strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    time_span_days: int
    data_points: int
    next_milestone: Optional[float] = None


@dataclass
class ProgressInsight:
    """Generated insight about user progress"""
    insight_type: str
    title: str
    description: str
    confidence: float
    actionable_recommendation: str
    metric_evidence: List[ProgressMetric]
    timestamp: datetime


@dataclass
class UserGoal:
    """User-defined or system-generated goal"""
    goal_id: str
    goal_type: str
    target_metric: ProgressMetric
    target_value: float
    target_date: datetime
    created_date: datetime
    description: str
    is_active: bool = True
    progress_percentage: float = 0.0


class ProgressTracker:
    """
    Comprehensive progress tracking system for meditation practice
    """
    
    def __init__(self, 
                 user_id: str,
                 lookback_days: int = 90,
                 min_sessions_for_trend: int = 5,
                 confidence_threshold: float = 0.6):
        
        self.user_id = user_id
        self.lookback_days = lookback_days
        self.min_sessions_for_trend = min_sessions_for_trend
        self.confidence_threshold = confidence_threshold
        
        # Data storage
        self.progress_data: List[ProgressDataPoint] = []
        self.insights_history: List[ProgressInsight] = []
        self.user_goals: List[UserGoal] = []
        
        # Progress baselines (learned from early sessions)
        self.baselines: Dict[ProgressMetric, float] = {}
        self.baseline_established = False
        
        # Adaptive parameters
        self.learning_rate = 0.1
        self.adaptation_sensitivity = 0.7
        
        # Load existing data if available
        self._load_progress_data()
    
    def add_session_data(self, 
                        session_id: str,
                        session_metrics: Dict[str, Any],
                        user_feedback: Optional[Dict[str, Any]] = None) -> None:
        """Add session data and calculate progress metrics"""
        
        timestamp = datetime.now()
        
        # Extract session information
        duration = session_metrics.get('duration', 0.0)
        meditation_type = session_metrics.get('meditation_type', 'unknown')
        user_rating = session_metrics.get('user_rating')
        
        # Calculate and store various progress metrics
        progress_points = []
        
        # Posture improvement
        posture_score = session_metrics.get('average_posture_score', 0.0)
        if posture_score > 0:
            progress_points.append(ProgressDataPoint(
                timestamp=timestamp,
                session_id=session_id,
                metric_type=ProgressMetric.POSTURE_IMPROVEMENT,
                value=posture_score,
                session_duration=duration,
                meditation_type=meditation_type,
                user_rating=user_rating
            ))
        
        # Engagement score
        engagement = session_metrics.get('engagement_score', 0.0)
        if engagement > 0:
            progress_points.append(ProgressDataPoint(
                timestamp=timestamp,
                session_id=session_id,
                metric_type=ProgressMetric.ENGAGEMENT,
                value=engagement,
                session_duration=duration,
                meditation_type=meditation_type,
                user_rating=user_rating
            ))
        
        # Duration tolerance (completion rate as proxy)
        completion_rate = session_metrics.get('completion_rate', 0.0)
        progress_points.append(ProgressDataPoint(
            timestamp=timestamp,
            session_id=session_id,
            metric_type=ProgressMetric.DURATION_TOLERANCE,
            value=completion_rate,
            session_duration=duration,
            meditation_type=meditation_type,
            user_rating=user_rating
        ))
        
        # Focus quality (inverse of interruptions)
        interruptions = session_metrics.get('interruptions', 0)
        focus_score = max(0.0, 1.0 - interruptions * 0.2)  # Each interruption reduces focus by 20%
        progress_points.append(ProgressDataPoint(
            timestamp=timestamp,
            session_id=session_id,
            metric_type=ProgressMetric.FOCUS_QUALITY,
            value=focus_score,
            session_duration=duration,
            meditation_type=meditation_type,
            user_rating=user_rating
        ))
        
        # Add user feedback derived metrics
        if user_feedback:
            stress_reduction = self._calculate_stress_reduction(user_feedback)
            if stress_reduction is not None:
                progress_points.append(ProgressDataPoint(
                    timestamp=timestamp,
                    session_id=session_id,
                    metric_type=ProgressMetric.STRESS_REDUCTION,
                    value=stress_reduction,
                    session_duration=duration,
                    meditation_type=meditation_type,
                    user_rating=user_rating
                ))
        
        # Store progress points
        self.progress_data.extend(progress_points)
        
        # Calculate consistency (needs historical data)
        consistency_score = self._calculate_consistency_score()
        if consistency_score is not None:
            self.progress_data.append(ProgressDataPoint(
                timestamp=timestamp,
                session_id=session_id,
                metric_type=ProgressMetric.CONSISTENCY,
                value=consistency_score,
                session_duration=duration,
                meditation_type=meditation_type,
                user_rating=user_rating
            ))
        
        # Establish baselines if not done yet
        if not self.baseline_established:
            self._establish_baselines()
        
        # Save data
        self._save_progress_data()
        
        # Generate new insights
        self._generate_insights()
    
    def get_progress_summary(self, days: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive progress summary"""
        if days is None:
            days = self.lookback_days
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_data = [dp for dp in self.progress_data if dp.timestamp >= cutoff_date]
        
        if not recent_data:
            return self._empty_summary()
        
        summary = {
            'user_id': self.user_id,
            'time_period_days': days,
            'total_sessions': len(set(dp.session_id for dp in recent_data)),
            'trends': {},
            'current_scores': {},
            'improvements': {},
            'insights': self._get_recent_insights(days=7),
            'goals_progress': self._calculate_goals_progress(),
            'recommendations': []
        }
        
        # Calculate trends for each metric
        for metric in ProgressMetric:
            trend = self._calculate_trend(metric, days)
            if trend:
                summary['trends'][metric.value] = asdict(trend)
                summary['current_scores'][metric.value] = trend.current_value
                
                # Calculate improvement from baseline
                baseline = self.baselines.get(metric, trend.current_value)
                improvement = ((trend.current_value - baseline) / max(baseline, 0.1)) * 100
                summary['improvements'][metric.value] = improvement
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations()
        
        return summary
    
    def get_detailed_analysis(self, metric: ProgressMetric, days: int = 30) -> Dict[str, Any]:
        """Get detailed analysis for specific metric"""
        cutoff_date = datetime.now() - timedelta(days=days)
        metric_data = [dp for dp in self.progress_data 
                      if dp.metric_type == metric and dp.timestamp >= cutoff_date]
        
        if not metric_data:
            return {'error': f'No data for metric {metric.value}'}
        
        # Sort by timestamp
        metric_data.sort(key=lambda x: x.timestamp)
        
        values = [dp.value for dp in metric_data]
        timestamps = [dp.timestamp for dp in metric_data]
        
        analysis = {
            'metric': metric.value,
            'time_period_days': days,
            'data_points': len(metric_data),
            'current_value': values[-1],
            'min_value': min(values),
            'max_value': max(values),
            'mean_value': np.mean(values),
            'std_dev': np.std(values),
            'trend': self._calculate_trend(metric, days),
            'correlation_with_rating': self._calculate_rating_correlation(metric_data),
            'best_sessions': self._find_best_sessions(metric_data),
            'patterns': self._detect_patterns(metric_data)
        }
        
        return analysis
    
    def set_user_goal(self, goal_type: str, target_metric: ProgressMetric, 
                     target_value: float, target_date: datetime, description: str) -> str:
        """Set new user goal"""
        goal_id = f"goal_{len(self.user_goals)}_{int(datetime.now().timestamp())}"
        
        goal = UserGoal(
            goal_id=goal_id,
            goal_type=goal_type,
            target_metric=target_metric,
            target_value=target_value,
            target_date=target_date,
            created_date=datetime.now(),
            description=description
        )
        
        self.user_goals.append(goal)
        self._save_progress_data()
        
        return goal_id
    
    def update_goal_progress(self) -> None:
        """Update progress on all active goals"""
        for goal in self.user_goals:
            if not goal.is_active:
                continue
            
            # Get current value for target metric
            recent_data = self._get_recent_metric_data(goal.target_metric, days=7)
            if recent_data:
                current_value = np.mean([dp.value for dp in recent_data])
                baseline = self.baselines.get(goal.target_metric, 0.0)
                
                # Calculate progress percentage
                if goal.target_value > baseline:
                    progress = ((current_value - baseline) / (goal.target_value - baseline)) * 100
                else:
                    progress = ((baseline - current_value) / (baseline - goal.target_value)) * 100
                
                goal.progress_percentage = max(0.0, min(100.0, progress))
                
                # Mark as completed if target reached
                if ((goal.target_value > baseline and current_value >= goal.target_value) or
                    (goal.target_value < baseline and current_value <= goal.target_value)):
                    goal.is_active = False
    
    def _calculate_stress_reduction(self, user_feedback: Dict[str, Any]) -> Optional[float]:
        """Calculate stress reduction from user feedback"""
        # Look for stress-related feedback
        feedback_text = user_feedback.get('feedback_text', '').lower()
        stress_keywords = ['stress', 'anxious', 'calm', 'relaxed', 'peaceful', 'tension']
        
        positive_keywords = ['calm', 'relaxed', 'peaceful', 'better', 'reduced']
        negative_keywords = ['stress', 'anxious', 'tense', 'worried', 'overwhelmed']
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in feedback_text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in feedback_text)
        
        if positive_count > 0 or negative_count > 0:
            total_words = len(feedback_text.split())
            stress_reduction = (positive_count - negative_count) / max(total_words * 0.1, 1.0)
            return max(0.0, min(1.0, stress_reduction + 0.5))  # Normalize to 0-1
        
        return None
    
    def _calculate_consistency_score(self) -> Optional[float]:
        """Calculate consistency score based on session frequency"""
        if len(self.progress_data) < 7:  # Need at least a week of data
            return None
        
        # Get session dates from last 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_sessions = set()
        
        for dp in self.progress_data:
            if dp.timestamp >= cutoff_date:
                session_date = dp.timestamp.date()
                recent_sessions.add(session_date)
        
        if not recent_sessions:
            return 0.0
        
        # Calculate consistency based on regularity
        session_dates = sorted(recent_sessions)
        if len(session_dates) < 3:
            return 0.3  # Low consistency with few sessions
        
        # Calculate intervals between sessions
        intervals = [(session_dates[i+1] - session_dates[i]).days 
                    for i in range(len(session_dates)-1)]
        
        if not intervals:
            return 0.5
        
        # Consistency is higher when intervals are more regular
        mean_interval = np.mean(intervals)
        interval_std = np.std(intervals)
        
        # Good consistency: sessions every 1-3 days with low variation
        consistency = 1.0 / (1.0 + interval_std / max(mean_interval, 1.0))
        
        # Bonus for frequent sessions
        if mean_interval <= 2.0:
            consistency *= 1.2
        elif mean_interval >= 7.0:
            consistency *= 0.8
        
        return min(1.0, consistency)
    
    def _establish_baselines(self) -> None:
        """Establish baseline values from first few sessions"""
        if len(self.progress_data) < 10:  # Need minimum data
            return
        
        # Use first 20% of data as baseline period
        baseline_count = max(5, len(self.progress_data) // 5)
        baseline_data = self.progress_data[:baseline_count]
        
        # Calculate baseline for each metric
        metric_values = defaultdict(list)
        for dp in baseline_data:
            metric_values[dp.metric_type].append(dp.value)
        
        for metric, values in metric_values.items():
            if len(values) >= 3:  # Need minimum samples
                self.baselines[metric] = np.mean(values)
        
        self.baseline_established = True
    
    def _calculate_trend(self, metric: ProgressMetric, days: int) -> Optional[ProgressTrend]:
        """Calculate trend for specific metric"""
        cutoff_date = datetime.now() - timedelta(days=days)
        metric_data = [dp for dp in self.progress_data 
                      if dp.metric_type == metric and dp.timestamp >= cutoff_date]
        
        if len(metric_data) < self.min_sessions_for_trend:
            return None
        
        # Sort by timestamp
        metric_data.sort(key=lambda x: x.timestamp)
        
        values = np.array([dp.value for dp in metric_data])
        timestamps = np.array([(dp.timestamp - metric_data[0].timestamp).total_seconds() 
                              for dp in metric_data])
        
        # Calculate linear trend
        if len(values) > 1:
            slope, intercept = np.polyfit(timestamps, values, 1)
            current_value = float(values[-1])
            
            # Determine trend direction and strength
            if abs(slope) < 0.001:  # Very small slope
                direction = 'stable'
                strength = 0.0
            elif slope > 0:
                direction = 'improving'
                strength = min(1.0, abs(slope) * (timestamps[-1] - timestamps[0]) / max(current_value, 0.1))
            else:
                direction = 'declining'
                strength = min(1.0, abs(slope) * (timestamps[-1] - timestamps[0]) / max(current_value, 0.1))
            
            # Calculate confidence based on R-squared
            if len(values) > 2:
                correlation = np.corrcoef(timestamps, values)[0, 1]
                confidence = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                confidence = 0.5
            
            # Calculate next milestone
            next_milestone = None
            if direction == 'improving' and current_value < 1.0:
                # Next 10% improvement
                milestone_increase = min(0.1, (1.0 - current_value) / 2)
                next_milestone = current_value + milestone_increase
            elif direction == 'declining' and current_value > 0.0:
                # Next 10% decline to watch for
                milestone_decrease = min(0.1, current_value / 2)
                next_milestone = current_value - milestone_decrease
            
            return ProgressTrend(
                metric=metric,
                current_value=current_value,
                trend_direction=direction,
                trend_strength=strength,
                confidence=confidence,
                time_span_days=days,
                data_points=len(metric_data),
                next_milestone=next_milestone
            )
        
        return None
    
    def _generate_insights(self) -> None:
        """Generate new insights based on recent progress"""
        new_insights = []
        
        # Analyze recent trends
        for metric in ProgressMetric:
            trend = self._calculate_trend(metric, days=14)  # 2 week trend
            
            if trend and trend.confidence >= self.confidence_threshold:
                insight = self._create_trend_insight(trend)
                if insight:
                    new_insights.append(insight)
        
        # Analyze goal progress
        goal_insights = self._analyze_goal_progress()
        new_insights.extend(goal_insights)
        
        # Detect patterns
        pattern_insights = self._detect_progress_patterns()
        new_insights.extend(pattern_insights)
        
        # Add new insights
        self.insights_history.extend(new_insights)
        
        # Keep only recent insights
        cutoff_date = datetime.now() - timedelta(days=30)
        self.insights_history = [insight for insight in self.insights_history 
                               if insight.timestamp >= cutoff_date]
    
    def _create_trend_insight(self, trend: ProgressTrend) -> Optional[ProgressInsight]:
        """Create insight from trend analysis"""
        if trend.trend_direction == 'improving' and trend.trend_strength > 0.3:
            return ProgressInsight(
                insight_type='positive_trend',
                title=f'Great improvement in {trend.metric.value.replace("_", " ").title()}',
                description=f'Your {trend.metric.value.replace("_", " ")} has been steadily improving over the past {trend.time_span_days} days. Current score: {trend.current_value:.2f}',
                confidence=trend.confidence,
                actionable_recommendation=f'Keep up the great work! Consider increasing session frequency to accelerate progress.',
                metric_evidence=[trend.metric],
                timestamp=datetime.now()
            )
        elif trend.trend_direction == 'declining' and trend.trend_strength > 0.3:
            return ProgressInsight(
                insight_type='concerning_trend',
                title=f'{trend.metric.value.replace("_", " ").title()} needs attention',
                description=f'Your {trend.metric.value.replace("_", " ")} has been declining over the past {trend.time_span_days} days. Current score: {trend.current_value:.2f}',
                confidence=trend.confidence,
                actionable_recommendation=f'Focus on this area in upcoming sessions. Consider shorter, more frequent practices.',
                metric_evidence=[trend.metric],
                timestamp=datetime.now()
            )
        elif trend.trend_direction == 'stable' and trend.current_value > 0.8:
            return ProgressInsight(
                insight_type='mastery_achieved',
                title=f'Excellent {trend.metric.value.replace("_", " ").title()} mastery',
                description=f'You\'ve maintained consistently high {trend.metric.value.replace("_", " ")} scores ({trend.current_value:.2f}) - showing mastery of this skill.',
                confidence=trend.confidence,
                actionable_recommendation=f'Consider exploring advanced techniques or helping others develop this skill.',
                metric_evidence=[trend.metric],
                timestamp=datetime.now()
            )
        
        return None
    
    def _analyze_goal_progress(self) -> List[ProgressInsight]:
        """Analyze progress on user goals"""
        insights = []
        
        for goal in self.user_goals:
            if not goal.is_active:
                continue
            
            if goal.progress_percentage >= 80:
                insights.append(ProgressInsight(
                    insight_type='goal_near_completion',
                    title=f'Almost reached your {goal.goal_type} goal!',
                    description=f'You\'re {goal.progress_percentage:.0f}% of the way to your goal: {goal.description}',
                    confidence=0.9,
                    actionable_recommendation='Keep up the consistent practice - you\'re so close!',
                    metric_evidence=[goal.target_metric],
                    timestamp=datetime.now()
                ))
            elif goal.progress_percentage < 20 and (datetime.now() - goal.created_date).days > 14:
                insights.append(ProgressInsight(
                    insight_type='goal_needs_attention',
                    title=f'{goal.goal_type} goal progress is slow',
                    description=f'Only {goal.progress_percentage:.0f}% progress on: {goal.description}',
                    confidence=0.8,
                    actionable_recommendation='Consider breaking this goal into smaller milestones or adjusting your approach.',
                    metric_evidence=[goal.target_metric],
                    timestamp=datetime.now()
                ))
        
        return insights
    
    def _detect_progress_patterns(self) -> List[ProgressInsight]:
        """Detect interesting patterns in progress data"""
        insights = []
        
        # Check for breakthrough moments (sudden improvements)
        for metric in ProgressMetric:
            breakthrough = self._detect_breakthrough(metric)
            if breakthrough:
                insights.append(breakthrough)
        
        # Check for plateau detection
        plateaus = self._detect_plateaus()
        insights.extend(plateaus)
        
        return insights
    
    def _detect_breakthrough(self, metric: ProgressMetric) -> Optional[ProgressInsight]:
        """Detect sudden improvements (breakthroughs)"""
        recent_data = self._get_recent_metric_data(metric, days=7)
        older_data = self._get_recent_metric_data(metric, days=21, offset=7)
        
        if len(recent_data) < 3 or len(older_data) < 3:
            return None
        
        recent_avg = np.mean([dp.value for dp in recent_data])
        older_avg = np.mean([dp.value for dp in older_data])
        
        improvement = (recent_avg - older_avg) / max(older_avg, 0.1)
        
        if improvement > 0.3:  # 30% improvement
            return ProgressInsight(
                insight_type='breakthrough',
                title=f'Breakthrough in {metric.value.replace("_", " ").title()}!',
                description=f'You\'ve made a significant jump in {metric.value.replace("_", " ")} - {improvement*100:.0f}% improvement in just one week!',
                confidence=0.8,
                actionable_recommendation='Analyze what you did differently this week and try to replicate it.',
                metric_evidence=[metric],
                timestamp=datetime.now()
            )
        
        return None
    
    def _detect_plateaus(self) -> List[ProgressInsight]:
        """Detect progress plateaus"""
        insights = []
        
        for metric in ProgressMetric:
            data = self._get_recent_metric_data(metric, days=21)
            if len(data) < 10:
                continue
            
            values = [dp.value for dp in data]
            recent_std = np.std(values[-7:]) if len(values) >= 7 else np.std(values)
            
            # Check for low variation (plateau)
            if recent_std < 0.05 and len(values) >= 7:
                current_avg = np.mean(values[-7:])
                
                if current_avg < 0.7:  # Plateau at suboptimal level
                    insights.append(ProgressInsight(
                        insight_type='plateau_detected',
                        title=f'{metric.value.replace("_", " ").title()} plateau detected',
                        description=f'Your {metric.value.replace("_", " ")} has stabilized at {current_avg:.2f} but could improve further.',
                        confidence=0.7,
                        actionable_recommendation='Try varying your meditation techniques or session length to break through this plateau.',
                        metric_evidence=[metric],
                        timestamp=datetime.now()
                    ))
        
        return insights
    
    def _get_recent_metric_data(self, metric: ProgressMetric, days: int, offset: int = 0) -> List[ProgressDataPoint]:
        """Get recent data for specific metric"""
        end_date = datetime.now() - timedelta(days=offset)
        start_date = end_date - timedelta(days=days)
        
        return [dp for dp in self.progress_data 
                if dp.metric_type == metric and start_date <= dp.timestamp <= end_date]
    
    def _calculate_rating_correlation(self, metric_data: List[ProgressDataPoint]) -> float:
        """Calculate correlation between metric values and user ratings"""
        values = []
        ratings = []
        
        for dp in metric_data:
            if dp.user_rating is not None:
                values.append(dp.value)
                ratings.append(dp.user_rating)
        
        if len(values) < 3:
            return 0.0
        
        correlation = np.corrcoef(values, ratings)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _find_best_sessions(self, metric_data: List[ProgressDataPoint], top_n: int = 3) -> List[Dict[str, Any]]:
        """Find best performing sessions for a metric"""
        sorted_data = sorted(metric_data, key=lambda x: x.value, reverse=True)
        
        best_sessions = []
        for dp in sorted_data[:top_n]:
            best_sessions.append({
                'session_id': dp.session_id,
                'value': dp.value,
                'date': dp.timestamp.strftime('%Y-%m-%d'),
                'meditation_type': dp.meditation_type,
                'duration': dp.session_duration
            })
        
        return best_sessions
    
    def _detect_patterns(self, metric_data: List[ProgressDataPoint]) -> Dict[str, Any]:
        """Detect patterns in metric data"""
        if len(metric_data) < 7:
            return {}
        
        # Time of day pattern
        morning_sessions = [dp for dp in metric_data if 5 <= dp.timestamp.hour < 12]
        evening_sessions = [dp for dp in metric_data if 18 <= dp.timestamp.hour <= 23]
        
        patterns = {}
        
        if morning_sessions and evening_sessions:
            morning_avg = np.mean([dp.value for dp in morning_sessions])
            evening_avg = np.mean([dp.value for dp in evening_sessions])
            
            if morning_avg > evening_avg + 0.1:
                patterns['time_preference'] = 'morning_better'
            elif evening_avg > morning_avg + 0.1:
                patterns['time_preference'] = 'evening_better'
        
        # Duration pattern
        short_sessions = [dp for dp in metric_data if dp.session_duration < 900]  # < 15 min
        long_sessions = [dp for dp in metric_data if dp.session_duration > 1500]  # > 25 min
        
        if short_sessions and long_sessions:
            short_avg = np.mean([dp.value for dp in short_sessions])
            long_avg = np.mean([dp.value for dp in long_sessions])
            
            if long_avg > short_avg + 0.1:
                patterns['duration_preference'] = 'longer_better'
            elif short_avg > long_avg + 0.1:
                patterns['duration_preference'] = 'shorter_better'
        
        return patterns
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on recent insights
        recent_insights = self._get_recent_insights(days=7)
        for insight in recent_insights:
            if insight.actionable_recommendation:
                recommendations.append(insight.actionable_recommendation)
        
        # Based on goal progress
        slow_goals = [g for g in self.user_goals 
                     if g.is_active and g.progress_percentage < 25 
                     and (datetime.now() - g.created_date).days > 7]
        
        if slow_goals:
            recommendations.append("Consider breaking your goals into smaller, more achievable milestones.")
        
        # Based on consistency
        consistency_trend = self._calculate_trend(ProgressMetric.CONSISTENCY, days=21)
        if consistency_trend and consistency_trend.current_value < 0.5:
            recommendations.append("Focus on building a regular meditation routine - even 5 minutes daily helps.")
        
        return recommendations[:5]  # Limit to top 5
    
    def _get_recent_insights(self, days: int = 7) -> List[ProgressInsight]:
        """Get recent insights"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [insight for insight in self.insights_history if insight.timestamp >= cutoff_date]
    
    def _calculate_goals_progress(self) -> Dict[str, Any]:
        """Calculate overall progress on goals"""
        active_goals = [g for g in self.user_goals if g.is_active]
        
        if not active_goals:
            return {'active_goals': 0}
        
        avg_progress = np.mean([g.progress_percentage for g in active_goals])
        completed_goals = len([g for g in self.user_goals if not g.is_active and g.progress_percentage >= 100])
        
        return {
            'active_goals': len(active_goals),
            'completed_goals': completed_goals,
            'average_progress': avg_progress,
            'goals_on_track': len([g for g in active_goals if g.progress_percentage >= 30])
        }
    
    def _empty_summary(self) -> Dict[str, Any]:
        """Return empty summary structure"""
        return {
            'user_id': self.user_id,
            'time_period_days': 0,
            'total_sessions': 0,
            'trends': {},
            'current_scores': {},
            'improvements': {},
            'insights': [],
            'goals_progress': {'active_goals': 0},
            'recommendations': ['Start with regular short sessions to begin tracking your progress.']
        }
    
    def _load_progress_data(self) -> None:
        """Load progress data from file"""
        data_file = Path(f"progress_data_{self.user_id}.json")
        if data_file.exists():
            try:
                with data_file.open('r') as f:
                    data = json.load(f)
                
                # Load progress data points
                for dp_data in data.get('progress_data', []):
                    dp = ProgressDataPoint(
                        timestamp=datetime.fromisoformat(dp_data['timestamp']),
                        session_id=dp_data['session_id'],
                        metric_type=ProgressMetric(dp_data['metric_type']),
                        value=dp_data['value'],
                        session_duration=dp_data['session_duration'],
                        meditation_type=dp_data['meditation_type'],
                        user_rating=dp_data.get('user_rating'),
                        notes=dp_data.get('notes', '')
                    )
                    self.progress_data.append(dp)
                
                # Load baselines
                baselines_data = data.get('baselines', {})
                for metric_str, value in baselines_data.items():
                    self.baselines[ProgressMetric(metric_str)] = value
                
                self.baseline_established = data.get('baseline_established', False)
                
            except Exception as e:
                print(f"Error loading progress data: {e}")
    
    def _save_progress_data(self) -> None:
        """Save progress data to file"""
        data_file = Path(f"progress_data_{self.user_id}.json")
        
        try:
            data = {
                'progress_data': [],
                'baselines': {},
                'baseline_established': self.baseline_established
            }
            
            # Save progress data points
            for dp in self.progress_data:
                dp_data = {
                    'timestamp': dp.timestamp.isoformat(),
                    'session_id': dp.session_id,
                    'metric_type': dp.metric_type.value,
                    'value': dp.value,
                    'session_duration': dp.session_duration,
                    'meditation_type': dp.meditation_type,
                    'user_rating': dp.user_rating,
                    'notes': dp.notes
                }
                data['progress_data'].append(dp_data)
            
            # Save baselines
            for metric, value in self.baselines.items():
                data['baselines'][metric.value] = value
            
            with data_file.open('w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving progress data: {e}")


# Example usage
def demo_progress_tracking():
    """Demonstrate progress tracking functionality"""
    
    tracker = ProgressTracker(user_id="demo_user_001")
    
    # Simulate adding session data over time
    for i in range(20):
        session_metrics = {
            'duration': 600 + np.random.randint(-120, 180),  # ~10 min sessions with variation
            'meditation_type': np.random.choice(['mindfulness', 'breathing', 'body_scan']),
            'average_posture_score': 0.6 + 0.3 * np.random.random() + i * 0.01,  # Gradual improvement
            'engagement_score': 0.7 + 0.2 * np.random.random() + i * 0.005,
            'completion_rate': 0.8 + 0.15 * np.random.random() + i * 0.002,
            'interruptions': max(0, int(3 - i * 0.1 + np.random.randint(-1, 2))),  # Decreasing interruptions
            'user_rating': 3.0 + 1.5 * np.random.random() + i * 0.02
        }
        
        user_feedback = {
            'feedback_text': 'Feeling more calm and relaxed after this session.'
        }
        
        tracker.add_session_data(f"session_{i:03d}", session_metrics, user_feedback)
    
    # Get progress summary
    summary = tracker.get_progress_summary()
    print("Progress Summary:")
    print(f"Total sessions: {summary['total_sessions']}")
    print(f"Current scores: {summary['current_scores']}")
    print(f"Recent insights: {len(summary['insights'])}")
    
    # Set a goal
    goal_date = datetime.now() + timedelta(days=30)
    tracker.set_user_goal(
        goal_type="posture_improvement",
        target_metric=ProgressMetric.POSTURE_IMPROVEMENT,
        target_value=0.9,
        target_date=goal_date,
        description="Achieve excellent posture consistently"
    )
    
    # Get detailed analysis
    posture_analysis = tracker.get_detailed_analysis(ProgressMetric.POSTURE_IMPROVEMENT)
    print(f"\nPosture Analysis:")
    print(f"Current value: {posture_analysis['current_value']:.2f}")
    print(f"Trend: {posture_analysis['trend'].trend_direction if posture_analysis['trend'] else 'N/A'}")
    
    return tracker


if __name__ == "__main__":
    demo_progress_tracking()