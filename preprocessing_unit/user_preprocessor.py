# preprocessing_unit/user_preprocessor.py
"""
User Data Preprocessor (UP)

Handles user data preprocessing tasks:
- Feedback aggregation across sessions
- Historical trend analysis
- Preference extraction
- Data cleaning and validation
"""

import argparse
import json
import re
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class UserDataPreprocessor:
    """
    User data preprocessor for meditation module
    Handles user feedback and session history preprocessing
    """
    
    def __init__(self,
                 lookback_days: int = 30,
                 min_feedback_length: int = 5,
                 max_feedback_length: int = 1000):
        
        self.lookback_days = lookback_days
        self.min_feedback_length = min_feedback_length
        self.max_feedback_length = max_feedback_length
        
        # Text cleaning patterns
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'[\+]?[1-9]?[0-9]{7,12}')
        
        # Emotion keywords for sentiment analysis
        self.emotion_keywords = {
            'positive': [
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                'love', 'like', 'enjoy', 'happy', 'calm', 'peaceful', 'relaxed',
                'better', 'improved', 'helpful', 'beneficial', 'effective'
            ],
            'negative': [
                'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 
                'angry', 'frustrated', 'stressed', 'anxious', 'worried', 'sad',
                'worse', 'difficult', 'hard', 'challenging', 'unhelpful'
            ],
            'neutral': [
                'okay', 'ok', 'fine', 'average', 'normal', 'usual', 'same',
                'moderate', 'medium', 'standard', 'typical'
            ]
        }
        
        # Session quality indicators
        self.quality_indicators = {
            'completion': ['completed', 'finished', 'done', 'full session'],
            'interruption': ['interrupted', 'stopped', 'quit', 'left early', 'distracted'],
            'focus': ['focused', 'concentrated', 'attentive', 'present', 'mindful'],
            'distraction': ['distracted', 'wandering', 'lost focus', 'mind wandering']
        }
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted content and normalizing"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove HTML tags
        text = self.html_pattern.sub(' ', text)
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove email addresses
        text = self.email_pattern.sub(' ', text)
        
        # Remove phone numbers
        text = self.phone_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\'\"]', ' ', text)
        
        # Convert to lowercase and strip
        text = text.lower().strip()
        
        return text
    
    def validate_feedback_entry(self, entry: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single feedback entry"""
        issues = []
        
        # Check required fields
        required_fields = ['userId', 'timestamp', 'feedbackText']
        for field in required_fields:
            if field not in entry or entry[field] is None:
                issues.append(f"Missing required field: {field}")
        
        # Validate feedback text length
        feedback_text = entry.get('feedbackText', '')
        if isinstance(feedback_text, str):
            clean_text = self.clean_text(feedback_text)
            if len(clean_text) < self.min_feedback_length:
                issues.append("Feedback text too short")
            elif len(clean_text) > self.max_feedback_length:
                issues.append("Feedback text too long")
        else:
            issues.append("Feedback text is not a string")
        
        # Validate timestamp
        timestamp = entry.get('timestamp')
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    # Try to parse timestamp
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d %H:%M:%S']:
                        try:
                            datetime.strptime(timestamp, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        issues.append("Invalid timestamp format")
            except Exception:
                issues.append("Cannot parse timestamp")
        
        # Validate rating if present
        rating = entry.get('rating')
        if rating is not None:
            try:
                rating_val = float(rating)
                if not (1.0 <= rating_val <= 5.0):
                    issues.append("Rating should be between 1 and 5")
            except (ValueError, TypeError):
                issues.append("Invalid rating format")
        
        return len(issues) == 0, issues
    
    def extract_sentiment_scores(self, text: str) -> Dict[str, float]:
        """Extract sentiment scores from text"""
        clean_text = self.clean_text(text)
        words = clean_text.split()
        
        if not words:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for word in words:
            for sentiment, keywords in self.emotion_keywords.items():
                if word in keywords:
                    sentiment_counts[sentiment] += 1
                    break
        
        total_sentiment_words = sum(sentiment_counts.values())
        
        if total_sentiment_words == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        # Normalize to probabilities
        sentiment_scores = {
            sentiment: count / total_sentiment_words 
            for sentiment, count in sentiment_counts.items()
        }
        
        return sentiment_scores
    
    def extract_session_quality_indicators(self, text: str) -> Dict[str, float]:
        """Extract session quality indicators from feedback text"""
        clean_text = self.clean_text(text)
        
        quality_scores = {}
        
        for indicator, keywords in self.quality_indicators.items():
            score = 0.0
            for keyword in keywords:
                if keyword in clean_text:
                    score += 1.0
            
            # Normalize by number of keywords
            quality_scores[indicator] = min(1.0, score / len(keywords))
        
        return quality_scores
    
    def analyze_temporal_patterns(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in user feedback"""
        if not entries:
            return {
                'session_frequency': 0.0,
                'preferred_times': [],
                'consistency_score': 0.0,
                'trend_direction': 'stable'
            }
        
        # Sort entries by timestamp
        timestamped_entries = []
        for entry in entries:
            timestamp = entry.get('timestamp')
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                            try:
                                dt = datetime.strptime(timestamp, fmt)
                                timestamped_entries.append((dt, entry))
                                break
                            except ValueError:
                                continue
                except Exception:
                    pass
        
        if not timestamped_entries:
            return {
                'session_frequency': 0.0,
                'preferred_times': [],
                'consistency_score': 0.0,
                'trend_direction': 'stable'
            }
        
        # Sort by timestamp
        timestamped_entries.sort(key=lambda x: x[0])
        
        # Calculate session frequency (sessions per week)
        if len(timestamped_entries) >= 2:
            time_span = (timestamped_entries[-1][0] - timestamped_entries[0][0]).days
            if time_span > 0:
                session_frequency = len(timestamped_entries) * 7.0 / time_span
            else:
                session_frequency = len(timestamped_entries)
        else:
            session_frequency = 1.0
        
        # Analyze preferred times of day
        hour_counts = Counter()
        for dt, _ in timestamped_entries:
            hour_counts[dt.hour] += 1
        
        preferred_times = []
        if hour_counts:
            max_count = max(hour_counts.values())
            for hour, count in hour_counts.items():
                if count == max_count:
                    if 6 <= hour < 12:
                        preferred_times.append('morning')
                    elif 12 <= hour < 18:
                        preferred_times.append('afternoon')
                    else:
                        preferred_times.append('evening')
        
        preferred_times = list(set(preferred_times))
        
        # Calculate consistency (regularity of sessions)
        if len(timestamped_entries) >= 3:
            intervals = []
            for i in range(1, len(timestamped_entries)):
                interval = (timestamped_entries[i][0] - timestamped_entries[i-1][0]).days
                intervals.append(interval)
            
            if intervals:
                avg_interval = np.mean(intervals)
                interval_std = np.std(intervals)
                consistency_score = 1.0 / (1.0 + interval_std / max(avg_interval, 1.0))
            else:
                consistency_score = 1.0
        else:
            consistency_score = 0.5  # Moderate consistency for few sessions
        
        # Analyze rating trend if available
        ratings = []
        for dt, entry in timestamped_entries:
            rating = entry.get('rating')
            if rating is not None:
                try:
                    ratings.append(float(rating))
                except (ValueError, TypeError):
                    pass
        
        if len(ratings) >= 3:
            # Simple linear trend analysis
            x = np.arange(len(ratings))
            coeffs = np.polyfit(x, ratings, 1)
            slope = coeffs[0]
            
            if slope > 0.1:
                trend_direction = 'improving'
            elif slope < -0.1:
                trend_direction = 'declining'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'stable'
        
        return {
            'session_frequency': float(session_frequency),
            'preferred_times': preferred_times,
            'consistency_score': float(consistency_score),
            'trend_direction': trend_direction
        }
    
    def aggregate_feedback_data(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple feedback entries into summary statistics"""
        if not entries:
            return self._empty_aggregation()
        
        # Filter recent entries
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        recent_entries = []
        
        for entry in entries:
            timestamp = entry.get('timestamp')
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        dt = datetime.strptime(timestamp.split()[0], '%Y-%m-%d')
                        if dt >= cutoff_date:
                            recent_entries.append(entry)
                except Exception:
                    recent_entries.append(entry)  # Include if parsing fails
            else:
                recent_entries.append(entry)  # Include if no timestamp
        
        if not recent_entries:
            recent_entries = entries[-min(10, len(entries)):]  # Use last 10 as fallback
        
        # Aggregate text content
        all_feedback_text = []
        sentiment_scores = []
        quality_indicators = []
        ratings = []
        
        for entry in recent_entries:
            # Validate entry
            is_valid, issues = self.validate_feedback_entry(entry)
            if not is_valid:
                continue
            
            feedback_text = entry.get('feedbackText', '')
            if feedback_text:
                clean_text = self.clean_text(feedback_text)
                all_feedback_text.append(clean_text)
                
                # Extract sentiment
                sentiment = self.extract_sentiment_scores(clean_text)
                sentiment_scores.append(sentiment)
                
                # Extract quality indicators
                quality = self.extract_session_quality_indicators(clean_text)
                quality_indicators.append(quality)
            
            # Collect ratings
            rating = entry.get('rating')
            if rating is not None:
                try:
                    ratings.append(float(rating))
                except (ValueError, TypeError):
                    pass
        
        # Aggregate sentiment scores
        if sentiment_scores:
            avg_sentiment = {
                'positive': np.mean([s['positive'] for s in sentiment_scores]),
                'negative': np.mean([s['negative'] for s in sentiment_scores]),
                'neutral': np.mean([s['neutral'] for s in sentiment_scores])
            }
        else:
            avg_sentiment = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        # Aggregate quality indicators
        if quality_indicators:
            avg_quality = {}
            for indicator in self.quality_indicators.keys():
                avg_quality[indicator] = np.mean([q.get(indicator, 0.0) for q in quality_indicators])
        else:
            avg_quality = {indicator: 0.0 for indicator in self.quality_indicators.keys()}
        
        # Calculate rating statistics
        if ratings:
            rating_stats = {
                'mean': float(np.mean(ratings)),
                'std': float(np.std(ratings)),
                'count': len(ratings),
                'trend': self._calculate_rating_trend(ratings)
            }
        else:
            rating_stats = {'mean': 3.0, 'std': 0.0, 'count': 0, 'trend': 'stable'}
        
        # Analyze temporal patterns
        temporal_analysis = self.analyze_temporal_patterns(recent_entries)
        
        # Create combined text for overall analysis
        combined_text = ' '.join(all_feedback_text) if all_feedback_text else ''
        
        return {
            'user_id': recent_entries[0].get('userId', 'unknown') if recent_entries else 'unknown',
            'total_entries': len(entries),
            'recent_entries': len(recent_entries),
            'aggregated_text': combined_text,
            'sentiment_analysis': avg_sentiment,
            'quality_indicators': avg_quality,
            'rating_statistics': rating_stats,
            'temporal_patterns': temporal_analysis,
            'text_statistics': {
                'total_words': len(combined_text.split()),
                'avg_feedback_length': np.mean([len(text.split()) for text in all_feedback_text]) if all_feedback_text else 0,
                'vocabulary_size': len(set(combined_text.split())) if combined_text else 0
            }
        }
    
    def _calculate_rating_trend(self, ratings: List[float]) -> str:
        """Calculate trend direction for ratings"""
        if len(ratings) < 3:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(ratings))
        coeffs = np.polyfit(x, ratings, 1)
        slope = coeffs[0]
        
        if slope > 0.2:
            return 'improving'
        elif slope < -0.2:
            return 'declining'
        else:
            return 'stable'
    
    def _empty_aggregation(self) -> Dict[str, Any]:
        """Return empty aggregation structure"""
        return {
            'user_id': 'unknown',
            'total_entries': 0,
            'recent_entries': 0,
            'aggregated_text': '',
            'sentiment_analysis': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
            'quality_indicators': {indicator: 0.0 for indicator in self.quality_indicators.keys()},
            'rating_statistics': {'mean': 3.0, 'std': 0.0, 'count': 0, 'trend': 'stable'},
            'temporal_patterns': {
                'session_frequency': 0.0,
                'preferred_times': [],
                'consistency_score': 0.0,
                'trend_direction': 'stable'
            },
            'text_statistics': {
                'total_words': 0,
                'avg_feedback_length': 0,
                'vocabulary_size': 0
            }
        }
    
    def process_user_feedback(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Process user feedback data"""
        if isinstance(data, dict):
            # Single user's data
            if 'recent_sessions' in data:
                entries = data['recent_sessions']
            elif 'feedback_entries' in data:
                entries = data['feedback_entries']
            elif 'sessions' in data:
                entries = data['sessions']
            else:
                # Treat the dict itself as a single entry
                entries = [data]
        elif isinstance(data, list):
            # List of feedback entries
            entries = data
        else:
            # Convert to list
            entries = [data]
        
        return self.aggregate_feedback_data(entries)


def main():
    """CLI interface for user data preprocessor"""
    parser = argparse.ArgumentParser(description="User Data Preprocessor for meditation module")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSON file with user feedback data")
    parser.add_argument("--output", type=str, default="preprocess_output/user_feedback_processed.json",
                       help="Output JSON file for processed data")
    parser.add_argument("--lookback-days", type=int, default=30,
                       help="Number of days to look back for recent data")
    parser.add_argument("--min-feedback-length", type=int, default=5,
                       help="Minimum feedback text length in characters")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = UserDataPreprocessor(
        lookback_days=args.lookback_days,
        min_feedback_length=args.min_feedback_length
    )
    
    # Load input data
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with input_path.open("r", encoding="utf-8") as f:
        input_data = json.load(f)
    
    # Process data
    if isinstance(input_data, list) and len(input_data) > 0:
        # Check if it's a list of users or a list of feedback entries
        first_item = input_data[0]
        if isinstance(first_item, dict) and ('userId' in first_item or 'user_id' in first_item):
            # List of users
            results = []
            for user_data in input_data:
                result = preprocessor.process_user_feedback(user_data)
                results.append(result)
        else:
            # Single user with list of feedback entries
            results = [preprocessor.process_user_feedback(input_data)]
    else:
        # Single user data
        results = [preprocessor.process_user_feedback(input_data)]
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"User data preprocessing complete. Processed {len(results)} users.")
    print(f"Results saved to {output_path}")
    
    # Summary statistics
    if results:
        total_entries = sum(r['total_entries'] for r in results)
        avg_sentiment = np.mean([r['sentiment_analysis']['positive'] for r in results])
        avg_rating = np.mean([r['rating_statistics']['mean'] for r in results])
        
        print(f"\nSummary:")
        print(f"Total feedback entries: {total_entries}")
        print(f"Average positive sentiment: {avg_sentiment:.3f}")
        print(f"Average rating: {avg_rating:.2f}")


if __name__ == "__main__":
    main()