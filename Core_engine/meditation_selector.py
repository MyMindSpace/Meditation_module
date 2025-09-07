import json
import argparse
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

# Default input/output paths for direct run (no CLI)
DEFAULT_FEEDBACK_PATH = "preprocess_output/user_feedback_processed.json"
DEFAULT_DIAGNOSIS_PATH = "preprocess_output/diagnosis_processed.json"
DEFAULT_OUTPUT_PATH = "meditation_recommendations_output.json"

@dataclass
class MeditationRecommendation:
    """Data class for meditation recommendations"""
    meditation_type: str
    confidence: float
    rationale: str
    source: str  # 'rule_based' or 'ml_enhanced'

class MeditationSelectorModule:
    """
    A hybrid meditation selector that combines rule-based logic with ML enhancement
    """
    
    def __init__(self, use_ml: bool = True):
        # Rule-based mappings from the documentation
        self.disorder_mappings = {
            'major_depressive_disorder': {
                'meditations': ['Mindfulness-Based Interventions', 'Body-Mind Techniques', 'Loving-Kindness', 'Visualization'],
                'rationale': 'Encourage nonjudgmental awareness, positive emotions, self-compassion, and gentle movement'
            },
            'generalized_anxiety_disorder': {
                'meditations': ['Mindfulness Meditation', 'Breathwork', 'Body Scan', 'Guided Grounding Techniques', 'Focused Attention'],
                'rationale': 'Reduces physiological arousal, increases present-moment focus, interrupts rumination'
            },
            'ptsd': {
                'meditations': ['Mindfulness Meditation', 'Transcendental Meditation', 'Sudarshan Kriya Yoga', 'Visualization'],
                'rationale': 'Calms hyperarousal, improves emotional regulation, processes traumatic memory with non-reaction'
            },
            'substance_use_disorder': {
                'meditations': ['Mindfulness-Based Relapse Prevention', 'Acceptance and Commitment Therapy', 'Movement Meditation'],
                'rationale': 'Reduces craving, emotional distress; improves impulse control'
            },
            'adhd': {
                'meditations': ['Focused Attention Meditation', 'Movement Meditation', 'Body Scan', 'Breath Awareness'],
                'rationale': 'Trains mind to return to anchor, improves concentration, offers physical engagement'
            },
            'obsessive_thoughts': {
                'meditations': ['Noting Meditation', 'Mindfulness', 'Mantra Meditation', 'Compassion'],
                'rationale': 'Loosens attachment to repetitive thoughts, builds awareness'
            }
        }
        
        self.emotion_mappings = {
            'overwhelm': {
                'meditations': ['Grounding Meditation', 'Body Scan', 'Breath Awareness', '5-4-3-2-1 Technique'],
                'rationale': 'Anchors the mind in the body and present moment'
            },
            'stress': {
                'meditations': ['Grounding Meditation', 'Body Scan', 'Breath Awareness', '5-4-3-2-1 Technique'],
                'rationale': 'Anchors the mind in the body and present moment'
            },
            'anger': {
                'meditations': ['Compassion', 'Loving-Kindness', 'Reflection'],
                'rationale': 'Promotes empathy, self-soothing, and insight'
            },
            'sadness': {
                'meditations': ['Loving-Kindness', 'Mindfulness', 'Guided Imagery'],
                'rationale': 'Generates positive affect, strengthens connection'
            },
            'isolation': {
                'meditations': ['Loving-Kindness', 'Mindfulness', 'Guided Imagery'],
                'rationale': 'Generates positive affect, strengthens connection'
            },
            'worry': {
                'meditations': ['Grounding Techniques', 'Breath Awareness', 'Visualization'],
                'rationale': 'Activates parasympathetic system, reduces arousal'
            },
            'fear': {
                'meditations': ['Grounding Techniques', 'Breath Awareness', 'Visualization'],
                'rationale': 'Activates parasympathetic system, reduces arousal'
            },
            'panic': {
                'meditations': ['Grounding Techniques', 'Breath Awareness', 'Visualization'],
                'rationale': 'Activates parasympathetic system, reduces arousal'
            },
            'low_motivation': {
                'meditations': ['Movement Meditation', 'Visualization', 'Focused Meditation'],
                'rationale': 'Engages body, stimulates mental clarity'
            }
        }
        
        self.special_cases = {
            'focus': {
                'meditations': ['Focused Attention Meditation', 'Body Scan', 'Breath Awareness'],
                'rationale': 'Directs attention to single object, training concentration and sustained attention'
            },
            'grounding': {
                'meditations': ['5-4-3-2-1 Grounding', 'Body Scan', 'Visualization'],
                'rationale': 'Connects with bodily sensations, anchors awareness in present moment'
            }
        }
        
        # Keywords for ML-enhanced matching
        self.disorder_keywords = {
            'major_depressive_disorder': ['depression', 'depressed', 'sad', 'hopeless', 'worthless', 'empty', 'mood', 'low'],
            'generalized_anxiety_disorder': ['anxiety', 'anxious', 'worried', 'nervous', 'tense', 'restless', 'panic'],
            'ptsd': ['trauma', 'traumatic', 'flashback', 'nightmare', 'hypervigilant', 'avoidance', 'intrusive'],
            'substance_use_disorder': ['addiction', 'substance', 'alcohol', 'drugs', 'craving', 'relapse', 'withdrawal'],
            'adhd': ['attention', 'focus', 'concentration', 'hyperactive', 'impulsive', 'distracted'],
            'obsessive_thoughts': ['obsessive', 'compulsive', 'repetitive', 'intrusive thoughts', 'rumination', 'worry loop']
        }
        
        self.emotion_keywords = {
            'overwhelm': ['overwhelmed', 'too much', 'can\'t cope', 'drowning', 'flooded'],
            'stress': ['stressed', 'pressure', 'tension', 'burden', 'strained'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'rage', 'frustrated'],
            'sadness': ['sad', 'grief', 'sorrow', 'melancholy', 'blue', 'down'],
            'isolation': ['lonely', 'alone', 'isolated', 'disconnected', 'withdrawn'],
            'worry': ['worried', 'anxious', 'concern', 'fear', 'apprehension'],
            'fear': ['afraid', 'scared', 'terrified', 'phobia', 'panic'],
            'panic': ['panic', 'panic attack', 'overwhelming fear', 'can\'t breathe'],
            'low_motivation': ['unmotivated', 'lethargic', 'no energy', 'apathy', 'lazy']
        }
        
        # Initialize ML components (optional) without importing sklearn at module import time
        self.use_ml = bool(use_ml)
        self.vectorizer = None
        self.training_vectors = None
        self.training_labels = []
        self._cosine_similarity = None
        if self.use_ml:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
                self.vectorizer = _TfidfVectorizer(stop_words='english', max_features=1000)
                self._cosine_similarity = _cosine_similarity
                self._prepare_ml_model()
            except Exception:
                # Disable ML if sklearn/scipy stack unavailable
                self.use_ml = False
                self.vectorizer = None
                self.training_vectors = None
                self.training_labels = []
    
    def _prepare_ml_model(self):
        """Prepare the ML model with keyword-based training data"""
        # Create training corpus
        training_texts = []
        training_labels = []
        
        # Add disorder keywords
        for disorder, keywords in self.disorder_keywords.items():
            for keyword in keywords:
                training_texts.append(keyword)
                training_labels.append(f"disorder_{disorder}")
        
        # Add emotion keywords
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                training_texts.append(keyword)
                training_labels.append(f"emotion_{emotion}")
        
        # Fit vectorizer
        if self.vectorizer is not None:
            self.vectorizer.fit(training_texts)
            self.training_vectors = self.vectorizer.transform(training_texts)
            self.training_labels = training_labels
    
    def _normalize_text(self, text: str) -> str:
        """Normalize input text for processing"""
        return re.sub(r'[^\w\s]', '', text.lower().strip())
    
    def _extract_keywords_from_feedback(self, feedback: str) -> List[str]:
        """Extract relevant keywords from user feedback"""
        normalized = self._normalize_text(feedback)
        words = normalized.split()
        
        # Filter for meaningful words (basic approach)
        meaningful_words = [word for word in words if len(word) > 2]
        return meaningful_words
    
    def _rule_based_selection(self, diagnosis: Dict, feedback: str) -> List[MeditationRecommendation]:
        """Rule-based meditation selection"""
        recommendations = []
        
        # Process diagnosis
        if 'mental_disorder' in diagnosis:
            disorder = self._normalize_text(diagnosis['mental_disorder'])
            
            # Map common disorder names to our keys
            disorder_mapping = {
                'depression': 'major_depressive_disorder',
                'anxiety': 'generalized_anxiety_disorder',
                'ptsd': 'ptsd',
                'addiction': 'substance_use_disorder',
                'adhd': 'adhd',
                'ocd': 'obsessive_thoughts'
            }
            
            for key, mapped_disorder in disorder_mapping.items():
                if key in disorder:
                    if mapped_disorder in self.disorder_mappings:
                        mapping = self.disorder_mappings[mapped_disorder]
                        for meditation in mapping['meditations']:
                            recommendations.append(
                                MeditationRecommendation(
                                    meditation_type=meditation,
                                    confidence=0.9,
                                    rationale=mapping['rationale'],
                                    source='rule_based'
                                )
                            )
        
        # Process emotions from feedback
        feedback_lower = feedback.lower()
        for emotion, mapping in self.emotion_mappings.items():
            if emotion in feedback_lower:
                for meditation in mapping['meditations']:
                    recommendations.append(
                        MeditationRecommendation(
                            meditation_type=meditation,
                            confidence=0.8,
                            rationale=mapping['rationale'],
                            source='rule_based'
                        )
                    )
        
        # Check for special cases
        for special_case, mapping in self.special_cases.items():
            if special_case in feedback_lower:
                for meditation in mapping['meditations']:
                    recommendations.append(
                        MeditationRecommendation(
                            meditation_type=meditation,
                            confidence=0.85,
                            rationale=mapping['rationale'],
                            source='rule_based'
                        )
                    )
        
        return recommendations
    
    def _ml_enhanced_selection(self, feedback: str) -> List[MeditationRecommendation]:
        """ML-enhanced meditation selection using similarity matching"""
        if not self.use_ml or self.vectorizer is None or self.training_vectors is None or self._cosine_similarity is None:
            return []
        recommendations = []
        
        # Vectorize user feedback
        feedback_vector = self.vectorizer.transform([feedback])
        
        # Calculate similarities
        similarities = self._cosine_similarity(feedback_vector, self.training_vectors)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[-5:][::-1]  # Top 5 matches
        
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                label = self.training_labels[idx]
                confidence = float(similarities[idx])
                
                if label.startswith('disorder_'):
                    disorder = label.replace('disorder_', '')
                    if disorder in self.disorder_mappings:
                        mapping = self.disorder_mappings[disorder]
                        for meditation in mapping['meditations']:
                            recommendations.append(
                                MeditationRecommendation(
                                    meditation_type=meditation,
                                    confidence=confidence * 0.7,  # Lower confidence for ML
                                    rationale=f"ML-detected: {mapping['rationale']}",
                                    source='ml_enhanced'
                                )
                            )
                
                elif label.startswith('emotion_'):
                    emotion = label.replace('emotion_', '')
                    if emotion in self.emotion_mappings:
                        mapping = self.emotion_mappings[emotion]
                        for meditation in mapping['meditations']:
                            recommendations.append(
                                MeditationRecommendation(
                                    meditation_type=meditation,
                                    confidence=confidence * 0.7,
                                    rationale=f"ML-detected: {mapping['rationale']}",
                                    source='ml_enhanced'
                                )
                            )
        
        return recommendations
    
    def _merge_and_rank_recommendations(self, recommendations: List[MeditationRecommendation]) -> List[MeditationRecommendation]:
        """Merge duplicate recommendations and rank by confidence"""
        # Group by meditation type
        meditation_groups = defaultdict(list)
        for rec in recommendations:
            meditation_groups[rec.meditation_type].append(rec)
        
        # Merge duplicates by taking highest confidence
        merged_recommendations = []
        for meditation_type, recs in meditation_groups.items():
            best_rec = max(recs, key=lambda x: x.confidence)
            
            # Boost confidence if multiple sources agree
            if len(recs) > 1:
                sources = set(rec.source for rec in recs)
                if len(sources) > 1:  # Both rule-based and ML agree
                    best_rec.confidence = min(0.95, best_rec.confidence * 1.2)
                    best_rec.rationale += " (Multiple methods agree)"
            
            merged_recommendations.append(best_rec)
        
        # Sort by confidence
        return sorted(merged_recommendations, key=lambda x: x.confidence, reverse=True)
    
    def select_meditation(self, user_feedback_path: str, user_diagnosis_path: str) -> Dict:
        """
        Main method to select meditation based on user feedback and diagnosis
        
        Args:
            user_feedback_path: Path to JSON file containing user feedback
            user_diagnosis_path: Path to JSON file containing diagnosis
            
        Returns:
            Dictionary containing recommendations
        """
        try:
            # Load input files
            with open(user_feedback_path, 'r') as f:
                feedback_data = json.load(f)
            
            with open(user_diagnosis_path, 'r') as f:
                diagnosis_data = json.load(f)
            
            # Extract feedback text
            feedback_text = ""
            if isinstance(feedback_data, dict):
                if 'feedback' in feedback_data:
                    feedback_text = feedback_data['feedback']
                elif 'user_prompt' in feedback_data:
                    feedback_text = feedback_data['user_prompt']
                else:
                    feedback_text = str(feedback_data)
            else:
                feedback_text = str(feedback_data)
            
            # Get recommendations from both methods
            rule_based_recs = self._rule_based_selection(diagnosis_data, feedback_text)
            ml_enhanced_recs = self._ml_enhanced_selection(feedback_text)
            
            # Merge and rank recommendations
            all_recommendations = rule_based_recs + ml_enhanced_recs
            final_recommendations = self._merge_and_rank_recommendations(all_recommendations)
            
            # Prepare output
            result = {
                'user_feedback': feedback_text,
                'diagnosis': diagnosis_data,
                'recommendations': [
                    {
                        'meditation_type': rec.meditation_type,
                        'confidence_score': round(rec.confidence, 3),
                        'rationale': rec.rationale,
                        'selection_method': rec.source
                    }
                    for rec in final_recommendations[:5]  # Top 5 recommendations
                ],
                'total_recommendations_found': len(final_recommendations),
                'metadata': {
                    'rule_based_matches': len(rule_based_recs),
                    'ml_enhanced_matches': len(ml_enhanced_recs),
                    'hybrid_boost_applied': any(
                        'Multiple methods agree' in rec.rationale 
                        for rec in final_recommendations
                    ),
                    'ml_enabled': bool(self.use_ml)
                }
            }
            
            return result
            
        except FileNotFoundError as e:
            return {'error': f'File not found: {e}'}
        except json.JSONDecodeError as e:
            return {'error': f'Invalid JSON format: {e}'}
        except Exception as e:
            return {'error': f'Unexpected error: {e}'}
    
    def select_meditation_direct(self, feedback_text: str, diagnosis_dict: Dict) -> Dict:
        """
        Direct method to select meditation without file I/O
        
        Args:
            feedback_text: User feedback as string
            diagnosis_dict: Diagnosis data as dictionary
            
        Returns:
            Dictionary containing recommendations
        """
        # Get recommendations from both methods
        rule_based_recs = self._rule_based_selection(diagnosis_dict, feedback_text)
        ml_enhanced_recs = self._ml_enhanced_selection(feedback_text)
        
        # Merge and rank recommendations
        all_recommendations = rule_based_recs + ml_enhanced_recs
        final_recommendations = self._merge_and_rank_recommendations(all_recommendations)
        
        # Prepare output
        result = {
            'user_feedback': feedback_text,
            'diagnosis': diagnosis_dict,
            'recommendations': [
                {
                    'meditation_type': rec.meditation_type,
                    'confidence_score': round(rec.confidence, 3),
                    'rationale': rec.rationale,
                    'selection_method': rec.source
                }
                for rec in final_recommendations[:5]  # Top 5 recommendations
            ],
            'total_recommendations_found': len(final_recommendations),
            'metadata': {
                'rule_based_matches': len(rule_based_recs),
                'ml_enhanced_matches': len(ml_enhanced_recs),
                'hybrid_boost_applied': any(
                    'Multiple methods agree' in rec.rationale 
                    for rec in final_recommendations
                )
            }
        }
        
        return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Meditation Selector - JSON input")
    parser.add_argument("--feedback", type=str, default="preprocess_input/user_feedback.json",
                        help="Path to user feedback JSON")
    parser.add_argument("--diagnosis", type=str, default="preprocess_input/diagnosis_data.json",
                        help="Path to diagnosis JSON")
    parser.add_argument("--diary", type=str, default="preprocess_input/diary_entry.json",
                        help="Path to user's diary/intent JSON (optional)")
    parser.add_argument("--output", type=str, default="meditation_recommendations_output.json",
                        help="Path to write recommendations JSON")
    parser.add_argument("--rule-only", action="store_true",
                        help="Disable ML and use only rule-based selection (avoids sklearn import)")
    args = parser.parse_args()

    selector = MeditationSelectorModule(use_ml=(not args.rule_only))
    # Optionally augment feedback with diary intent
    try:
        with open(args.feedback, 'r', encoding='utf-8') as f:
            fb = json.load(f)
    except Exception:
        fb = {}
    try:
        with open(args.diary, 'r', encoding='utf-8') as f:
            diary = json.load(f)
    except Exception:
        diary = {}

    # Build a synthetic feedback string combining both
    if isinstance(fb, dict):
        base_text = fb.get('feedback', '') or fb.get('user_prompt', '')
        if not base_text and fb.get('recent_sessions'):
            base_text = fb['recent_sessions'][0].get('feedbackText', '')
    else:
        base_text = str(fb)
    diary_text = ''
    if isinstance(diary, dict):
        diary_text = diary.get('entry', '') or diary.get('intent', '')
    combined_feedback = (base_text + ' ' + diary_text).strip()

    # Persist a small combined feedback file for transparency
    combined_path = 'preprocess_output/combined_feedback.json'
    try:
        Path('preprocess_output').mkdir(parents=True, exist_ok=True)
        with open(combined_path, 'w', encoding='utf-8') as cf:
            json.dump({
                'userId': (fb.get('userId') if isinstance(fb, dict) else 'test_user_1'),
                'feedback': combined_feedback
            }, cf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Run selector using combined feedback and diagnosis
    result = selector.select_meditation(combined_path, args.diagnosis)

    # Always write to file so results are accessible even if stdout is hidden
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # As a fallback, try to at least print the error
        print(f"Failed to write output: {e}")

    # Minimal stdout for interactive runs
    try:
        print(f"Wrote recommendations to {args.output}")
    except Exception:
        pass


if __name__ == "__main__":
    # Direct-run mode: read default JSON inputs and write output file
    try:
        selector = MeditationSelectorModule(use_ml=False)
        result = selector.select_meditation(DEFAULT_FEEDBACK_PATH, DEFAULT_DIAGNOSIS_PATH)
        with open(DEFAULT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Wrote recommendations to {DEFAULT_OUTPUT_PATH}")
    except Exception as e:
        # Fall back to CLI if direct mode fails
        main()