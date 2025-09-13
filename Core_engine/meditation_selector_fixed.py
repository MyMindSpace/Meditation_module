import json
import argparse
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
from pathlib import Path

# Constants
DEFAULT_FEEDBACK_PATH = "preprocess_output/user_feedback_processed.json"
DEFAULT_DIAGNOSIS_PATH = "preprocess_output/diagnosis_processed.json"
DEFAULT_OUTPUT_PATH = "meditation_recommendations_output.json"

@dataclass
class MeditationRecommendation:
    meditation_type: str
    confidence: float
    rationale: str
    source: str  # 'rule_based' or 'ml_enhanced'

class MeditationSelectorModule:
    """
    A hybrid meditation selector that combines rule-based logic with ML enhancement
    """
    
    def __init__(self, use_ml: bool = True):
        # List of valid meditation types from meditation.csv
        self.valid_meditations = [
            'Mindfulness Meditation',
            'Transcendental Meditation',
            'Loving-Kindness Meditation',
            'Body Scan Meditation',
            'Vipassana Meditation',
            'Zen Meditation',
            'Chakra Meditation',
            'Mantra Meditation',
            'Breath Counting Meditation',
            'Walking Meditation',
            'Visualization Meditation',
            'Yoga Nidra Meditation',
            'Sound Meditation',
            'Qi Gong Meditation',
            'Kundalini Meditation',
            'Silent Meditation',
            'Open Monitoring Meditation',
            'Focused Attention Meditation',
            'Metta Meditation',
            'Body Awareness Meditation',
            'Compassion Meditation',
            'Breath Awareness and Body Scan Meditation',
            'Progressive Muscle Relaxation',
            'Guided Meditation',
            'Mindful Eating Meditation',
            'Mindful Breathing Meditation',
            'Compassionate Mind Meditation',
            'Spiritual Meditation',
            'Chanting Meditation',
            'Third Eye Meditation',
            'Mudra Meditation',
            'Self-Inquiry Meditation',
            'Inner Light Meditation',
            'Mindful Movement Meditation',
            'Gratitude Meditation',
            'Affirmation Meditation',
            'Compassionate Body Scan Meditation',
            'Breath and Sound Meditation',
            'Morning Meditation',
            'Evening Meditation',
            'Breath and Body Awareness Meditation',
            'Mindful Communication Meditation',
            'Labyrinth Meditation',
            'Color Visualization Meditation',
            'Breath and Loving-Kindness Meditation',
            'Gratitude Journaling Meditation',
            'Sitting with Difficult Emotions Meditation',
            'Mindful Walking Meditation',
            'Gratitude Body Scan Meditation',
            'Breath and Body Awareness in Nature Meditation',
            'Breath and Loving-Kindness in Nature Meditation',
            'Compassionate Forgiveness Meditation',
            'Breath and Body Awareness with Affirmations Meditation',
            'Body Awareness and Gratitude Meditation',
            'Breath and Open Awareness Meditation',
            'Singing Bowl Meditation',
            'Body Scan and Loving-Kindness Meditation',
            'Breath and Movement Meditation',
            'Gratitude Visualization Meditation',
            'Breath and Body Awareness with Mantra Meditation',
            'Relaxation Meditation',
            'Self-Compassion Meditation',
            'Breath and Loving-Kindness in Community Meditation',
            'Gratitude and Loving-Kindness Meditation',
            'Breath and Body Awareness with Chakra Meditation',
            'Compassionate Self-Talk Meditation'
        ]
        
        # Clinical rule base for disorder mapping
        self.disorder_rulebase = {
            'anxiety': [
                'Mindfulness Meditation', 'Breath Counting Meditation', 'Body Scan Meditation',
                'Guided Meditation', 'Breath Awareness and Body Scan Meditation', 'Yoga Nidra Meditation',
                'Sound Meditation', 'Progressive Muscle Relaxation'
            ],
            'stress': [
                'Mindfulness Meditation', 'Breath Counting Meditation', 'Body Scan Meditation',
                'Guided Meditation', 'Breath Awareness and Body Scan Meditation', 'Yoga Nidra Meditation',
                'Sound Meditation', 'Progressive Muscle Relaxation'
            ],
            'depression': [
                'Mindfulness Meditation', 'Loving-Kindness Meditation', 'Metta Meditation', 'Compassion Meditation',
                'Gratitude Meditation', 'Affirmation Meditation', 'Guided Meditation', 'Body Awareness Meditation',
                'Visualization Meditation', 'Self-Compassion Meditation'
            ],
            'low mood': [
                'Mindfulness Meditation', 'Loving-Kindness Meditation', 'Metta Meditation', 'Compassion Meditation',
                'Gratitude Meditation', 'Affirmation Meditation', 'Guided Meditation', 'Body Awareness Meditation',
                'Visualization Meditation', 'Self-Compassion Meditation'
            ],
            'adhd': [
                'Focused Attention Meditation', 'Mantra Meditation', 'Breath Counting Meditation',
                'Mindful Breathing Meditation', 'Qi Gong Meditation', 'Mindful Movement Meditation',
                'Walking Meditation', 'Sound Meditation'
            ],
            'attention issues': [
                'Focused Attention Meditation', 'Mantra Meditation', 'Breath Counting Meditation',
                'Mindful Breathing Meditation', 'Qi Gong Meditation', 'Mindful Movement Meditation',
                'Walking Meditation', 'Sound Meditation'
            ],
            'substance use disorders': [
                'Mindfulness Meditation', 'Transcendental Meditation', 'Vipassana Meditation',
                'Body Scan Meditation', 'Self-Inquiry Meditation', 'Guided Meditation'
            ],
            'insomnia': [
                'Yoga Nidra Meditation', 'Guided Meditation', 'Breath Awareness and Body Scan Meditation',
                'Progressive Muscle Relaxation', 'Sound Meditation', 'Evening Meditation'
            ],
            'sleep problems': [
                'Yoga Nidra Meditation', 'Guided Meditation', 'Breath Awareness and Body Scan Meditation',
                'Progressive Muscle Relaxation', 'Sound Meditation', 'Evening Meditation'
            ],
            'ptsd': [
                'Body Scan Meditation', 'Compassion Meditation', 'Loving-Kindness Meditation',
                'Self-Compassion Meditation', 'Compassionate Forgiveness Meditation',
                'Guided Meditation', 'Mindful Movement Meditation'
            ],
            'trauma': [
                'Body Scan Meditation', 'Compassion Meditation', 'Loving-Kindness Meditation',
                'Self-Compassion Meditation', 'Compassionate Forgiveness Meditation',
                'Guided Meditation', 'Mindful Movement Meditation'
            ],
            'emotional pain': [
                'Body Scan Meditation', 'Compassion Meditation', 'Loving-Kindness Meditation',
                'Self-Compassion Meditation', 'Compassionate Forgiveness Meditation',
                'Guided Meditation', 'Mindful Movement Meditation'
            ],
            'social anxiety': [
                'Loving-Kindness Meditation', 'Compassion Meditation', 'Mindful Communication Meditation',
                'Gratitude Meditation', 'Metta Meditation'
            ],
            'anger': [
                'Loving-Kindness Meditation', 'Compassion Meditation', 'Mindful Communication Meditation',
                'Gratitude Meditation', 'Metta Meditation'
            ]
        }
        
        # Emotion-based rule mapping
        self.emotion_rulebase = {
            'sadness': ['Loving-Kindness Meditation', 'Gratitude Meditation', 'Affirmation Meditation'],
            'fear': ['Mindful Breathing Meditation', 'Body Awareness Meditation', 'Guided Meditation'],
            'worry': ['Mindful Breathing Meditation', 'Body Awareness Meditation', 'Guided Meditation'],
            'anger': ['Compassion Meditation', 'Body Scan Meditation', 'Metta Meditation', 'Gratitude Meditation'],
            'irritability': ['Compassion Meditation', 'Body Scan Meditation', 'Metta Meditation', 'Gratitude Meditation'],
            'overwhelmed': ['Body Scan Meditation', 'Visualization Meditation', 'Open Monitoring Meditation'],
            'loneliness': ['Compassion Meditation', 'Breath and Loving-Kindness in Community Meditation', 'Mindful Communication Meditation']
        }
        
        # Outcome-based rule mapping
        self.outcome_rulebase = {
            'improved focus': ['Focused Attention Meditation', 'Mantra Meditation', 'Breath Counting Meditation', 'Third Eye Meditation', 'Mindful Walking Meditation'],
            'grounding': ['Body Scan Meditation', 'Breath Awareness and Body Scan Meditation', 'Mindful Movement Meditation', 'Sound Meditation', 'Walking Meditation'],
            'centering': ['Zen Meditation', 'Vipassana Meditation', 'Silent Meditation', 'Open Monitoring Meditation'],
            'clarity': ['Zen Meditation', 'Vipassana Meditation', 'Silent Meditation', 'Open Monitoring Meditation'],
            'building energy': ['Kundalini Meditation', 'Qi Gong Meditation', 'Breath and Movement Meditation']
        }
        
        self.use_ml = use_ml
    
    def _rule_based_selection(self, diagnosis, feedback: str):
        """Select meditations based on clinical rules"""
        recommendations = []
        
        # Handle diagnosis data - could be dict or list
        diagnosis_text = ""
        if isinstance(diagnosis, list) and len(diagnosis) > 0:
            first_diagnosis = diagnosis[0]
            if isinstance(first_diagnosis, dict):
                diagnosis_text = str(first_diagnosis.get('mental_disorder', '')).lower()
        elif isinstance(diagnosis, dict):
            diagnosis_text = str(diagnosis.get('mental_disorder', '')).lower()
        
        feedback_text = feedback.lower()
        
        # Check disorder mappings
        for disorder, meditations in self.disorder_rulebase.items():
            if disorder in diagnosis_text or disorder in feedback_text:
                for meditation in meditations[:3]:
                    if meditation in self.valid_meditations:
                        recommendations.append(MeditationRecommendation(
                            meditation_type=meditation,
                            confidence=0.9,
                            rationale=f"Recommended for {disorder}",
                            source='rule_base_disorder'
                        ))
        
        # Check emotion mappings
        for emotion, meditations in self.emotion_rulebase.items():
            if emotion in feedback_text:
                for meditation in meditations[:3]:
                    if meditation in self.valid_meditations:
                        recommendations.append(MeditationRecommendation(
                            meditation_type=meditation,
                            confidence=0.85,
                            rationale=f"Recommended for {emotion}",
                            source='rule_base_emotion'
                        ))
        
        # Check outcome mappings
        for outcome, meditations in self.outcome_rulebase.items():
            if outcome in feedback_text:
                for meditation in meditations[:3]:
                    if meditation in self.valid_meditations:
                        recommendations.append(MeditationRecommendation(
                            meditation_type=meditation,
                            confidence=0.8,
                            rationale=f"Recommended for {outcome}",
                            source='rule_base_outcome'
                        ))
        
        # If no clear match, add safe defaults
        if not recommendations:
            recommendations.append(MeditationRecommendation(
                meditation_type='Mindful Breathing Meditation',
                confidence=0.6,
                rationale='Safe, universal default',
                source='rule_base_default'
            ))
            recommendations.append(MeditationRecommendation(
                meditation_type='Body Scan Meditation',
                confidence=0.7,
                rationale='Safe, accessible default',
                source='rule_base_default'
            ))
        
        return recommendations
    
    def _ml_enhanced_selection(self, feedback: str) -> List[MeditationRecommendation]:
        """ML-enhanced selection (placeholder - returns empty list when ML disabled)"""
        if not self.use_ml:
            return []
        
        # Placeholder for ML logic - would implement actual ML here
        return []
    
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
            if isinstance(feedback_data, list) and len(feedback_data) > 0:
                # Handle list of feedback objects
                first_feedback = feedback_data[0]
                if isinstance(first_feedback, dict):
                    feedback_text = first_feedback.get('feedbackText', '') or first_feedback.get('user_prompt', '')
                else:
                    feedback_text = str(first_feedback)
            elif isinstance(feedback_data, dict):
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
                        'confidence': rec.confidence,
                        'rationale': rec.rationale,
                        'source': rec.source
                    }
                    for rec in final_recommendations[:5]  # Top 5 recommendations
                ],
                'total_candidates': len(final_recommendations),
                'method': 'hybrid' if self.use_ml else 'rule_based_only'
            }
            
            return result
            
        except Exception as e:
            return {'error': f"Unexpected error: {e}"}

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
    base_text = ''
    if isinstance(fb, list) and len(fb) > 0:
        # Handle list of feedback objects
        first_feedback = fb[0]
        base_text = first_feedback.get('feedbackText', '') or first_feedback.get('user_prompt', '')
    elif isinstance(fb, dict):
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
                'userId': (fb[0].get('userId') if isinstance(fb, list) and len(fb) > 0 
                          else fb.get('userId') if isinstance(fb, dict) 
                          else 'test_user_1'),
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
