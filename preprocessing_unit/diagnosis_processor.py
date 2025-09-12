# preprocessing_unit/diagnosis_processor.py
"""
Diagnosis Processor (DP)

Handles diagnosis data preprocessing:
- Text cleaning and normalization
- Medical terminology standardization
- Severity assessment
- Clinical feature extraction
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class DiagnosisProcessor:
    """
    Diagnosis processor for meditation module
    Handles clinical text preprocessing and normalization
    """
    
    def __init__(self, normalize_medical_terms: bool = True):
        self.normalize_medical_terms = normalize_medical_terms
        
        # Medical abbreviation mappings
        self.medical_abbreviations = {
            # Common clinical abbreviations
            'pt': 'patient', 'pts': 'patients',
            'hx': 'history', 'h/o': 'history of',
            'dx': 'diagnosis', 'ddx': 'differential diagnosis',
            'sx': 'symptoms', 'sxs': 'symptoms',
            'tx': 'treatment', 'rx': 'prescription',
            'c/o': 'complains of', 'c/w': 'consistent with',
            'r/o': 'rule out', 's/p': 'status post',
            'w/': 'with', 'w/o': 'without',
            'vs': 'versus', 'v/s': 'vital signs',
            'b/l': 'bilateral', 'bilat': 'bilateral',
            'unilat': 'unilateral', 'u/l': 'unilateral',
            
            # Psychiatric abbreviations
            'mdd': 'major depressive disorder',
            'gad': 'generalized anxiety disorder',
            'ocd': 'obsessive compulsive disorder',
            'ptsd': 'post traumatic stress disorder',
            'bpd': 'borderline personality disorder',
            'adhd': 'attention deficit hyperactivity disorder',
            'add': 'attention deficit disorder',
            'sad': 'seasonal affective disorder',
            'pmdd': 'premenstrual dysphoric disorder',
            
            # Medication abbreviations
            'ssri': 'selective serotonin reuptake inhibitor',
            'snri': 'serotonin norepinephrine reuptake inhibitor',
            'tca': 'tricyclic antidepressant',
            'maoi': 'monoamine oxidase inhibitor',
            'benzo': 'benzodiazepine',
            'antipsych': 'antipsychotic',
            
            # Frequency/dosing
            'qd': 'once daily', 'bid': 'twice daily',
            'tid': 'three times daily', 'qid': 'four times daily',
            'prn': 'as needed', 'hs': 'at bedtime',
            'ac': 'before meals', 'pc': 'after meals',
            
            # Assessment abbreviations
            'wdl': 'within defined limits',
            'wnl': 'within normal limits',
            'nad': 'no acute distress',
            'nkda': 'no known drug allergies',
            'nkfa': 'no known food allergies',
        }
        
        # Disorder synonyms and variations
        self.disorder_synonyms = {
            'depression': [
                'depression', 'depressive disorder', 'major depression',
                'clinical depression', 'unipolar depression', 'mdd'
            ],
            'anxiety': [
                'anxiety', 'anxiety disorder', 'generalized anxiety',
                'gad', 'anxiousness', 'nervousness'
            ],
            'panic_disorder': [
                'panic disorder', 'panic attacks', 'panic syndrome',
                'episodic anxiety', 'acute anxiety'
            ],
            'ptsd': [
                'ptsd', 'post-traumatic stress', 'trauma disorder',
                'combat stress', 'shell shock'
            ],
            'ocd': [
                'ocd', 'obsessive-compulsive', 'obsessional disorder',
                'compulsive disorder', 'intrusive thoughts'
            ],
            'bipolar': [
                'bipolar', 'manic-depressive', 'bipolar disorder',
                'mood disorder', 'cyclothymia', 'hypomania'
            ],
            'schizophrenia': [
                'schizophrenia', 'psychotic disorder', 'psychosis',
                'schizoaffective', 'delusional disorder'
            ],
            'adhd': [
                'adhd', 'add', 'attention deficit', 'hyperactivity',
                'concentration disorder', 'focus problems'
            ],
            'eating_disorder': [
                'eating disorder', 'anorexia', 'bulimia', 'binge eating',
                'restrictive eating', 'purging disorder'
            ],
            'substance_abuse': [
                'substance abuse', 'addiction', 'dependency', 'alcoholism',
                'drug abuse', 'chemical dependency', 'substance use disorder'
            ]
        }
        
        # Severity keywords
        self.severity_terms = {
            'mild': [
                'mild', 'minimal', 'slight', 'minor', 'low-grade',
                'subclinical', 'subsyndromal', 'emerging'
            ],
            'moderate': [
                'moderate', 'medium', 'average', 'typical', 'standard',
                'moderate-to-severe', 'clinically significant'
            ],
            'severe': [
                'severe', 'major', 'significant', 'marked', 'pronounced',
                'extreme', 'intense', 'debilitating', 'disabling'
            ],
            'critical': [
                'critical', 'acute', 'crisis', 'emergency', 'urgent',
                'life-threatening', 'imminent danger', 'suicidal',
                'psychotic break', 'decompensated'
            ]
        }
        
        # Functional impact terms
        self.functional_impact_terms = {
            'occupational': [
                'work', 'job', 'employment', 'career', 'workplace',
                'productivity', 'performance', 'absenteeism', 'fired'
            ],
            'social': [
                'social', 'relationships', 'friends', 'family', 'marriage',
                'isolation', 'withdrawn', 'interpersonal', 'dating'
            ],
            'academic': [
                'school', 'college', 'university', 'grades', 'studying',
                'learning', 'education', 'academic', 'classes'
            ],
            'daily_living': [
                'daily activities', 'self-care', 'hygiene', 'cooking',
                'cleaning', 'shopping', 'driving', 'chores', 'routine'
            ]
        }
        
        # Treatment terms
        self.treatment_terms = {
            'medication': [
                'medication', 'medicine', 'pills', 'prescription', 'drugs',
                'antidepressant', 'anxiolytic', 'mood stabilizer', 'antipsychotic'
            ],
            'psychotherapy': [
                'therapy', 'counseling', 'psychotherapy', 'cbt', 'dbt',
                'psychoanalysis', 'group therapy', 'individual therapy'
            ],
            'hospitalization': [
                'hospital', 'inpatient', 'admission', 'psychiatric ward',
                'emergency room', 'crisis intervention', 'involuntary hold'
            ],
            'alternative': [
                'meditation', 'yoga', 'acupuncture', 'massage', 'exercise',
                'diet', 'supplements', 'herbal', 'holistic'
            ]
        }
    
    def normalize_medical_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations to full terms"""
        if not self.normalize_medical_terms:
            return text
        
        # Replace abbreviations with full terms
        words = text.split()
        normalized_words = []
        
        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            if clean_word in self.medical_abbreviations:
                normalized_words.append(self.medical_abbreviations[clean_word])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def standardize_disorder_names(self, text: str) -> str:
        """Standardize disorder names to canonical forms"""
        if not self.normalize_medical_terms:
            return text
        
        text_lower = text.lower()
        
        # Replace disorder variations with standard terms
        for canonical_name, variations in self.disorder_synonyms.items():
            for variation in variations:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(variation) + r'\b'
                text_lower = re.sub(pattern, canonical_name, text_lower)
        
        return text_lower
    
    def clean_diagnosis_text(self, text: str) -> str:
        """Clean and normalize diagnosis text"""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove common clinical prefixes that don't add meaning
        prefixes_to_remove = [
            'patient presents with',
            'patient reports', 
            'pt c/o',
            'chief complaint:',
            'cc:',
            'history of present illness:',
            'hpi:'
        ]
        
        for prefix in prefixes_to_remove:
            text = text.replace(prefix, '')
        
        # Normalize medical abbreviations
        text = self.normalize_medical_abbreviations(text)
        
        # Standardize disorder names
        text = self.standardize_disorder_names(text)
        
        # Remove extra punctuation and normalize spacing
        text = re.sub(r'[^\w\s\.\,\;\:\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short words (likely abbreviations we missed)
        words = text.split()
        words = [word for word in words if len(word) >= 2 or word in ['i', 'a']]
        
        return ' '.join(words).strip()
    
    def extract_disorders(self, text: str) -> List[str]:
        """Extract disorder mentions from text"""
        clean_text = self.clean_diagnosis_text(text)
        found_disorders = []
        
        for canonical_name, variations in self.disorder_synonyms.items():
            for variation in variations:
                pattern = r'\b' + re.escape(variation) + r'\b'
                if re.search(pattern, clean_text):
                    found_disorders.append(canonical_name)
                    break  # Found this disorder, move to next
        
        return list(set(found_disorders))  # Remove duplicates
    
    def assess_severity_from_text(self, text: str) -> Dict[str, float]:
        """Assess severity level from text content"""
        clean_text = self.clean_diagnosis_text(text)
        severity_scores = {level: 0.0 for level in self.severity_terms.keys()}
        
        # Count severity indicators
        for level, terms in self.severity_terms.items():
            for term in terms:
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = len(re.findall(pattern, clean_text))
                severity_scores[level] += matches
        
        # Normalize to probabilities
        total_score = sum(severity_scores.values())
        if total_score > 0:
            severity_scores = {k: v / total_score for k, v in severity_scores.items()}
        else:
            # Default distribution if no explicit severity mentioned
            severity_scores = {'mild': 0.4, 'moderate': 0.4, 'severe': 0.15, 'critical': 0.05}
        
        return severity_scores
    
    def extract_functional_impact(self, text: str) -> Dict[str, float]:
        """Extract functional impairment indicators"""
        clean_text = self.clean_diagnosis_text(text)
        impact_scores = {}
        
        for domain, terms in self.functional_impact_terms.items():
            score = 0.0
            for term in terms:
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = len(re.findall(pattern, clean_text))
                score += matches
            
            # Normalize by number of terms in domain
            impact_scores[domain] = min(1.0, score / len(terms))
        
        return impact_scores
    
    def extract_treatment_history(self, text: str) -> Dict[str, float]:
        """Extract treatment history indicators"""
        clean_text = self.clean_diagnosis_text(text)
        treatment_scores = {}
        
        for treatment_type, terms in self.treatment_terms.items():
            score = 0.0
            for term in terms:
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = len(re.findall(pattern, clean_text))
                score += matches
            
            treatment_scores[treatment_type] = min(1.0, score)
        
        return treatment_scores
    
    def extract_temporal_markers(self, text: str) -> Dict[str, List[str]]:
        """Extract temporal information about symptoms and disorders"""
        clean_text = self.clean_diagnosis_text(text)
        
        temporal_patterns = {
            'onset': [
                r'started (\w+ )?(\w+ )?ago',
                r'began (\w+ )?(\w+ )?ago', 
                r'first noticed (\w+ )?(\w+ )?ago',
                r'since (\w+)',
                r'for the past (\w+ \w+)',
                r'over the last (\w+ \w+)'
            ],
            'duration': [
                r'lasting (\w+ \w+)',
                r'for (\w+ \w+)',
                r'(\w+ \w+) duration',
                r'ongoing for (\w+ \w+)'
            ],
            'frequency': [
                r'(\w+) times? (per |a )(\w+)',
                r'(\w+) episodes?',
                r'intermittent',
                r'chronic',
                r'acute',
                r'episodic'
            ]
        }
        
        extracted_markers = {category: [] for category in temporal_patterns.keys()}
        
        for category, patterns in temporal_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, clean_text)
                if matches:
                    extracted_markers[category].extend([' '.join(match) if isinstance(match, tuple) else match for match in matches])
        
        return extracted_markers
    
    def calculate_text_complexity(self, text: str) -> Dict[str, float]:
        """Calculate complexity metrics for diagnosis text"""
        clean_text = self.clean_diagnosis_text(text)
        words = clean_text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if s.strip()]
        
        if not words:
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0.0,
                'avg_sentence_length': 0.0,
                'lexical_diversity': 0.0,
                'medical_term_density': 0.0
            }
        
        # Basic metrics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_word_length = np.mean([len(word) for word in words])
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Lexical diversity (type-token ratio)
        unique_words = len(set(words))
        lexical_diversity = unique_words / word_count
        
        # Medical term density
        medical_terms = set()
        for disorder_list in self.disorder_synonyms.values():
            medical_terms.update(disorder_list)
        for abbrev in self.medical_abbreviations.keys():
            medical_terms.add(abbrev)
        
        medical_word_count = sum(1 for word in words if word in medical_terms)
        medical_term_density = medical_word_count / word_count
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': float(avg_word_length),
            'avg_sentence_length': float(avg_sentence_length),
            'lexical_diversity': float(lexical_diversity),
            'medical_term_density': float(medical_term_density)
        }
    
    def process_diagnosis_entry(self, entry: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Process a single diagnosis entry"""
        # Extract text from various input formats
        if isinstance(entry, str):
            text = entry
            structured_data = {}
        elif isinstance(entry, dict):
            # Try to extract text from common fields
            text_fields = [
                'diagnosis', 'description', 'symptoms', 'history', 
                'notes', 'text', 'mental_disorder', 'clinical_notes',
                'assessment', 'impression', 'chief_complaint'
            ]
            
            text_parts = []
            for field in text_fields:
                value = entry.get(field)
                if value:
                    if isinstance(value, list):
                        text_parts.extend([str(v) for v in value])
                    else:
                        text_parts.append(str(value))
            
            text = ' '.join(text_parts) if text_parts else ''
            structured_data = entry
        else:
            text = str(entry)
            structured_data = {}
        
        if not text.strip():
            return self._empty_result("Empty diagnosis text")
        
        # Process text
        clean_text = self.clean_diagnosis_text(text)
        
        # Extract features
        disorders = self.extract_disorders(text)
        severity = self.assess_severity_from_text(text)
        functional_impact = self.extract_functional_impact(text)
        treatment_history = self.extract_treatment_history(text)
        temporal_markers = self.extract_temporal_markers(text)
        complexity_metrics = self.calculate_text_complexity(text)
        
        # Determine primary disorder
        primary_disorder = disorders[0] if disorders else 'unspecified'
        
        # Calculate overall severity score
        severity_score = (
            severity.get('mild', 0) * 1.0 +
            severity.get('moderate', 0) * 2.0 +
            severity.get('severe', 0) * 3.0 +
            severity.get('critical', 0) * 4.0
        ) / 4.0
        
        return {
            'original_text': text,
            'processed_text': clean_text,
            'extracted_disorders': disorders,
            'primary_disorder': primary_disorder,
            'severity_assessment': severity,
            'severity_score': float(severity_score),
            'functional_impact': functional_impact,
            'treatment_history': treatment_history,
            'temporal_markers': temporal_markers,
            'complexity_metrics': complexity_metrics,
            'structured_data': structured_data,
            'processing_metadata': {
                'abbreviations_expanded': self.normalize_medical_terms,
                'text_length_original': len(text),
                'text_length_processed': len(clean_text),
                'disorders_found': len(disorders)
            }
        }
    
    def _empty_result(self, error: str = None) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'original_text': '',
            'processed_text': '',
            'extracted_disorders': [],
            'primary_disorder': 'unspecified',
            'severity_assessment': {'mild': 0.5, 'moderate': 0.3, 'severe': 0.15, 'critical': 0.05},
            'severity_score': 0.0,
            'functional_impact': {domain: 0.0 for domain in self.functional_impact_terms.keys()},
            'treatment_history': {treatment: 0.0 for treatment in self.treatment_terms.keys()},
            'temporal_markers': {category: [] for category in ['onset', 'duration', 'frequency']},
            'complexity_metrics': {
                'word_count': 0, 'sentence_count': 0, 'avg_word_length': 0.0,
                'avg_sentence_length': 0.0, 'lexical_diversity': 0.0, 'medical_term_density': 0.0
            },
            'structured_data': {},
            'processing_metadata': {
                'abbreviations_expanded': self.normalize_medical_terms,
                'text_length_original': 0,
                'text_length_processed': 0,
                'disorders_found': 0,
                'error': error
            }
        }


def main():
    """CLI interface for diagnosis processor"""
    parser = argparse.ArgumentParser(description="Diagnosis Processor for meditation module")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSON file or text file with diagnosis data")
    parser.add_argument("--output", type=str, default="preprocess_output/diagnosis_processed.json",
                       help="Output JSON file for processed data")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Disable medical term normalization")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DiagnosisProcessor(normalize_medical_terms=not args.no_normalize)
    
    # Load input data
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Try to load as JSON first, then as text
    try:
        with input_path.open("r", encoding="utf-8") as f:
            if input_path.suffix.lower() == '.json':
                input_data = json.load(f)
            else:
                input_data = f.read()
    except json.JSONDecodeError:
        # If JSON parsing fails, treat as text
        with input_path.open("r", encoding="utf-8") as f:
            input_data = f.read()
    
    # Process data
    if isinstance(input_data, list):
        results = []
        for item in input_data:
            result = processor.process_diagnosis_entry(item)
            results.append(result)
    else:
        results = [processor.process_diagnosis_entry(input_data)]
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Diagnosis preprocessing complete. Processed {len(results)} entries.")
    print(f"Results saved to {output_path}")
    
    # Summary statistics
    if results:
        all_disorders = []
        severity_scores = []
        
        for result in results:
            all_disorders.extend(result.get('extracted_disorders', []))
            severity_scores.append(result.get('severity_score', 0))
        
        disorder_counts = Counter(all_disorders)
        avg_severity = np.mean(severity_scores) if severity_scores else 0
        
        print(f"\nSummary:")
        print(f"Most common disorders: {dict(disorder_counts.most_common(5))}")
        print(f"Average severity score: {avg_severity:.3f}")
        print(f"Normalization enabled: {not args.no_normalize}")


if __name__ == "__main__":
    main()