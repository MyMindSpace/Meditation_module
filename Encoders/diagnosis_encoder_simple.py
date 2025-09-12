# Encoders/diagnosis_encoder.py
"""
Diagnosis Encoder (DE)

Processes diagnosis data to extract:
- BERT/BioBERT embeddings for medical text
- Medical entity recognition
- Symptom embeddings
- Clinical feature extraction
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional imports with fallbacks
try:
    from transformers import (
        AutoTokenizer, AutoModel, pipeline,
        BertTokenizer, BertModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class DiagnosisEncoder:
    """
    Diagnosis Encoder for meditation module
    Extracts embeddings and features from medical/diagnostic text
    """
    
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 use_biobert: bool = False,
                 max_length: int = 512):
        
        self.model_name = model_name
        self.use_biobert = use_biobert
        self.max_length = max_length
        
        # Initialize transformers model
        if TRANSFORMERS_AVAILABLE:
            self._init_transformer_model()
        else:
            self.tokenizer = None
            self.model = None
            
        # Initialize TF-IDF as fallback
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=300,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            self.scaler = StandardScaler()
        else:
            self.tfidf_vectorizer = None
            self.scaler = None
            
        # Initialize NLP pipeline
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_md")
                except OSError:
                    print("Warning: No spaCy model found. Install with: python -m spacy download en_core_web_sm")
        
        # Medical/psychiatric terms mapping
        self.disorder_mapping = {
            'depression': ['depression', 'depressed', 'sad', 'sadness', 'hopeless', 'despair', 'melancholy'],
            'anxiety': ['anxiety', 'anxious', 'worry', 'worried', 'nervous', 'panic', 'fear', 'phobia'],
            'stress': ['stress', 'stressed', 'tension', 'pressure', 'overwhelmed', 'burden'],
            'trauma': ['trauma', 'traumatic', 'ptsd', 'flashback', 'nightmare', 'abuse'],
            'addiction': ['addiction', 'substance', 'alcohol', 'drug', 'dependency', 'withdrawal'],
            'adhd': ['adhd', 'attention', 'hyperactive', 'impulsive', 'focus', 'concentration'],
            'bipolar': ['bipolar', 'manic', 'mania', 'mood swing', 'euphoric'],
            'ocd': ['ocd', 'obsessive', 'compulsive', 'ritual', 'checking', 'counting'],
            'psychosis': ['psychosis', 'psychotic', 'hallucination', 'delusion', 'paranoid'],
            'eating_disorder': ['anorexia', 'bulimia', 'binge', 'eating disorder', 'body image']
        }
        
        self.symptom_mapping = {
            'mood': ['sad', 'happy', 'angry', 'irritable', 'mood', 'emotional'],
            'sleep': ['sleep', 'insomnia', 'nightmare', 'sleepless', 'tired', 'fatigue'],
            'cognitive': ['memory', 'concentration', 'focus', 'thinking', 'confused', 'foggy'],
            'physical': ['headache', 'pain', 'ache', 'nausea', 'dizzy', 'tremor'],
            'social': ['isolated', 'withdrawn', 'lonely', 'social', 'relationship'],
            'behavioral': ['aggressive', 'impulsive', 'restless', 'hyperactive', 'compulsive']
        }
        
        self.severity_indicators = {
            'mild': ['mild', 'slight', 'minor', 'little', 'somewhat', 'occasionally'],
            'moderate': ['moderate', 'medium', 'regular', 'frequent', 'often', 'some'],
            'severe': ['severe', 'serious', 'major', 'extreme', 'intense', 'constant', 'always'],
            'critical': ['critical', 'crisis', 'emergency', 'suicidal', 'dangerous', 'life-threatening']
        }
    
    def _init_transformer_model(self):
        """Initialize transformer model for embeddings"""
        try:
            if self.use_biobert:
                # Use BioBERT for medical text
                model_names = [
                    "dmis-lab/biobert-base-cased-v1.1",
                    "emilyalsentzer/Bio_ClinicalBERT",
                    "bert-base-uncased"  # Fallback
                ]
            else:
                model_names = ["bert-base-uncased"]
            
            for model_name in model_names:
                try:
                    print(f"Loading model: {model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModel.from_pretrained(model_name)
                    self.model_name = model_name
                    break
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
                    continue
            
            if self.model is None:
                raise Exception("Could not load any transformer model")
                
            # Set model to evaluation mode
            self.model.eval()
            
        except Exception as e:
            print(f"Error initializing transformer model: {e}")
            self.tokenizer = None
            self.model = None
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess diagnostic text"""
        if not isinstance(text, str):
            text = str(text)
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep medical terminology
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle common medical abbreviations
        abbreviations = {
            'pt': 'patient',
            'hx': 'history',
            'dx': 'diagnosis',
            'sx': 'symptoms',
            'tx': 'treatment',
            'c/o': 'complains of',
            'r/o': 'rule out',
            's/p': 'status post',
            'w/': 'with',
            'w/o': 'without'
        }
        
        for abbrev, full in abbreviations.items():
            text = re.sub(rf'\b{re.escape(abbrev)}\b', full, text)
        
        return text
    
    def extract_bert_embeddings(self, text: str) -> Optional[np.ndarray]:
        """Extract BERT/BioBERT embeddings from text"""
        if not TRANSFORMERS_AVAILABLE or self.tokenizer is None or self.model is None:
            return None
            
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True
            )
            
            # Get embeddings
            with torch.no_grad() if 'torch' in globals() else contextlib.nullcontext():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                return embeddings.flatten()
                
        except Exception as e:
            print(f"Error extracting BERT embeddings: {e}")
            return None
    
    def extract_tfidf_embeddings(self, text: str, fit_on_text: bool = True) -> Optional[np.ndarray]:
        """Extract TF-IDF embeddings as fallback"""
        if not SKLEARN_AVAILABLE or self.tfidf_vectorizer is None:
            return None
            
        try:
            if fit_on_text:
                # For single text, create a simple corpus
                corpus = [text, "healthy normal baseline"]  # Add baseline for comparison
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
                return tfidf_matrix[0].toarray().flatten()  # Return first document (the input)
            else:
                # Assume vectorizer is already fitted
                tfidf_matrix = self.tfidf_vectorizer.transform([text])
                return tfidf_matrix.toarray().flatten()
                
        except Exception as e:
            print(f"Error extracting TF-IDF embeddings: {e}")
            return None
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using NLP"""
        entities = {
            'disorders': [],
            'symptoms': [],
            'medications': [],
            'procedures': [],
            'anatomy': []
        }
        
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                
                # Extract named entities
                for ent in doc.ents:
                    if ent.label_ in ['DISEASE', 'SYMPTOM']:
                        entities['disorders'].append(ent.text.lower())
                    elif ent.label_ == 'CHEMICAL':
                        entities['medications'].append(ent.text.lower())
                    elif ent.label_ in ['ORGAN', 'BODY_PART']:
                        entities['anatomy'].append(ent.text.lower())
                        
            except Exception as e:
                print(f"Error in NER: {e}")
        
        # Rule-based entity extraction as fallback/supplement
        text_lower = text.lower()
        
        # Extract disorders
        for disorder, terms in self.disorder_mapping.items():
            for term in terms:
                if term in text_lower:
                    entities['disorders'].append(disorder)
                    break
        
        # Extract symptoms
        for symptom_category, terms in self.symptom_mapping.items():
            for term in terms:
                if term in text_lower:
                    entities['symptoms'].append(symptom_category)
                    break
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def assess_severity(self, text: str) -> Dict[str, float]:
        """Assess severity indicators in text"""
        text_lower = text.lower()
        severity_scores = {level: 0.0 for level in self.severity_indicators.keys()}
        
        for level, indicators in self.severity_indicators.items():
            for indicator in indicators:
                # Count occurrences (with word boundaries)
                pattern = rf'\b{re.escape(indicator)}\b'
                matches = len(re.findall(pattern, text_lower))
                severity_scores[level] += matches
        
        # Normalize scores
        total_scores = sum(severity_scores.values())
        if total_scores > 0:
            severity_scores = {k: v / total_scores for k, v in severity_scores.items()}
        
        return severity_scores
    
    def extract_clinical_features(self, text: str, entities: Dict[str, List[str]]) -> Dict[str, float]:
        """Extract clinical features from text and entities"""
        features = {}
        
        # Count features
        features['disorder_count'] = len(entities['disorders'])
        features['symptom_count'] = len(entities['symptoms'])
        features['medication_count'] = len(entities['medications'])
        features['text_length'] = len(text.split())
        
        # Presence of key disorder categories (binary features)
        disorder_categories = ['depression', 'anxiety', 'trauma', 'addiction', 'adhd']
        for category in disorder_categories:
            features[f'has_{category}'] = 1.0 if category in entities['disorders'] else 0.0
        
        # Symptom category coverage
        symptom_categories = ['mood', 'sleep', 'cognitive', 'physical', 'social', 'behavioral']
        for category in symptom_categories:
            features[f'has_{category}_symptoms'] = 1.0 if category in entities['symptoms'] else 0.0
        
        # Text complexity indicators
        sentences = text.split('.')
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_sentence_length'] = features['text_length'] / max(features['sentence_count'], 1)
        
        # Clinical language indicators
        clinical_terms = ['diagnosis', 'treatment', 'therapy', 'medication', 'symptoms', 'patient', 'history']
        clinical_term_count = sum(1 for term in clinical_terms if term in text.lower())
        features['clinical_language_ratio'] = clinical_term_count / max(features['text_length'], 1)
        
        return features
    
    def create_diagnosis_embedding(self, text: str) -> Dict[str, Any]:
        """Create comprehensive diagnosis embedding"""
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        # Extract different types of features
        bert_embedding = self.extract_bert_embeddings(clean_text)
        tfidf_embedding = self.extract_tfidf_embeddings(clean_text)
        entities = self.extract_medical_entities(clean_text)
        severity_scores = self.assess_severity(clean_text)
        clinical_features = self.extract_clinical_features(clean_text, entities)
        
        # Combine embeddings
        combined_embedding = []
        
        if bert_embedding is not None:
            # Use BERT as primary embedding
            combined_embedding = bert_embedding[:256].tolist()  # Limit size
        elif tfidf_embedding is not None:
            # Use TF-IDF as fallback
            combined_embedding = tfidf_embedding[:256].tolist()
        else:
            # Create simple bag-of-words embedding
            combined_embedding = self._create_bow_embedding(clean_text)
        
        # Add clinical features to embedding
        feature_vector = list(clinical_features.values()) + list(severity_scores.values())
        combined_embedding.extend(feature_vector)
        
        return {
            'embedding': combined_embedding,
            'entities': entities,
            'severity': severity_scores,
            'clinical_features': clinical_features,
            'primary_disorder': self._get_primary_disorder(entities, severity_scores),
            'risk_level': self._assess_risk_level(entities, severity_scores, clean_text)
        }
    
    def _create_bow_embedding(self, text: str, vocab_size: int = 128) -> List[float]:
        """Create simple bag-of-words embedding as last resort"""
        words = text.split()
        
        # Create simple hash-based embedding
        embedding = [0.0] * vocab_size
        
        for word in words:
            # Simple hash function
            hash_val = abs(hash(word)) % vocab_size
            embedding[hash_val] += 1.0
        
        # Normalize
        total = sum(embedding)
        if total > 0:
            embedding = [x / total for x in embedding]
        
        return embedding
    
    def _get_primary_disorder(self, entities: Dict[str, List[str]], 
                             severity_scores: Dict[str, float]) -> str:
        """Determine primary disorder from entities and severity"""
        if not entities['disorders']:
            return 'unspecified'
        
        # Weight disorders by severity
        disorder_weights = {}
        for disorder in entities['disorders']:
            # Base weight
            weight = 1.0
            
            # Increase weight for severe cases
            if severity_scores.get('severe', 0) > 0.3:
                weight *= 2.0
            elif severity_scores.get('moderate', 0) > 0.3:
                weight *= 1.5
            
            disorder_weights[disorder] = weight
        
        # Return disorder with highest weight
        return max(disorder_weights.items(), key=lambda x: x[1])[0]
    
    def _assess_risk_level(self, entities: Dict[str, List[str]], 
                          severity_scores: Dict[str, float], text: str) -> str:
        """Assess overall risk level"""
        risk_score = 0.0
        
        # Risk from disorders
        high_risk_disorders = ['trauma', 'psychosis', 'addiction']
        medium_risk_disorders = ['depression', 'bipolar']
        
        for disorder in entities['disorders']:
            if disorder in high_risk_disorders:
                risk_score += 3.0
            elif disorder in medium_risk_disorders:
                risk_score += 2.0
            else:
                risk_score += 1.0
        
        # Risk from severity
        risk_score += severity_scores.get('severe', 0) * 3.0
        risk_score += severity_scores.get('critical', 0) * 5.0
        
        # Risk from crisis indicators
        crisis_terms = ['suicide', 'kill myself', 'end it all', 'can\'t go on', 'hopeless']
        for term in crisis_terms:
            if term in text.lower():
                risk_score += 5.0
                break
        
        # Convert to categorical risk level
        if risk_score >= 5.0:
            return 'high'
        elif risk_score >= 2.0:
            return 'moderate'
        elif risk_score >= 1.0:
            return 'low'
        else:
            return 'minimal'
    
    def process_diagnosis_data(self, data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Process diagnosis data from various formats"""
        # Extract text from different input formats
        if isinstance(data, str):
            text = data
            structured_data = {}
        elif isinstance(data, dict):
            # Try to extract text from common fields
            text_fields = ['diagnosis', 'description', 'symptoms', 'history', 'notes', 'text']
            text_parts = []
            
            for field in text_fields:
                if field in data and data[field]:
                    text_parts.append(str(data[field]))
            
            text = ' '.join(text_parts) if text_parts else str(data)
            structured_data = data
        else:
            text = str(data)
            structured_data = {}
        
        if not text.strip():
            return self._empty_result("Empty input")
        
        # Create embedding
        result = self.create_diagnosis_embedding(text)
        
        # Add metadata
        result.update({
            'original_text': text,
            'structured_data': structured_data,
            'model_used': self.model_name if self.model is not None else 'fallback',
            'text_length': len(text.split())
        })
        
        return result
    
    def _empty_result(self, error: str = None) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'embedding': [0.0] * 128,
            'entities': {cat: [] for cat in ['disorders', 'symptoms', 'medications', 'procedures', 'anatomy']},
            'severity': {level: 0.0 for level in self.severity_indicators.keys()},
            'clinical_features': {},
            'primary_disorder': 'unspecified',
            'risk_level': 'minimal',
            'original_text': '',
            'structured_data': {},
            'model_used': 'none',
            'text_length': 0,
            'error': error
        }


def main():
    """CLI interface for diagnosis encoder"""
    parser = argparse.ArgumentParser(description="Diagnosis Encoder for meditation module")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSON file with diagnosis data")
    parser.add_argument("--output", type=str, default="preprocess_output/diagnosis_encoded.json",
                       help="Output JSON file")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                       help="Transformer model name")
    parser.add_argument("--use-biobert", action="store_true",
                       help="Use BioBERT for medical text")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum text length for transformer")
    
    args = parser.parse_args()
    
    # Initialize encoder
    encoder = DiagnosisEncoder(
        model_name=args.model,
        use_biobert=args.use_biobert,
        max_length=args.max_length
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
        for item in input_data:
            result = encoder.process_diagnosis_data(item)
            results.append(result)
    else:
        results = [encoder.process_diagnosis_data(input_data)]
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Diagnosis encoding complete. Processed {len(results)} records.")
    print(f"Results saved to {output_path}")
    
    # Summary statistics
    if results:
        primary_disorders = [r.get('primary_disorder', 'unspecified') for r in results]
        risk_levels = [r.get('risk_level', 'minimal') for r in results]
        
        print(f"\nSummary:")
        print(f"Primary disorders: {dict(zip(*np.unique(primary_disorders, return_counts=True)))}")
        print(f"Risk levels: {dict(zip(*np.unique(risk_levels, return_counts=True)))}")


if __name__ == "__main__":
    main()