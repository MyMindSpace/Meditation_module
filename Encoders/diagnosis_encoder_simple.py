import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import hashlib
import re


class SimpleDiagnosisEncoder:
    """
    Simple Diagnosis Encoder (DE) that generates numerical embeddings without heavy ML dependencies.
    
    Features:
    - TF-IDF style embeddings for text representation
    - Simple medical entity recognition using keyword matching
    - Symptom embeddings using hash-based vectors
    """
    
    def __init__(self):
        """Initialize the simple diagnosis encoder."""
        
        # Medical entity keywords
        self.medical_keywords = {
            'diseases': {
                'anxiety', 'depression', 'stress', 'adhd', 'insomnia', 'ptsd', 'ocd', 
                'bipolar', 'panic', 'trauma', 'burnout', 'chronic', 'acute'
            },
            'symptoms': {
                'racing', 'thoughts', 'difficulty', 'concentrating', 'restlessness', 
                'tension', 'fatigue', 'irritability', 'sleep', 'disturbances', 
                'headache', 'pain', 'nausea', 'dizzy', 'muscle', 'joint', 'back', 'neck'
            },
            'treatments': {
                'medication', 'therapy', 'counseling', 'meditation', 'exercise', 
                'treatment', 'intervention', 'support', 'care'
            },
            'body_parts': {
                'head', 'neck', 'back', 'chest', 'muscle', 'joint', 'brain', 'heart', 
                'stomach', 'leg', 'arm', 'shoulder'
            },
            'medications': {
                'medication', 'drug', 'pill', 'tablet', 'injection', 'prescription'
            }
        }
        
        # Severity keywords with weights
        self.severity_weights = {
            'mild': 1, 'moderate': 2, 'severe': 3, 'extreme': 4, 'chronic': 3,
            'acute': 2, 'occasional': 1, 'frequent': 3, 'constant': 4, 'intermittent': 2
        }
        
        # Create vocabulary for TF-IDF style embeddings
        self.vocabulary = set()
        for category in self.medical_keywords.values():
            self.vocabulary.update(category)
        self.vocabulary = sorted(list(self.vocabulary))
        self.vocab_size = len(self.vocabulary)
        self.vocab_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return self.clean_text(text).split()
    
    def create_tfidf_embedding(self, text: str) -> np.ndarray:
        """
        Create TF-IDF style embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            TF-IDF style embedding vector
        """
        tokens = self.tokenize(text)
        if not tokens:
            return np.zeros(self.vocab_size, dtype=np.float32)
        
        # Count term frequencies
        term_counts = {}
        for token in tokens:
            if token in self.vocab_to_idx:
                term_counts[token] = term_counts.get(token, 0) + 1
        
        # Create embedding
        embedding = np.zeros(self.vocab_size, dtype=np.float32)
        total_terms = len(tokens)
        
        for term, count in term_counts.items():
            idx = self.vocab_to_idx[term]
            # Simple TF-IDF: tf * log(total_terms / count)
            tf = count / total_terms
            idf = np.log(total_terms / count) if count > 0 else 0
            embedding[idx] = tf * idf
        
        return embedding
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities using keyword matching.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        tokens = self.tokenize(text)
        entities = {category: [] for category in self.medical_keywords.keys()}
        
        for token in tokens:
            for category, keywords in self.medical_keywords.items():
                if token in keywords:
                    entities[category].append(token)
        
        # Remove duplicates
        for category in entities:
            entities[category] = list(set(entities[category]))
        
        return entities
    
    def create_symptom_embeddings(self, symptoms: List[str]) -> Dict[str, np.ndarray]:
        """
        Create hash-based embeddings for symptoms.
        
        Args:
            symptoms: List of symptom strings
            
        Returns:
            Dictionary mapping symptoms to their embeddings
        """
        symptom_embeddings = {}
        embedding_dim = 128  # Fixed dimension for hash embeddings
        
        for symptom in symptoms:
            if not symptom.strip():
                continue
            
            # Create hash-based embedding
            hash_obj = hashlib.md5(symptom.encode())
            hash_bytes = hash_obj.digest()
            
            # Convert to float vector
            embedding = np.zeros(embedding_dim, dtype=np.float32)
            for i, byte_val in enumerate(hash_bytes):
                if i < embedding_dim:
                    embedding[i] = (byte_val - 128) / 128.0  # Normalize to [-1, 1]
            
            symptom_embeddings[symptom] = embedding
        
        return symptom_embeddings
    
    def calculate_severity_score(self, text: str) -> float:
        """Calculate severity score from text."""
        tokens = self.tokenize(text)
        severity_scores = []
        
        for token in tokens:
            if token in self.severity_weights:
                severity_scores.append(self.severity_weights[token])
        
        if not severity_scores:
            return 1.0  # Default mild severity
        
        return sum(severity_scores) / len(severity_scores)
    
    def process_diagnosis_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single diagnosis record to generate embeddings.
        
        Args:
            record: Preprocessed diagnosis record
            
        Returns:
            Record with numerical embeddings
        """
        # Extract text components
        original_text = record.get('original_text', '')
        cleaned_text = record.get('cleaned_text', '')
        tokens = record.get('tokens', [])
        
        # Generate main text embedding
        main_embedding = self.create_tfidf_embedding(cleaned_text)
        
        # Extract medical entities
        medical_entities = self.extract_medical_entities(cleaned_text)
        
        # Create symptom embeddings
        symptoms = medical_entities.get('symptoms', [])
        symptom_embeddings = self.create_symptom_embeddings(symptoms)
        
        # Aggregate symptom embeddings (mean pooling)
        if symptom_embeddings:
            symptom_vectors = list(symptom_embeddings.values())
            aggregated_symptom_embedding = np.mean(symptom_vectors, axis=0)
        else:
            aggregated_symptom_embedding = np.zeros(128, dtype=np.float32)
        
        # Create entity embeddings for other medical terms
        entity_embeddings = {}
        for entity_type, entities in medical_entities.items():
            if entity_type == 'symptoms':  # Already handled above
                continue
            if entities:
                entity_text = ' '.join(entities)
                entity_embeddings[entity_type] = self.create_tfidf_embedding(entity_text)
            else:
                entity_embeddings[entity_type] = np.zeros(self.vocab_size, dtype=np.float32)
        
        # Calculate severity score
        severity_score = self.calculate_severity_score(cleaned_text)
        
        # Combine all embeddings
        combined_embedding = np.concatenate([
            main_embedding,  # vocab_size dims
            aggregated_symptom_embedding,  # 128 dims
            entity_embeddings.get('diseases', np.zeros(self.vocab_size)),  # vocab_size dims
            entity_embeddings.get('treatments', np.zeros(self.vocab_size)),  # vocab_size dims
            entity_embeddings.get('body_parts', np.zeros(self.vocab_size)),  # vocab_size dims
            entity_embeddings.get('medications', np.zeros(self.vocab_size)),  # vocab_size dims
            [severity_score]  # 1 dim
        ])  # Total: 5*vocab_size + 128 + 1 dims
        
        return {
            'userId': record.get('userId'),
            'original_text': original_text,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'features': record.get('features', {}),
            'medical_entities': medical_entities,
            'embeddings': {
                'main_text': main_embedding.tolist(),
                'symptoms': {k: v.tolist() for k, v in symptom_embeddings.items()},
                'aggregated_symptoms': aggregated_symptom_embedding.tolist(),
                'diseases': entity_embeddings.get('diseases', np.zeros(self.vocab_size)).tolist(),
                'treatments': entity_embeddings.get('treatments', np.zeros(self.vocab_size)).tolist(),
                'body_parts': entity_embeddings.get('body_parts', np.zeros(self.vocab_size)).tolist(),
                'medications': entity_embeddings.get('medications', np.zeros(self.vocab_size)).tolist(),
                'combined': combined_embedding.tolist()
            },
            'embedding_dimensions': {
                'main_text': len(main_embedding),
                'symptoms_individual': len(aggregated_symptom_embedding),
                'combined': len(combined_embedding),
                'vocabulary_size': self.vocab_size
            },
            'severity_score': severity_score
        }


def load_preprocessed_data(input_path: Path) -> List[Dict[str, Any]]:
    """Load preprocessed diagnosis data."""
    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of diagnosis records")
    
    return data


def save_encoded_data(output_path: Path, data: List[Dict[str, Any]]) -> None:
    """Save encoded diagnosis data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Simple Diagnosis Encoder (DE) for generating numerical embeddings")
    parser.add_argument("--input", type=str, default="preprocess_output/diagnosis_processed.json", 
                       help="Path to preprocessed diagnosis data")
    parser.add_argument("--output", type=str, default="preprocess_output/diagnosis_encoded.json",
                       help="Path to save encoded diagnosis data")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {input_path}...")
    records = load_preprocessed_data(input_path)
    
    # Initialize encoder
    encoder = SimpleDiagnosisEncoder()
    
    # Process records
    print(f"Processing {len(records)} diagnosis records...")
    encoded_records = []
    
    for i, record in enumerate(records):
        print(f"Processing record {i+1}/{len(records)}...")
        try:
            encoded_record = encoder.process_diagnosis_record(record)
            encoded_records.append(encoded_record)
        except Exception as e:
            print(f"Error processing record {i+1}: {e}")
            continue
    
    # Save encoded data
    save_encoded_data(output_path, encoded_records)
    print(f"Saved {len(encoded_records)} encoded records to {output_path}")
    
    # Print summary
    if encoded_records:
        sample_record = encoded_records[0]
        print(f"\nEmbedding dimensions:")
        for key, dim in sample_record['embedding_dimensions'].items():
            print(f"  {key}: {dim}")
        
        print(f"\nSample medical entities found:")
        entities = sample_record['medical_entities']
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"  {entity_type}: {entity_list}")
        
        print(f"\nVocabulary size: {sample_record['embedding_dimensions']['vocabulary_size']}")


if __name__ == "__main__":
    main()
