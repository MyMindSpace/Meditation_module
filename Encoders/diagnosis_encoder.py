import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn.functional as F
except ImportError:
    raise RuntimeError("Transformers and PyTorch are required. Install with: pip install transformers torch numpy")

try:
    import spacy
    from spacy import displacy
except ImportError:
    spacy = None
    print("Warning: spaCy not found. Medical entity recognition will be limited.")


class DiagnosisEncoder:
    """
    Diagnosis Encoder (DE) for generating numerical embeddings from preprocessed diagnosis text.
    
    Features:
    - BERT/BioBERT embeddings for context-aware text representation
    - Medical Entity Recognition for identifying medical terms
    - Symptom embeddings for dense vector representations
    """
    
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", device: str = "auto"):
        """
        Initialize the diagnosis encoder.
        
        Args:
            model_name: HuggingFace model name (BioBERT or BERT)
            device: Device to run model on ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        
        # Load tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Load spaCy model for medical NER
        self.nlp = None
        if spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
        
        # Medical entity categories
        self.medical_entities = {
            'diseases': [],
            'symptoms': [],
            'treatments': [],
            'body_parts': [],
            'medications': []
        }
        
        # Symptom embedding cache
        self.symptom_embeddings = {}
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def encode_text(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Generate BERT/BioBERT embeddings for text.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Contextual embeddings (768-dim for BioBERT)
        """
        if not text.strip():
            return np.zeros(768, dtype=np.float32)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings[0]  # Return first (and only) sequence
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities using spaCy NER.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            'diseases': [],
            'symptoms': [],
            'treatments': [],
            'body_parts': [],
            'medications': []
        }
        
        if not self.nlp or not text.strip():
            return entities
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entity_text = ent.text.lower().strip()
            label = ent.label_
            
            # Map spaCy labels to our categories
            if label in ['DISEASE', 'CONDITION']:
                entities['diseases'].append(entity_text)
            elif label in ['SYMPTOM', 'SIGN']:
                entities['symptoms'].append(entity_text)
            elif label in ['TREATMENT', 'PROCEDURE']:
                entities['treatments'].append(entity_text)
            elif label in ['ANATOMY', 'BODY_PART']:
                entities['body_parts'].append(entity_text)
            elif label in ['MEDICATION', 'DRUG']:
                entities['medications'].append(entity_text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def create_symptom_embeddings(self, symptoms: List[str]) -> Dict[str, np.ndarray]:
        """
        Create dense vector representations for individual symptoms.
        
        Args:
            symptoms: List of symptom strings
            
        Returns:
            Dictionary mapping symptoms to their embeddings
        """
        symptom_embeddings = {}
        
        for symptom in symptoms:
            if not symptom.strip():
                continue
                
            # Check cache first
            if symptom in self.symptom_embeddings:
                symptom_embeddings[symptom] = self.symptom_embeddings[symptom]
                continue
            
            # Generate embedding
            embedding = self.encode_text(symptom)
            symptom_embeddings[symptom] = embedding
            self.symptom_embeddings[symptom] = embedding
        
        return symptom_embeddings
    
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
        main_embedding = self.encode_text(cleaned_text)
        
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
            aggregated_symptom_embedding = np.zeros(768, dtype=np.float32)
        
        # Create entity embeddings for other medical terms
        entity_embeddings = {}
        for entity_type, entities in medical_entities.items():
            if entity_type == 'symptoms':  # Already handled above
                continue
            if entities:
                entity_text = ' '.join(entities)
                entity_embeddings[entity_type] = self.encode_text(entity_text)
            else:
                entity_embeddings[entity_type] = np.zeros(768, dtype=np.float32)
        
        # Combine all embeddings
        combined_embedding = np.concatenate([
            main_embedding,  # 768 dims
            aggregated_symptom_embedding,  # 768 dims
            entity_embeddings.get('diseases', np.zeros(768)),  # 768 dims
            entity_embeddings.get('treatments', np.zeros(768)),  # 768 dims
            entity_embeddings.get('body_parts', np.zeros(768)),  # 768 dims
            entity_embeddings.get('medications', np.zeros(768))  # 768 dims
        ])  # Total: 4608 dims
        
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
                'diseases': entity_embeddings.get('diseases', np.zeros(768)).tolist(),
                'treatments': entity_embeddings.get('treatments', np.zeros(768)).tolist(),
                'body_parts': entity_embeddings.get('body_parts', np.zeros(768)).tolist(),
                'medications': entity_embeddings.get('medications', np.zeros(768)).tolist(),
                'combined': combined_embedding.tolist()
            },
            'embedding_dimensions': {
                'main_text': len(main_embedding),
                'symptoms_individual': len(aggregated_symptom_embedding),
                'combined': len(combined_embedding)
            }
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
    parser = argparse.ArgumentParser(description="Diagnosis Encoder (DE) for generating numerical embeddings")
    parser.add_argument("--input", type=str, default="preprocess_output/diagnosis_processed.json", 
                       help="Path to preprocessed diagnosis data")
    parser.add_argument("--output", type=str, default="preprocess_output/diagnosis_encoded.json",
                       help="Path to save encoded diagnosis data")
    parser.add_argument("--model", type=str, default="dmis-lab/biobert-base-cased-v1.1",
                       help="HuggingFace model name (BioBERT or BERT)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run model on (auto, cpu, cuda)")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {input_path}...")
    records = load_preprocessed_data(input_path)
    
    # Initialize encoder
    encoder = DiagnosisEncoder(model_name=args.model, device=args.device)
    
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


if __name__ == "__main__":
    main()
