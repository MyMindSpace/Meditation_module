# Encoders/audio_encoder.py
"""
Audio Encoder (AE)

Processes audio data to extract:
- Mel-frequency features from MFCCs
- Voice Activity Detection (VAD)
- Emotion recognition from voice patterns
- Audio embeddings for ARM
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional imports with fallbacks
try:
    import librosa
    import librosa.feature
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AudioEncoder:
    """
    Audio Encoder for meditation module
    Extracts audio features and emotion embeddings from voice input
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 frame_length: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 n_mfcc: int = 13):
        
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
        # Initialize scalers for feature normalization
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=32)  # Reduce dimensionality
        else:
            self.scaler = None
            self.pca = None
        
        # Emotion mapping (heuristic-based)
        self.emotion_classes = [
            'calm', 'happy', 'sad', 'angry', 'fear', 'surprise', 'neutral'
        ]
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return samples with sample rate"""
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for audio processing")
            
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file {audio_path}: {e}")
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio: normalize, remove silence"""
        if audio.size == 0:
            return audio
            
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Remove leading/trailing silence using energy-based VAD
        if LIBROSA_AVAILABLE:
            # Compute short-time energy
            frame_length = min(1024, len(audio) // 4)
            energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=frame_length//2)[0]
            
            # Find non-silent regions (threshold = 1% of max energy)
            energy_threshold = 0.01 * np.max(energy)
            non_silent = energy > energy_threshold
            
            if np.any(non_silent):
                # Convert frame indices to sample indices
                hop = frame_length // 2
                start_frame = np.where(non_silent)[0][0]
                end_frame = np.where(non_silent)[0][-1]
                
                start_sample = max(0, start_frame * hop - hop)
                end_sample = min(len(audio), (end_frame + 1) * hop + hop)
                
                audio = audio[start_sample:end_sample]
        
        return audio
    
    def extract_mfcc_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract MFCC features from audio"""
        if not LIBROSA_AVAILABLE:
            # Fallback: basic spectral features
            return self._extract_basic_features(audio)
        
        if audio.size == 0:
            return {
                'mfcc': np.zeros((self.n_mfcc, 1)),
                'mfcc_delta': np.zeros((self.n_mfcc, 1)),
                'mfcc_delta2': np.zeros((self.n_mfcc, 1))
            }
        
        try:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.frame_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Compute delta and delta-delta features
            mfcc_delta = librosa.feature.delta(mfcc, order=1)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            return {
                'mfcc': mfcc,
                'mfcc_delta': mfcc_delta,
                'mfcc_delta2': mfcc_delta2
            }
            
        except Exception as e:
            print(f"Warning: MFCC extraction failed: {e}")
            return {
                'mfcc': np.zeros((self.n_mfcc, 1)),
                'mfcc_delta': np.zeros((self.n_mfcc, 1)),
                'mfcc_delta2': np.zeros((self.n_mfcc, 1))
            }
    
    def _extract_basic_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Fallback basic spectral features when librosa is not available"""
        if audio.size == 0:
            return {
                'mfcc': np.zeros((self.n_mfcc, 1)),
                'mfcc_delta': np.zeros((self.n_mfcc, 1)), 
                'mfcc_delta2': np.zeros((self.n_mfcc, 1))
            }
        
        # Simple spectral features using numpy FFT
        # Compute short-time FFT
        frame_length = min(self.frame_length, len(audio))
        hop_length = min(self.hop_length, frame_length // 2)
        
        frames = []
        for i in range(0, len(audio) - frame_length + 1, hop_length):
            frame = audio[i:i + frame_length]
            # Apply window
            window = np.hanning(len(frame))
            frame_windowed = frame * window
            frames.append(frame_windowed)
        
        if not frames:
            frames = [audio]
            
        # Compute magnitude spectrum for each frame
        spectra = []
        for frame in frames:
            spectrum = np.abs(np.fft.fft(frame))[:frame_length // 2]
            spectra.append(spectrum)
        
        spectra = np.array(spectra).T  # [freq_bins, time_frames]
        
        # Extract basic features (simulate MFCC structure)
        # Take log of spectral magnitudes
        log_spectra = np.log(spectra + 1e-8)
        
        # Downsample to n_mfcc features
        if log_spectra.shape[0] >= self.n_mfcc:
            indices = np.linspace(0, log_spectra.shape[0] - 1, self.n_mfcc).astype(int)
            mfcc = log_spectra[indices]
        else:
            # Pad with zeros if not enough frequency bins
            mfcc = np.zeros((self.n_mfcc, log_spectra.shape[1]))
            mfcc[:log_spectra.shape[0]] = log_spectra
        
        # Simple delta approximation
        if mfcc.shape[1] > 2:
            mfcc_delta = np.diff(mfcc, axis=1, prepend=mfcc[:, 0:1])
            mfcc_delta2 = np.diff(mfcc_delta, axis=1, prepend=mfcc_delta[:, 0:1])
        else:
            mfcc_delta = np.zeros_like(mfcc)
            mfcc_delta2 = np.zeros_like(mfcc)
        
        return {
            'mfcc': mfcc,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2
        }
    
    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract prosodic features for emotion recognition"""
        if audio.size == 0:
            return {
                'f0_mean': 0.0, 'f0_std': 0.0, 'f0_range': 0.0,
                'energy_mean': 0.0, 'energy_std': 0.0,
                'zcr_mean': 0.0, 'zcr_std': 0.0,
                'spectral_centroid': 0.0, 'spectral_bandwidth': 0.0
            }
        
        features = {}
        
        if LIBROSA_AVAILABLE:
            # Fundamental frequency (pitch)
            try:
                f0 = librosa.yin(audio, fmin=50, fmax=400, sr=self.sample_rate)
                f0_voiced = f0[f0 > 0]  # Remove unvoiced frames
                if len(f0_voiced) > 0:
                    features['f0_mean'] = float(np.mean(f0_voiced))
                    features['f0_std'] = float(np.std(f0_voiced))
                    features['f0_range'] = float(np.max(f0_voiced) - np.min(f0_voiced))
                else:
                    features['f0_mean'] = features['f0_std'] = features['f0_range'] = 0.0
            except:
                features['f0_mean'] = features['f0_std'] = features['f0_range'] = 0.0
            
            # Energy features
            try:
                energy = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
                features['energy_mean'] = float(np.mean(energy))
                features['energy_std'] = float(np.std(energy))
            except:
                features['energy_mean'] = features['energy_std'] = 0.0
            
            # Zero crossing rate
            try:
                zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
                features['zcr_mean'] = float(np.mean(zcr))
                features['zcr_std'] = float(np.std(zcr))
            except:
                features['zcr_mean'] = features['zcr_std'] = 0.0
            
            # Spectral features
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
                features['spectral_centroid'] = float(np.mean(spectral_centroids))
                
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
                features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
            except:
                features['spectral_centroid'] = features['spectral_bandwidth'] = 0.0
                
        else:
            # Basic fallback features
            # Energy (RMS)
            energy = np.sqrt(np.mean(audio**2))
            features['energy_mean'] = float(energy)
            features['energy_std'] = 0.0
            
            # Zero crossing rate
            zcr = np.mean(librosa.zero_crossings(audio, pad=False)) if hasattr(librosa, 'zero_crossings') else 0.0
            features['zcr_mean'] = float(zcr)
            features['zcr_std'] = 0.0
            
            # Set defaults for unavailable features
            for key in ['f0_mean', 'f0_std', 'f0_range', 'spectral_centroid', 'spectral_bandwidth']:
                features[key] = 0.0
        
        return features
    
    def detect_voice_activity(self, audio: np.ndarray, threshold: float = 0.01) -> Tuple[float, List[Tuple[float, float]]]:
        """Detect voice activity and return VAD ratio and segments"""
        if audio.size == 0:
            return 0.0, []
        
        if LIBROSA_AVAILABLE:
            # Energy-based VAD using librosa
            frame_length = 1024
            hop_length = 512
            
            try:
                energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
                energy_threshold = threshold * np.max(energy)
                
                # Find voiced frames
                voiced_frames = energy > energy_threshold
                vad_ratio = np.sum(voiced_frames) / len(voiced_frames)
                
                # Find continuous segments
                segments = []
                in_segment = False
                start_time = 0.0
                
                for i, is_voiced in enumerate(voiced_frames):
                    time = i * hop_length / self.sample_rate
                    
                    if is_voiced and not in_segment:
                        start_time = time
                        in_segment = True
                    elif not is_voiced and in_segment:
                        segments.append((start_time, time))
                        in_segment = False
                
                # Handle case where audio ends while in a segment
                if in_segment:
                    segments.append((start_time, len(audio) / self.sample_rate))
                
                return float(vad_ratio), segments
                
            except Exception as e:
                print(f"Warning: VAD failed: {e}")
                return 0.0, []
        else:
            # Simple energy-based VAD
            frame_size = 1024
            frames = [audio[i:i+frame_size] for i in range(0, len(audio), frame_size)]
            
            energies = [np.sqrt(np.mean(frame**2)) for frame in frames if len(frame) > 0]
            if not energies:
                return 0.0, []
            
            energy_threshold = threshold * max(energies)
            voiced_frames = [e > energy_threshold for e in energies]
            vad_ratio = sum(voiced_frames) / len(voiced_frames)
            
            return float(vad_ratio), []
    
    def recognize_emotion(self, audio: np.ndarray, mfcc_features: Dict[str, np.ndarray], 
                         prosodic_features: Dict[str, float]) -> Dict[str, float]:
        """Heuristic-based emotion recognition"""
        if audio.size == 0:
            # Return neutral emotion
            emotion_probs = {emotion: 0.0 for emotion in self.emotion_classes}
            emotion_probs['neutral'] = 1.0
            return emotion_probs
        
        # Heuristic rules based on prosodic features
        f0_mean = prosodic_features.get('f0_mean', 0.0)
        f0_std = prosodic_features.get('f0_std', 0.0)
        energy_mean = prosodic_features.get('energy_mean', 0.0)
        zcr_mean = prosodic_features.get('zcr_mean', 0.0)
        
        # Initialize emotion scores
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_classes}
        
        # Heuristic rules (these are simplified and should be replaced with trained models)
        
        # High pitch, high energy -> happy/surprise
        if f0_mean > 150 and energy_mean > 0.05:
            emotion_scores['happy'] += 0.3
            emotion_scores['surprise'] += 0.2
        
        # Low pitch, low energy -> sad
        if f0_mean < 120 and energy_mean < 0.03:
            emotion_scores['sad'] += 0.4
        
        # High pitch variation, high energy -> angry
        if f0_std > 50 and energy_mean > 0.06:
            emotion_scores['angry'] += 0.3
        
        # Low pitch, moderate energy -> calm
        if 100 < f0_mean < 140 and 0.02 < energy_mean < 0.05:
            emotion_scores['calm'] += 0.4
        
        # High zero crossing rate -> fear/anxiety
        if zcr_mean > 0.1:
            emotion_scores['fear'] += 0.2
        
        # Add MFCC-based rules
        if 'mfcc' in mfcc_features:
            mfcc = mfcc_features['mfcc']
            if mfcc.size > 0:
                # First coefficient relates to energy
                mfcc_energy = np.mean(mfcc[0])
                if mfcc_energy > 0:
                    emotion_scores['happy'] += 0.1
                else:
                    emotion_scores['sad'] += 0.1
                
                # Higher order coefficients relate to timbre
                if mfcc.shape[0] > 2:
                    mfcc_std = np.std(mfcc[1:3])  # Spectral variability
                    if mfcc_std > 1.0:
                        emotion_scores['angry'] += 0.1
                    else:
                        emotion_scores['calm'] += 0.1
        
        # Normalize scores to probabilities
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_probs = {k: v / total_score for k, v in emotion_scores.items()}
        else:
            # Default to neutral if no strong indicators
            emotion_probs = {emotion: 0.0 for emotion in self.emotion_classes}
            emotion_probs['neutral'] = 1.0
        
        return emotion_probs
    
    def create_audio_embedding(self, mfcc_features: Dict[str, np.ndarray], 
                              prosodic_features: Dict[str, float],
                              emotion_probs: Dict[str, float]) -> np.ndarray:
        """Create unified audio embedding"""
        # Aggregate MFCC features
        mfcc_embedding = []
        for feature_name in ['mfcc', 'mfcc_delta', 'mfcc_delta2']:
            if feature_name in mfcc_features:
                feat = mfcc_features[feature_name]
                if feat.size > 0:
                    # Statistical aggregation across time
                    mfcc_embedding.extend([
                        float(np.mean(feat)),
                        float(np.std(feat)),
                        float(np.min(feat)),
                        float(np.max(feat))
                    ])
                else:
                    mfcc_embedding.extend([0.0, 0.0, 0.0, 0.0])
            else:
                mfcc_embedding.extend([0.0, 0.0, 0.0, 0.0])
        
        # Add prosodic features
        prosodic_embedding = list(prosodic_features.values())
        
        # Add emotion probabilities
        emotion_embedding = [emotion_probs.get(emotion, 0.0) for emotion in self.emotion_classes]
        
        # Combine all features
        full_embedding = np.array(mfcc_embedding + prosodic_embedding + emotion_embedding, dtype=np.float32)
        
        # Optional: Apply PCA for dimensionality reduction
        if self.pca is not None and len(full_embedding) > 32:
            try:
                # Reshape for PCA (needs 2D input)
                embedding_2d = full_embedding.reshape(1, -1)
                reduced_embedding = self.pca.fit_transform(embedding_2d)
                full_embedding = reduced_embedding.flatten()
            except:
                # Keep original if PCA fails
                pass
        
        return full_embedding
    
    def process_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Process single audio file and return all features"""
        try:
            # Load and preprocess audio
            audio, sr = self.load_audio(audio_path)
            audio = self.preprocess_audio(audio)
            
            if audio.size == 0:
                return self._empty_result(audio_path)
            
            # Extract features
            mfcc_features = self.extract_mfcc_features(audio)
            prosodic_features = self.extract_prosodic_features(audio)
            vad_ratio, vad_segments = self.detect_voice_activity(audio)
            emotion_probs = self.recognize_emotion(audio, mfcc_features, prosodic_features)
            
            # Create unified embedding
            audio_embedding = self.create_audio_embedding(mfcc_features, prosodic_features, emotion_probs)
            
            return {
                "file": audio_path,
                "duration": len(audio) / self.sample_rate,
                "sample_rate": self.sample_rate,
                "vad_ratio": vad_ratio,
                "vad_segments": vad_segments,
                "embeddings": {
                    "audio": audio_embedding.tolist(),
                    "mfcc_stats": {
                        name: {
                            "mean": float(np.mean(feat)) if feat.size > 0 else 0.0,
                            "std": float(np.std(feat)) if feat.size > 0 else 0.0
                        } for name, feat in mfcc_features.items()
                    },
                    "prosodic": prosodic_features,
                    "emotion": list(emotion_probs.values())
                },
                "emotion_labels": self.emotion_classes,
                "dominant_emotion": max(emotion_probs.items(), key=lambda x: x[1])[0]
            }
            
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}")
            return self._empty_result(audio_path, str(e))
    
    def _empty_result(self, file_path: str, error: str = None) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            "file": file_path,
            "duration": 0.0,
            "sample_rate": self.sample_rate,
            "vad_ratio": 0.0,
            "vad_segments": [],
            "embeddings": {
                "audio": [0.0] * 32,  # Default embedding size
                "mfcc_stats": {},
                "prosodic": {key: 0.0 for key in ['f0_mean', 'f0_std', 'f0_range', 'energy_mean', 'energy_std', 'zcr_mean', 'zcr_std', 'spectral_centroid', 'spectral_bandwidth']},
                "emotion": [0.0] * len(self.emotion_classes)
            },
            "emotion_labels": self.emotion_classes,
            "dominant_emotion": "neutral",
            "error": error
        }


def main():
    """CLI interface for audio encoder"""
    parser = argparse.ArgumentParser(description="Audio Encoder for meditation module")
    parser.add_argument("--input", type=str, required=True,
                       help="Input audio file or directory")
    parser.add_argument("--output", type=str, default="preprocess_output/audio_encoded.json",
                       help="Output JSON file")
    parser.add_argument("--sample-rate", type=int, default=16000,
                       help="Audio sample rate for processing")
    parser.add_argument("--n-mfcc", type=int, default=13,
                       help="Number of MFCC coefficients")
    parser.add_argument("--n-mels", type=int, default=128,
                       help="Number of mel frequency bands")
    
    args = parser.parse_args()
    
    # Initialize encoder
    encoder = AudioEncoder(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        n_mels=args.n_mels
    )
    
    # Process input
    input_path = Path(args.input)
    results = []
    
    if input_path.is_file():
        # Single file
        result = encoder.process_audio_file(str(input_path))
        results.append(result)
    elif input_path.is_dir():
        # Directory of audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f"**/*{ext}"))
        
        for audio_file in sorted(audio_files):
            result = encoder.process_audio_file(str(audio_file))
            results.append(result)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Audio encoding complete. Processed {len(results)} files.")
    print(f"Results saved to {output_path}")
    
    # Summary statistics
    if results:
        total_duration = sum(r.get("duration", 0.0) for r in results)
        avg_vad = np.mean([r.get("vad_ratio", 0.0) for r in results])
        emotions = [r.get("dominant_emotion", "neutral") for r in results]
        emotion_counts = {e: emotions.count(e) for e in set(emotions)}
        
        print(f"\nSummary:")
        print(f"Total audio duration: {total_duration:.2f}s")
        print(f"Average VAD ratio: {avg_vad:.2%}")
        print(f"Emotion distribution: {emotion_counts}")


if __name__ == "__main__":
    main()