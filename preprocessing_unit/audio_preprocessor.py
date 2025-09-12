# preprocessing_unit/audio_preprocessor.py
"""
Audio Preprocessor (AP)

Handles audio preprocessing tasks:
- MFCC feature extraction
- Spectrogram generation  
- Noise reduction
- Audio normalization
- Voice activity detection preparation
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
    from scipy import signal
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class AudioPreprocessor:
    """
    Audio preprocessor for meditation module
    Handles audio feature extraction and preprocessing
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 frame_length: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 n_mfcc: int = 13,
                 n_fft: int = 2048):
        
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        
        # Audio quality thresholds
        self.min_duration = 0.5  # Minimum duration in seconds
        self.max_duration = 300.0  # Maximum duration in seconds
        self.silence_threshold = 0.01  # Threshold for silence detection
    
    def load_audio(self, audio_path: str, 
                  offset: float = 0.0, 
                  duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Load audio file with resampling"""
        audio_path = str(audio_path)
        
        if LIBROSA_AVAILABLE:
            try:
                audio, sr = librosa.load(
                    audio_path, 
                    sr=self.sample_rate, 
                    offset=offset,
                    duration=duration
                )
                return audio, sr
            except Exception as e:
                print(f"Librosa loading failed: {e}")
                
        if SCIPY_AVAILABLE:
            try:
                sr, audio = wavfile.read(audio_path)
                
                # Convert to float32 and normalize
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                elif audio.dtype == np.uint8:
                    audio = (audio.astype(np.float32) - 128) / 128.0
                
                # Handle stereo to mono conversion
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Apply offset and duration
                if offset > 0:
                    start_sample = int(offset * sr)
                    audio = audio[start_sample:]
                
                if duration is not None:
                    end_sample = int(duration * sr)
                    audio = audio[:end_sample]
                
                # Resample if necessary
                if sr != self.sample_rate:
                    if SCIPY_AVAILABLE:
                        # Simple resampling using scipy
                        num_samples = int(len(audio) * self.sample_rate / sr)
                        audio = signal.resample(audio, num_samples)
                        sr = self.sample_rate
                    else:
                        print(f"Warning: Cannot resample from {sr} to {self.sample_rate}")
                
                return audio, sr
                
            except Exception as e:
                print(f"Scipy loading failed: {e}")
        
        # Fallback: create dummy audio
        print(f"Warning: Cannot load {audio_path}, using dummy audio")
        dummy_duration = duration or 1.0
        dummy_samples = int(dummy_duration * self.sample_rate)
        return np.zeros(dummy_samples, dtype=np.float32), self.sample_rate
    
    def assess_audio_quality(self, audio: np.ndarray) -> Dict[str, float]:
        """Assess audio quality metrics"""
        if audio.size == 0:
            return {
                "duration": 0.0,
                "rms_energy": 0.0,
                "peak_amplitude": 0.0,
                "snr_estimate": 0.0,
                "quality_score": 0.0
            }
        
        duration = len(audio) / self.sample_rate
        
        # Energy metrics
        rms_energy = float(np.sqrt(np.mean(audio**2)))
        peak_amplitude = float(np.max(np.abs(audio)))
        
        # Simple SNR estimation (signal vs background noise)
        # Use first and last 10% as noise estimate
        noise_samples = int(len(audio) * 0.1)
        if noise_samples > 0:
            noise_start = audio[:noise_samples]
            noise_end = audio[-noise_samples:]
            noise_level = np.sqrt(np.mean(np.concatenate([noise_start, noise_end])**2))
            
            if noise_level > 0:
                snr_estimate = 20 * np.log10(rms_energy / noise_level)
            else:
                snr_estimate = 60.0  # Very high SNR if no noise detected
        else:
            snr_estimate = 20.0  # Default moderate SNR
        
        # Quality score (0-1)
        duration_score = 1.0 if self.min_duration <= duration <= self.max_duration else 0.5
        energy_score = min(1.0, rms_energy * 10)  # Scale energy to 0-1
        snr_score = min(1.0, max(0.0, snr_estimate / 30.0))  # Normalize SNR
        
        quality_score = (duration_score + energy_score + snr_score) / 3.0
        
        return {
            "duration": duration,
            "rms_energy": rms_energy,
            "peak_amplitude": peak_amplitude,
            "snr_estimate": float(snr_estimate),
            "quality_score": quality_score
        }
    
    def remove_silence(self, audio: np.ndarray, 
                      threshold: Optional[float] = None) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Remove silence from audio and return non-silent segments"""
        if threshold is None:
            threshold = self.silence_threshold
        
        if audio.size == 0:
            return audio, []
        
        if LIBROSA_AVAILABLE:
            try:
                # Use librosa's silence removal
                intervals = librosa.effects.split(
                    audio, 
                    top_db=20,  # dB threshold for silence
                    frame_length=self.frame_length,
                    hop_length=self.hop_length
                )
                
                if len(intervals) > 0:
                    # Concatenate non-silent intervals
                    segments = []
                    time_segments = []
                    
                    for start, end in intervals:
                        segments.append(audio[start:end])
                        time_segments.append((start / self.sample_rate, end / self.sample_rate))
                    
                    if segments:
                        trimmed_audio = np.concatenate(segments)
                        return trimmed_audio, time_segments
                
            except Exception as e:
                print(f"Librosa silence removal failed: {e}")
        
        # Fallback: simple energy-based silence removal
        frame_size = self.hop_length
        frames = [audio[i:i+frame_size] for i in range(0, len(audio), frame_size)]
        
        non_silent_frames = []
        time_segments = []
        
        for i, frame in enumerate(frames):
            if frame.size > 0:
                frame_energy = np.sqrt(np.mean(frame**2))
                if frame_energy > threshold:
                    non_silent_frames.append(frame)
                    start_time = i * frame_size / self.sample_rate
                    end_time = (i + 1) * frame_size / self.sample_rate
                    time_segments.append((start_time, end_time))
        
        if non_silent_frames:
            trimmed_audio = np.concatenate(non_silent_frames)
            return trimmed_audio, time_segments
        else:
            return audio, [(0.0, len(audio) / self.sample_rate)]
    
    def normalize_audio(self, audio: np.ndarray, method: str = "peak") -> np.ndarray:
        """Normalize audio using specified method"""
        if audio.size == 0 or np.max(np.abs(audio)) == 0:
            return audio
        
        if method == "peak":
            # Peak normalization
            return audio / np.max(np.abs(audio))
        
        elif method == "rms":
            # RMS normalization to -20dB
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_rms = 10**(-20/20)  # -20dB
                return audio * (target_rms / rms)
            else:
                return audio
        
        elif method == "lufs":
            # Simplified LUFS-like normalization
            # Target: approximately -23 LUFS
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_rms = 0.1  # Approximate -20dB
                return audio * (target_rms / rms)
            else:
                return audio
        
        else:
            return audio
    
    def extract_mfcc_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract MFCC features"""
        if audio.size == 0:
            return {
                "mfcc": np.zeros((self.n_mfcc, 1)),
                "delta": np.zeros((self.n_mfcc, 1)),
                "delta2": np.zeros((self.n_mfcc, 1))
            }
        
        if LIBROSA_AVAILABLE:
            try:
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=self.sample_rate,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels
                )
                
                # Compute deltas
                delta = librosa.feature.delta(mfcc, order=1)
                delta2 = librosa.feature.delta(mfcc, order=2)
                
                return {
                    "mfcc": mfcc,
                    "delta": delta,
                    "delta2": delta2
                }
                
            except Exception as e:
                print(f"MFCC extraction failed: {e}")
        
        # Fallback: basic spectral features
        return self._extract_basic_spectral_features(audio)
    
    def _extract_basic_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Fallback spectral feature extraction"""
        if audio.size == 0:
            return {
                "mfcc": np.zeros((self.n_mfcc, 1)),
                "delta": np.zeros((self.n_mfcc, 1)),
                "delta2": np.zeros((self.n_mfcc, 1))
            }
        
        # Simple frame-based processing
        frame_length = self.frame_length
        hop_length = self.hop_length
        
        frames = []
        for i in range(0, len(audio) - frame_length + 1, hop_length):
            frame = audio[i:i + frame_length]
            
            # Apply window
            window = np.hanning(len(frame))
            windowed_frame = frame * window
            
            # FFT
            spectrum = np.abs(np.fft.fft(windowed_frame))[:frame_length // 2]
            
            # Log spectrum (simulating mel-scale processing)
            log_spectrum = np.log(spectrum + 1e-8)
            
            # Simulate MFCC by taking DCT-like transform
            if len(log_spectrum) >= self.n_mfcc:
                # Simple sampling to get n_mfcc features
                indices = np.linspace(0, len(log_spectrum) - 1, self.n_mfcc).astype(int)
                mfcc_frame = log_spectrum[indices]
            else:
                # Pad if spectrum too short
                mfcc_frame = np.pad(log_spectrum, (0, self.n_mfcc - len(log_spectrum)))[:self.n_mfcc]
            
            frames.append(mfcc_frame)
        
        if frames:
            mfcc = np.array(frames).T  # Shape: [n_mfcc, n_frames]
        else:
            mfcc = np.zeros((self.n_mfcc, 1))
        
        # Simple delta computation
        if mfcc.shape[1] > 2:
            delta = np.diff(mfcc, axis=1, prepend=mfcc[:, 0:1])
            delta2 = np.diff(delta, axis=1, prepend=delta[:, 0:1])
        else:
            delta = np.zeros_like(mfcc)
            delta2 = np.zeros_like(mfcc)
        
        return {
            "mfcc": mfcc,
            "delta": delta,
            "delta2": delta2
        }
    
    def generate_spectrogram(self, audio: np.ndarray, 
                           spec_type: str = "mel") -> np.ndarray:
        """Generate spectrogram"""
        if audio.size == 0:
            return np.zeros((self.n_mels, 1))
        
        if LIBROSA_AVAILABLE:
            try:
                if spec_type == "mel":
                    # Mel spectrogram
                    mel_spec = librosa.feature.melspectrogram(
                        y=audio,
                        sr=self.sample_rate,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        n_mels=self.n_mels
                    )
                    return librosa.power_to_db(mel_spec, ref=np.max)
                
                elif spec_type == "stft":
                    # Short-time Fourier transform
                    stft = librosa.stft(
                        audio,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length
                    )
                    return librosa.amplitude_to_db(np.abs(stft), ref=np.max)
                
            except Exception as e:
                print(f"Spectrogram generation failed: {e}")
        
        # Fallback: basic spectrogram using numpy FFT
        return self._generate_basic_spectrogram(audio)
    
    def _generate_basic_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Fallback spectrogram generation"""
        frame_length = self.n_fft
        hop_length = self.hop_length
        
        spectrograms = []
        
        for i in range(0, len(audio) - frame_length + 1, hop_length):
            frame = audio[i:i + frame_length]
            
            # Apply window
            window = np.hanning(len(frame))
            windowed_frame = frame * window
            
            # FFT
            spectrum = np.abs(np.fft.fft(windowed_frame))[:frame_length // 2]
            
            # Convert to dB
            db_spectrum = 20 * np.log10(spectrum + 1e-8)
            
            spectrograms.append(db_spectrum)
        
        if spectrograms:
            spectrogram = np.array(spectrograms).T
        else:
            spectrogram = np.zeros((frame_length // 2, 1))
        
        # Downsample to n_mels if needed
        if spectrogram.shape[0] > self.n_mels:
            indices = np.linspace(0, spectrogram.shape[0] - 1, self.n_mels).astype(int)
            spectrogram = spectrogram[indices]
        
        return spectrogram
    
    def process_audio_file(self, audio_path: str,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process complete audio file"""
        try:
            # Load audio
            audio, sr = self.load_audio(audio_path)
            
            if audio.size == 0:
                return {
                    "file": str(audio_path),
                    "status": "failed",
                    "error": "No audio data loaded"
                }
            
            # Assess quality
            quality_metrics = self.assess_audio_quality(audio)
            
            # Preprocessing
            normalized_audio = self.normalize_audio(audio, method="peak")
            trimmed_audio, voice_segments = self.remove_silence(normalized_audio)
            
            # Feature extraction
            mfcc_features = self.extract_mfcc_features(trimmed_audio)
            mel_spectrogram = self.generate_spectrogram(trimmed_audio, "mel")
            
            # Prepare result
            result = {
                "file": str(audio_path),
                "status": "success",
                "audio_info": {
                    "sample_rate": sr,
                    "original_duration": len(audio) / sr,
                    "processed_duration": len(trimmed_audio) / sr,
                    "voice_segments": voice_segments
                },
                "quality_metrics": quality_metrics,
                "features": {
                    "mfcc_shape": mfcc_features["mfcc"].shape,
                    "spectrogram_shape": mel_spectrogram.shape
                }
            }
            
            # Save features if output directory specified
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                file_stem = Path(audio_path).stem
                
                # Save MFCC features
                mfcc_path = output_dir / f"{file_stem}_mfcc.npy"
                np.save(mfcc_path, mfcc_features["mfcc"])
                
                # Save mel spectrogram
                mel_path = output_dir / f"{file_stem}_mel.npy"
                np.save(mel_path, mel_spectrogram)
                
                result["output_files"] = {
                    "mfcc": str(mfcc_path),
                    "mel_spectrogram": str(mel_path)
                }
            
            return result
            
        except Exception as e:
            return {
                "file": str(audio_path),
                "status": "error",
                "error": str(e)
            }


def main():
    """CLI interface for audio preprocessor"""
    parser = argparse.ArgumentParser(description="Audio Preprocessor for meditation module")
    parser.add_argument("--input", type=str, required=True,
                       help="Input audio file or directory")
    parser.add_argument("--output-dir", type=str, default="preprocess_output/audio_features/",
                       help="Output directory for processed features")
    parser.add_argument("--metadata-file", type=str, default="preprocess_output/audio_metadata.json",
                       help="Output JSON file for metadata")
    parser.add_argument("--sample-rate", type=int, default=16000,
                       help="Target sample rate")
    parser.add_argument("--n-mfcc", type=int, default=13,
                       help="Number of MFCC coefficients")
    parser.add_argument("--n-mels", type=int, default=128,
                       help="Number of mel frequency bands")
    parser.add_argument("--frame-length", type=int, default=2048,
                       help="Frame length for STFT")
    parser.add_argument("--hop-length", type=int, default=512,
                       help="Hop length for STFT")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=args.sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc
    )
    
    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    results = []
    
    if input_path.is_file():
        # Single audio file
        result = preprocessor.process_audio_file(str(input_path), str(output_dir))
        results.append(result)
        
    elif input_path.is_dir():
        # Directory of audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f"**/*{ext}"))
            audio_files.extend(input_path.glob(f"**/*{ext.upper()}"))
        
        for audio_file in sorted(audio_files):
            result = preprocessor.process_audio_file(str(audio_file), str(output_dir))
            results.append(result)
            
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    # Save metadata
    metadata_path = Path(args.metadata_file)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Audio preprocessing complete. Processed {len(results)} files.")
    print(f"Features saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Summary statistics
    successful = [r for r in results if r.get("status") == "success"]
    if successful:
        total_duration = sum(r["audio_info"]["processed_duration"] for r in successful)
        avg_quality = np.mean([r["quality_metrics"]["quality_score"] for r in successful])
        
        print(f"\nSummary:")
        print(f"Successful files: {len(successful)}/{len(results)}")
        print(f"Total processed duration: {total_duration:.2f}s")
        print(f"Average audio quality: {avg_quality:.3f}")


if __name__ == "__main__":
    main()