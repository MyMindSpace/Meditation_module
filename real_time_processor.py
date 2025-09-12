# real_time_processor.py
"""
Real-time Processor

Handles real-time data processing pipeline:
- Live video/audio stream processing
- Multi-modal feature fusion
- Real-time decision making
- Performance optimization
"""

import asyncio
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np

# Import encoders and processors
from Encoders.vision_encoder import VisionEncoder
from Encoders.audio_encoder import AudioEncoder
from preprocessing_unit.video_preprocessor import VideoPreprocessor
from preprocessing_unit.audio_preprocessor import AudioPreprocessor
from Core_engine.fusion import MultiModalFusion
from Core_engine.decision_manager import DecisionManager


@dataclass
class ProcessingConfig:
    """Configuration for real-time processing"""
    enable_video: bool = True
    enable_audio: bool = True
    video_fps: float = 30.0
    audio_sample_rate: int = 16000
    chunk_duration: float = 1.0  # seconds
    max_queue_size: int = 10
    processing_timeout: float = 0.5  # seconds
    feature_cache_size: int = 100


@dataclass 
class StreamData:
    """Container for streaming data"""
    timestamp: float
    video_frame: Optional[np.ndarray] = None
    audio_chunk: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedOutput:
    """Container for processed output"""
    timestamp: float
    video_features: Optional[Dict[str, Any]] = None
    audio_features: Optional[Dict[str, Any]] = None
    fused_features: Optional[Dict[str, Any]] = None
    decisions: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)


class RealTimeProcessor:
    """
    Real-time multi-modal data processor
    Optimized for low-latency meditation session processing
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.is_processing = False
        
        # Initialize components
        if config.enable_video:
            self.video_preprocessor = VideoPreprocessor(
                target_size=(224, 224),
                quality_threshold=0.3
            )
            self.vision_encoder = VisionEncoder(
                use_torch=True,
                model_name="resnet50"
            )
        else:
            self.video_preprocessor = None
            self.vision_encoder = None
            
        if config.enable_audio:
            self.audio_preprocessor = AudioPreprocessor(
                sample_rate=config.audio_sample_rate,
                n_mfcc=13,
                n_mels=128
            )
            self.audio_encoder = AudioEncoder(
                sample_rate=config.audio_sample_rate
            )
        else:
            self.audio_preprocessor = None
            self.audio_encoder = None
        
        # Fusion and decision components
        self.fusion_engine = MultiModalFusion()
        self.decision_manager = DecisionManager()
        
        # Processing queues
        self.input_queue = queue.Queue(maxsize=config.max_queue_size)
        self.output_queue = queue.Queue(maxsize=config.max_queue_size)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_thread = None
        
        # Feature caching for temporal consistency
        self.feature_cache = {
            'video': [],
            'audio': [],
            'fused': []
        }
        
        # Performance metrics
        self.performance_stats = {
            'frames_processed': 0,
            'audio_chunks_processed': 0,
            'average_processing_time': 0.0,
            'dropped_frames': 0,
            'errors': 0
        }
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'output_ready': [],
            'error': [],
            'performance_update': []
        }
    
    def register_callback(self, event: str, callback: Callable):
        """Register callback for processing events"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event}")
    
    def start_processing(self):
        """Start real-time processing"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        print("Real-time processing started")
    
    def stop_processing(self):
        """Stop real-time processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        self.executor.shutdown(wait=True)
        print("Real-time processing stopped")
    
    def add_stream_data(self, data: StreamData) -> bool:
        """Add streaming data for processing"""
        try:
            self.input_queue.put(data, block=False)
            return True
        except queue.Full:
            self.performance_stats['dropped_frames'] += 1
            return False
    
    def get_processed_output(self, timeout: float = 0.1) -> Optional[ProcessedOutput]:
        """Get processed output"""
        try:
            return self.output_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.is_processing:
            try:
                # Get input data
                data = self.input_queue.get(timeout=0.1)
                
                # Process data
                start_time = time.time()
                output = self._process_stream_data(data)
                processing_time = time.time() - start_time
                
                if output:
                    output.processing_time = processing_time
                    
                    # Add to output queue
                    try:
                        self.output_queue.put(output, block=False)
                        self._trigger_callback('output_ready', output)
                    except queue.Full:
                        pass  # Drop output if queue full
                
                # Update performance stats
                self._update_performance_stats(processing_time)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.performance_stats['errors'] += 1
                self._trigger_callback('error', str(e))
    
    def _process_stream_data(self, data: StreamData) -> Optional[ProcessedOutput]:
        """Process single stream data item"""
        output = ProcessedOutput(timestamp=data.timestamp)
        
        # Process video if available
        if data.video_frame is not None and self.config.enable_video:
            output.video_features = self._process_video_frame(data.video_frame)
            if output.video_features:
                self.performance_stats['frames_processed'] += 1
        
        # Process audio if available
        if data.audio_chunk is not None and self.config.enable_audio:
            output.audio_features = self._process_audio_chunk(data.audio_chunk)
            if output.audio_features:
                self.performance_stats['audio_chunks_processed'] += 1
        
        # Fuse features if both modalities available
        if output.video_features or output.audio_features:
            output.fused_features = self._fuse_features(output.video_features, output.audio_features)
            
            # Make decisions based on fused features
            if output.fused_features:
                output.decisions = self._make_decisions(output.fused_features)
        
        # Calculate quality metrics
        output.quality_metrics = self._calculate_quality_metrics(output)
        
        return output
    
    def _process_video_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process single video frame"""
        try:
            # Preprocess frame
            processed_frames, metadata = self.video_preprocessor.process_frames_array(
                np.expand_dims(frame, axis=0)
            )
            
            if processed_frames.size == 0:
                return None
            
            # Extract features
            processed_frame = processed_frames[0]
            features = self.vision_encoder.process_frame(processed_frame)
            
            # Add to cache
            self._add_to_cache('video', features)
            
            # Add temporal context
            features['temporal_context'] = self._get_temporal_context('video')
            
            return features
            
        except Exception as e:
            return None
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process single audio chunk"""
        try:
            # Extract features using audio encoder
            # Create temporary file-like processing
            temp_path = "/tmp/temp_audio.wav"  # In production, use proper temp file handling
            
            # For now, process directly with encoder methods
            features = {
                'mfcc': self.audio_encoder.extract_mfcc_features(audio_chunk),
                'prosodic': self.audio_encoder.extract_prosodic_features(audio_chunk),
                'vad': self.audio_encoder.detect_voice_activity(audio_chunk),
                'emotion': self.audio_encoder.recognize_emotion(
                    audio_chunk, 
                    self.audio_encoder.extract_mfcc_features(audio_chunk),
                    self.audio_encoder.extract_prosodic_features(audio_chunk)
                )
            }
            
            # Add to cache
            self._add_to_cache('audio', features)
            
            # Add temporal context
            features['temporal_context'] = self._get_temporal_context('audio')
            
            return features
            
        except Exception as e:
            return None
    
    def _fuse_features(self, video_features: Optional[Dict[str, Any]], 
                      audio_features: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Fuse multi-modal features"""
        try:
            # Prepare inputs for fusion
            fusion_inputs = {}
            
            if video_features:
                # Extract pose embedding for fusion
                pose_embedding = video_features.get('pose_embedding', [])
                fusion_inputs['pdm'] = {
                    'embedding': pose_embedding,
                    'confidence': 1.0 if video_features.get('pose_detected', False) else 0.3
                }
                
                # Vision features
                visual_features = video_features.get('visual_features', [])
                fusion_inputs['msm'] = {
                    'embedding': visual_features[:32] if len(visual_features) > 32 else visual_features,
                    'confidence': 0.8
                }
            
            if audio_features:
                # Audio features
                audio_embedding = audio_features.get('mfcc', {}).get('mfcc', [])
                if isinstance(audio_embedding, np.ndarray):
                    audio_embedding = audio_embedding.flatten()[:32].tolist()
                elif isinstance(audio_embedding, list) and len(audio_embedding) > 0:
                    if isinstance(audio_embedding[0], (list, np.ndarray)):
                        audio_embedding = np.array(audio_embedding).flatten()[:32].tolist()
                
                fusion_inputs['arm'] = {
                    'embedding': audio_embedding,
                    'confidence': audio_features.get('vad', [0.5])[0] if audio_features.get('vad') else 0.5
                }
            
            # TTS placeholder
            fusion_inputs['tts'] = {'embedding': [], 'confidence': 0.0}
            
            # Perform fusion
            if fusion_inputs:
                fused = self.fusion_engine.fuse(fusion_inputs)
                
                # Add to cache
                self._add_to_cache('fused', fused)
                
                return fused
            
            return None
            
        except Exception as e:
            return None
    
    def _make_decisions(self, fused_features: Dict[str, Any]) -> Dict[str, Any]:
        """Make decisions based on fused features"""
        try:
            # Simple quality assessment
            quality_report = {
                'overall_ok': True,
                'models': {
                    'msm': {'ok': True, 'confidence': 0.8},
                    'pdm': {'ok': True, 'confidence': fused_features.get('weights', {}).get('pdm', 0.5)},
                    'arm': {'ok': True, 'confidence': fused_features.get('weights', {}).get('arm', 0.5)},
                    'tts': {'ok': True, 'confidence': 0.7}
                }
            }
            
            # Make decision
            decisions = self.decision_manager.decide(fused_features, quality_report)
            
            return decisions
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_quality_metrics(self, output: ProcessedOutput) -> Dict[str, float]:
        """Calculate quality metrics for output"""
        metrics = {}
        
        # Video quality
        if output.video_features:
            pose_detected = output.video_features.get('pose_detected', False)
            metrics['video_quality'] = 1.0 if pose_detected else 0.3
        
        # Audio quality
        if output.audio_features:
            vad_ratio = output.audio_features.get('vad', [0.5])[0] if output.audio_features.get('vad') else 0.5
            metrics['audio_quality'] = float(vad_ratio)
        
        # Fusion quality
        if output.fused_features:
            metrics['fusion_confidence'] = output.fused_features.get('combined_confidence', 0.5)
        
        # Overall quality
        if metrics:
            metrics['overall_quality'] = np.mean(list(metrics.values()))
        else:
            metrics['overall_quality'] = 0.5
        
        return metrics
    
    def _add_to_cache(self, modality: str, features: Dict[str, Any]):
        """Add features to temporal cache"""
        if modality in self.feature_cache:
            self.feature_cache[modality].append({
                'timestamp': time.time(),
                'features': features
            })
            
            # Limit cache size
            if len(self.feature_cache[modality]) > self.config.feature_cache_size:
                self.feature_cache[modality].pop(0)
    
    def _get_temporal_context(self, modality: str, window_size: int = 5) -> List[Dict[str, Any]]:
        """Get temporal context from cache"""
        if modality not in self.feature_cache:
            return []
        
        cache = self.feature_cache[modality]
        return cache[-window_size:] if len(cache) >= window_size else cache
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        # Update average processing time with exponential moving average
        alpha = 0.1
        if self.performance_stats['average_processing_time'] == 0.0:
            self.performance_stats['average_processing_time'] = processing_time
        else:
            self.performance_stats['average_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.performance_stats['average_processing_time']
            )
        
        # Trigger performance callback periodically
        total_processed = (self.performance_stats['frames_processed'] + 
                          self.performance_stats['audio_chunks_processed'])
        
        if total_processed > 0 and total_processed % 100 == 0:
            self._trigger_callback('performance_update', self.performance_stats.copy())
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """Trigger callbacks for event"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Callback error for {event}: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'frames_processed': 0,
            'audio_chunks_processed': 0,
            'average_processing_time': 0.0,
            'dropped_frames': 0,
            'errors': 0
        }


# Example usage and testing
class SimulatedDataSource:
    """Simulates real-time data for testing"""
    
    def __init__(self, fps: float = 30.0, audio_sample_rate: int = 16000):
        self.fps = fps
        self.audio_sample_rate = audio_sample_rate
        self.frame_interval = 1.0 / fps
        self.audio_chunk_size = int(audio_sample_rate * 0.5)  # 0.5 second chunks
        
    def generate_video_frame(self) -> np.ndarray:
        """Generate simulated video frame"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def generate_audio_chunk(self) -> np.ndarray:
        """Generate simulated audio chunk"""
        return np.random.randn(self.audio_chunk_size).astype(np.float32) * 0.1
    
    def stream_data(self, duration: float, processor: RealTimeProcessor):
        """Stream simulated data to processor"""
        end_time = time.time() + duration
        last_frame_time = 0
        
        while time.time() < end_time and processor.is_processing:
            current_time = time.time()
            
            # Send video frame
            if current_time - last_frame_time >= self.frame_interval:
                data = StreamData(
                    timestamp=current_time,
                    video_frame=self.generate_video_frame(),
                    audio_chunk=self.generate_audio_chunk()
                )
                
                success = processor.add_stream_data(data)
                if success:
                    last_frame_time = current_time
            
            time.sleep(0.01)  # Small sleep to prevent CPU overload


def run_real_time_demo():
    """Run real-time processing demonstration"""
    # Create configuration
    config = ProcessingConfig(
        enable_video=True,
        enable_audio=True,
        video_fps=10.0,  # Lower FPS for demo
        max_queue_size=5
    )
    
    # Create processor
    processor = RealTimeProcessor(config)
    
    # Register callbacks
    def on_output_ready(output: ProcessedOutput):
        print(f"Processed frame at {output.timestamp:.2f}s, "
              f"processing time: {output.processing_time*1000:.1f}ms, "
              f"quality: {output.quality_metrics.get('overall_quality', 0):.2f}")
    
    def on_performance_update(stats: Dict[str, Any]):
        print(f"Performance: {stats['frames_processed']} frames, "
              f"avg time: {stats['average_processing_time']*1000:.1f}ms, "
              f"dropped: {stats['dropped_frames']}")
    
    processor.register_callback('output_ready', on_output_ready)
    processor.register_callback('performance_update', on_performance_update)
    
    # Start processing
    processor.start_processing()
    
    try:
        # Create data source and stream
        data_source = SimulatedDataSource(fps=config.video_fps)
        
        print("Starting real-time processing demo...")
        data_source.stream_data(duration=10.0, processor=processor)
        
        # Process remaining items in queue
        time.sleep(2.0)
        
        # Print final stats
        final_stats = processor.get_performance_stats()
        print(f"\nFinal statistics:")
        print(f"Frames processed: {final_stats['frames_processed']}")
        print(f"Audio chunks processed: {final_stats['audio_chunks_processed']}")
        print(f"Average processing time: {final_stats['average_processing_time']*1000:.1f}ms")
        print(f"Dropped frames: {final_stats['dropped_frames']}")
        print(f"Errors: {final_stats['errors']}")
        
    finally:
        processor.stop_processing()


if __name__ == "__main__":
    run_real_time_demo()