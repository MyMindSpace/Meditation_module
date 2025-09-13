
import numpy as np
import cv2
import threading
import time
import queue
from typing import Dict, List, Any, Optional, Callable
import logging
from dataclasses import dataclass
from collections import deque

@dataclass
class ProcessingConfig:
    """Configuration for real-time processing"""
    enable_video: bool = True
    enable_audio: bool = False
    target_fps: int = 30
    frame_width: int = 640
    frame_height: int = 480
    pose_detection_confidence: float = 0.7
    pose_tracking_confidence: float = 0.7
    enable_visual_feedback: bool = True
    enable_posture_analysis: bool = True
    buffer_size: int = 100
    processing_threads: int = 2

class RealTimeProcessor:
    """Enhanced real-time processor with advanced posture analysis"""

    def __init__(self, config: ProcessingConfig, enhanced_posture_detector=None, vision_encoder=None):
        self.config = config
        self.enhanced_posture_detector = enhanced_posture_detector
        self.vision_encoder = vision_encoder

        # Processing state
        self.is_processing = False
        self.is_paused = False

        # Threading components
        self.processing_thread = None
        self.frame_queue = queue.Queue(maxsize=config.buffer_size)
        self.result_queue = queue.Queue(maxsize=config.buffer_size)
        self.processing_lock = threading.Lock()

        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=100)
        self.frame_drop_count = 0
        self.total_frames_processed = 0

        # Callbacks
        self.callbacks = {}

        # Logging
        self.logger = logging.getLogger(__name__)

        # Enhanced processing components
        self.frame_buffer = deque(maxlen=10)  # Recent frames buffer
        self.posture_history = deque(maxlen=50)  # Recent posture data
        self.performance_metrics = {
            'average_fps': 0.0,
            'average_processing_time': 0.0,
            'frame_drop_rate': 0.0,
            'posture_detection_rate': 0.0
        }

    def start(self, video_source=0):
        """Start real-time processing"""
        try:
            if self.is_processing:
                self.logger.warning("Processor already running")
                return False

            # Initialize video capture
            self.video_capture = cv2.VideoCapture(video_source)
            if not self.video_capture.isOpened():
                self.logger.error(f"Failed to open video source: {video_source}")
                return False

            # Configure video capture
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self.video_capture.set(cv2.CAP_PROP_FPS, self.config.target_fps)

            # Start processing
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()

            self.logger.info("Real-time processor started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start processor: {e}")
            return False

    def stop(self):
        """Stop real-time processing"""
        try:
            if not self.is_processing:
                return

            self.is_processing = False

            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)

            # Release video capture
            if hasattr(self, 'video_capture') and self.video_capture.isOpened():
                self.video_capture.release()

            # Clear queues
            self._clear_queues()

            self.logger.info("Real-time processor stopped")

        except Exception as e:
            self.logger.error(f"Error stopping processor: {e}")

    def pause(self):
        """Pause processing"""
        self.is_paused = True
        self.logger.info("Processing paused")

    def resume(self):
        """Resume processing"""
        self.is_paused = False
        self.logger.info("Processing resumed")

    def _processing_loop(self):
        """Main processing loop"""
        frame_time_target = 1.0 / self.config.target_fps
        last_frame_time = time.time()

        while self.is_processing:
            try:
                if self.is_paused:
                    time.sleep(0.1)
                    continue

                current_time = time.time()

                # Control frame rate
                elapsed = current_time - last_frame_time
                if elapsed < frame_time_target:
                    time.sleep(frame_time_target - elapsed)

                # Capture frame
                ret, frame = self.video_capture.read()
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    continue

                # Process frame
                processing_start = time.time()
                processed_result = self._process_frame_comprehensive(frame)
                processing_end = time.time()

                # Update performance metrics
                self._update_performance_metrics(processing_end - processing_start)

                # Store result
                if processed_result:
                    try:
                        self.result_queue.put_nowait(processed_result)
                    except queue.Full:
                        # Remove oldest result and add new one
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(processed_result)
                            self.frame_drop_count += 1
                        except queue.Empty:
                            pass

                # Trigger real-time callbacks
                if processed_result and 'frame_processed' in self.callbacks:
                    self.callbacks['frame_processed'](processed_result)

                last_frame_time = current_time
                self.total_frames_processed += 1

            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                time.sleep(0.1)  # Prevent tight error loop

    def _process_frame_comprehensive(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Comprehensive frame processing with enhanced features"""
        try:
            timestamp = time.time()

            # Store frame in buffer
            self.frame_buffer.append((frame.copy(), timestamp))

            # Initialize result
            result = {
                'timestamp': timestamp,
                'original_frame': frame,
                'processed_frame': frame.copy(),
                'pose_detected': False,
                'posture_analysis': {},
                'visual_feedback_frame': None,
                'processing_time': 0.0
            }

            # Process video frame
            if self.config.enable_video:
                video_result = self._process_video_frame(frame)
                result.update(video_result)

            # Enhanced posture analysis
            if self.config.enable_posture_analysis and self.enhanced_posture_detector:
                posture_result = self._process_enhanced_posture(frame)
                result['posture_analysis'] = posture_result

                # Store in posture history
                if posture_result:
                    self.posture_history.append((posture_result, timestamp))

            # Generate visual feedback
            if self.config.enable_visual_feedback and result.get('posture_analysis'):
                feedback_frame = self._apply_visual_feedback(frame, result['posture_analysis'])
                result['visual_feedback_frame'] = feedback_frame

            return result

        except Exception as e:
            self.logger.error(f"Comprehensive frame processing error: {e}")
            return None

    def _process_video_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Enhanced video frame processing"""
        try:
            if frame is not None and self.config.enable_video:
                # Use vision encoder if available
                if self.vision_encoder:
                    vision_result = self.vision_encoder.process_frame(frame)

                    return {
                        'visual_features': vision_result.get('pose_features', {}),
                        'pose_embedding': vision_result.get('pose_embedding', []),
                        'landmarks': vision_result.get('landmarks'),
                        'pose_detected': vision_result.get('landmarks') is not None,
                        'segmentation_mask': vision_result.get('segmentation_mask'),
                        'processed_frame': vision_result.get('processed_frame', frame)
                    }

                else:
                    # Basic processing without vision encoder
                    return {
                        'visual_features': {},
                        'pose_embedding': [],
                        'landmarks': None,
                        'pose_detected': False,
                        'processed_frame': frame
                    }

            return {}

        except Exception as e:
            self.logger.error(f"Video frame processing error: {e}")
            return {}

    def _process_enhanced_posture(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process frame with enhanced posture detection"""
        try:
            if not self.enhanced_posture_detector:
                return None

            # Get detailed posture analysis
            posture_result = self.enhanced_posture_detector.detect_meditation_posture(frame)

            if posture_result and 'error' not in posture_result:
                return {
                    'posture_score': posture_result.get('posture_score', 0.0),
                    'detailed_metrics': posture_result.get('detailed_metrics', {}),
                    'body_part_scores': posture_result.get('body_part_scores', {}),
                    'recommendations': posture_result.get('recommendations', []),
                    'stability_score': posture_result.get('stability_score', 0.0),
                    'landmarks': posture_result.get('landmarks'),
                    'timestamp': posture_result.get('timestamp', time.time())
                }

            return None

        except Exception as e:
            self.logger.error(f"Enhanced posture processing error: {e}")
            return None

    def _apply_visual_feedback(self, frame: np.ndarray, posture_data: Dict[str, Any]) -> np.ndarray:
        """Apply visual feedback to frame"""
        try:
            if not self.vision_encoder or not posture_data:
                return frame

            # Use vision encoder's feedback rendering
            feedback_frame = self.vision_encoder.render_posture_feedback(frame, posture_data)
            return feedback_frame

        except Exception as e:
            self.logger.error(f"Visual feedback application error: {e}")
            return frame

    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics"""
        try:
            current_time = time.time()

            # Update FPS counter
            self.fps_counter.append(current_time)

            # Update processing times
            self.processing_times.append(processing_time)

            # Calculate metrics
            if len(self.fps_counter) > 1:
                time_span = self.fps_counter[-1] - self.fps_counter[0]
                if time_span > 0:
                    self.performance_metrics['average_fps'] = (len(self.fps_counter) - 1) / time_span

            if self.processing_times:
                self.performance_metrics['average_processing_time'] = sum(self.processing_times) / len(self.processing_times)

            if self.total_frames_processed > 0:
                self.performance_metrics['frame_drop_rate'] = self.frame_drop_count / self.total_frames_processed

            # Calculate posture detection rate
            if self.posture_history:
                successful_detections = sum(1 for result, _ in self.posture_history if result is not None)
                self.performance_metrics['posture_detection_rate'] = successful_detections / len(self.posture_history)

        except Exception as e:
            self.logger.error(f"Performance metrics update error: {e}")

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Get latest processing result"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all available results"""
        results = []
        try:
            while True:
                result = self.result_queue.get_nowait()
                results.append(result)
        except queue.Empty:
            pass
        return results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.performance_metrics,
            'total_frames_processed': self.total_frames_processed,
            'frame_drop_count': self.frame_drop_count,
            'queue_size': self.result_queue.qsize(),
            'is_processing': self.is_processing,
            'is_paused': self.is_paused
        }

    def get_recent_posture_data(self, seconds: float = 10.0) -> List[Dict[str, Any]]:
        """Get recent posture data within specified time window"""
        try:
            current_time = time.time()
            cutoff_time = current_time - seconds

            recent_data = []
            for posture_result, timestamp in self.posture_history:
                if timestamp >= cutoff_time and posture_result:
                    recent_data.append({**posture_result, 'timestamp': timestamp})

            return recent_data

        except Exception as e:
            self.logger.error(f"Recent posture data retrieval error: {e}")
            return []

    def get_frame_buffer(self) -> List[Tuple[np.ndarray, float]]:
        """Get recent frame buffer"""
        return list(self.frame_buffer)

    def register_callback(self, event_name: str, callback: Callable):
        """Register callback for processing events"""
        self.callbacks[event_name] = callback

    def unregister_callback(self, event_name: str):
        """Unregister callback"""
        if event_name in self.callbacks:
            del self.callbacks[event_name]

    def _clear_queues(self):
        """Clear all queues"""
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
        except queue.Empty:
            pass

    def update_config(self, new_config: ProcessingConfig):
        """Update processing configuration"""
        try:
            was_processing = self.is_processing

            if was_processing:
                self.pause()

            self.config = new_config

            # Update video capture settings if active
            if hasattr(self, 'video_capture') and self.video_capture.isOpened():
                self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                self.video_capture.set(cv2.CAP_PROP_FPS, self.config.target_fps)

            if was_processing:
                self.resume()

            self.logger.info("Configuration updated")

        except Exception as e:
            self.logger.error(f"Configuration update error: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'processing_status': {
                'is_processing': self.is_processing,
                'is_paused': self.is_paused,
                'total_frames': self.total_frames_processed,
                'dropped_frames': self.frame_drop_count
            },
            'performance_metrics': self.get_performance_metrics(),
            'configuration': {
                'target_fps': self.config.target_fps,
                'frame_size': f"{self.config.frame_width}x{self.config.frame_height}",
                'video_enabled': self.config.enable_video,
                'posture_analysis_enabled': self.config.enable_posture_analysis,
                'visual_feedback_enabled': self.config.enable_visual_feedback
            },
            'components': {
                'enhanced_posture_detector': self.enhanced_posture_detector is not None,
                'vision_encoder': self.vision_encoder is not None,
                'video_capture_active': hasattr(self, 'video_capture') and self.video_capture.isOpened()
            }
        }

    def save_frame(self, frame: np.ndarray, filename: str):
        """Save a frame to file"""
        try:
            cv2.imwrite(filename, frame)
            self.logger.info(f"Frame saved to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Frame save error: {e}")
            return False

    def __del__(self):
        """Destructor"""
        try:
            self.stop()
        except:
            pass
