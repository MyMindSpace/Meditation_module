# session_manager.py
"""
Session Manager

Handles real-time meditation session orchestration:
- Session lifecycle management
- Real-time data processing pipeline
- Live posture corrections
- Dynamic adaptation based on user state
"""

import asyncio
import json
import logging
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np

# Import our modules
from Core_engine.fusion import MultiModalFusion
from Core_engine.quality_monitor import QualityMonitor
from Core_engine.decision_manager import DecisionManager
from Core_engine.Posture_detector import PostureDetector


class SessionState(Enum):
    """Session states"""
    IDLE = "idle"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDING = "ending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SessionConfig:
    """Session configuration"""
    session_id: str
    user_id: str
    meditation_type: str
    planned_duration: float  # in seconds
    enable_posture_detection: bool = True
    enable_audio_feedback: bool = True
    posture_correction_interval: float = 30.0  # seconds
    quality_check_interval: float = 10.0  # seconds
    adaptation_sensitivity: float = 0.7  # 0-1 scale
    save_session_data: bool = True


@dataclass
class RealTimeData:
    """Real-time data container"""
    timestamp: float
    video_frame: Optional[np.ndarray] = None
    audio_chunk: Optional[np.ndarray] = None
    user_input: Optional[str] = None
    posture_score: Optional[float] = None
    engagement_level: Optional[float] = None
    quality_metrics: Optional[Dict[str, float]] = None


@dataclass
class SessionMetrics:
    """Session performance metrics"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    planned_duration: float = 0.0
    completion_rate: float = 0.0
    average_posture_score: float = 0.0
    posture_corrections_given: int = 0
    engagement_score: float = 0.0
    user_rating: Optional[float] = None
    meditation_type: str = ""
    interruptions: int = 0
    quality_issues: List[str] = None


class SessionManager:
    """
    Real-time meditation session manager
    Orchestrates the entire meditation experience
    """
    
    def __init__(self, 
                 enable_logging: bool = True,
                 data_buffer_size: int = 100,
                 max_concurrent_sessions: int = 5):
        
        self.enable_logging = enable_logging
        self.data_buffer_size = data_buffer_size
        self.max_concurrent_sessions = max_concurrent_sessions
        
        # Initialize components
        self.fusion_engine = MultiModalFusion()
        self.quality_monitor = QualityMonitor(threshold=0.6)
        self.decision_manager = DecisionManager()
        self.posture_detector = PostureDetector(use_lstm=True)
        
        # Session tracking
        self.active_sessions: Dict[str, 'LiveSession'] = {}
        self.session_history: List[SessionMetrics] = []
        
        # Threading
        self.processing_thread = None
        self.is_processing = False
        self._lock = threading.RLock()
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'session_started': [],
            'posture_correction': [],
            'session_completed': [],
            'quality_alert': [],
            'adaptation_triggered': []
        }
        
        # Setup logging
        if self.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
    
    def register_callback(self, event: str, callback: Callable):
        """Register callback for session events"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event}")
    
    def _log(self, message: str, level: str = "info"):
        """Log message if logging enabled"""
        if self.logger:
            getattr(self.logger, level)(message)
    
    def create_session(self, config: SessionConfig) -> str:
        """Create new meditation session"""
        with self._lock:
            if len(self.active_sessions) >= self.max_concurrent_sessions:
                raise RuntimeError("Maximum concurrent sessions reached")
            
            if config.session_id in self.active_sessions:
                raise ValueError(f"Session {config.session_id} already exists")
            
            session = LiveSession(config, self)
            self.active_sessions[config.session_id] = session
            
            self._log(f"Created session {config.session_id} for user {config.user_id}")
            return config.session_id
    
    def start_session(self, session_id: str) -> bool:
        """Start meditation session"""
        with self._lock:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            success = session.start()
            
            if success:
                self._trigger_callback('session_started', session_id)
                
                # Start processing thread if not running
                if not self.is_processing:
                    self.start_processing()
            
            return success
    
    def pause_session(self, session_id: str) -> bool:
        """Pause meditation session"""
        with self._lock:
            if session_id not in self.active_sessions:
                return False
            
            return self.active_sessions[session_id].pause()
    
    def resume_session(self, session_id: str) -> bool:
        """Resume paused session"""
        with self._lock:
            if session_id not in self.active_sessions:
                return False
            
            return self.active_sessions[session_id].resume()
    
    def end_session(self, session_id: str, user_rating: Optional[float] = None) -> Optional[SessionMetrics]:
        """End meditation session"""
        with self._lock:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            metrics = session.end(user_rating)
            
            # Move to history
            if metrics:
                self.session_history.append(metrics)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            self._trigger_callback('session_completed', session_id, metrics)
            
            # Stop processing if no active sessions
            if not self.active_sessions and self.is_processing:
                self.stop_processing()
            
            return metrics
    
    def process_real_time_data(self, session_id: str, data: RealTimeData) -> bool:
        """Process real-time data for session"""
        with self._lock:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            return session.process_data(data)
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session status"""
        with self._lock:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            return session.get_status()
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        with self._lock:
            return list(self.active_sessions.keys())
    
    def start_processing(self):
        """Start background processing thread"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        self._log("Started background processing")
    
    def stop_processing(self):
        """Stop background processing thread"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        self._log("Stopped background processing")
    
    def _processing_loop(self):
        """Background processing loop"""
        while self.is_processing:
            try:
                with self._lock:
                    active_session_list = list(self.active_sessions.values())
                
                # Process each active session
                for session in active_session_list:
                    if session.state == SessionState.ACTIVE:
                        session.background_processing()
                
                # Sleep briefly to avoid excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self._log(f"Error in processing loop: {e}", "error")
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """Trigger callbacks for event"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self._log(f"Callback error for {event}: {e}", "error")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_processing()
        
        # End all active sessions
        with self._lock:
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                self.end_session(session_id)


class LiveSession:
    """
    Individual live meditation session
    """
    
    def __init__(self, config: SessionConfig, manager: SessionManager):
        self.config = config
        self.manager = manager
        self.state = SessionState.IDLE
        
        # Timing
        self.start_time: Optional[datetime] = None
        self.pause_time: Optional[datetime] = None
        self.total_paused_duration: float = 0.0
        
        # Data buffers
        self.data_buffer: deque = deque(maxlen=manager.data_buffer_size)
        self.posture_scores: deque = deque(maxlen=50)
        self.quality_history: deque = deque(maxlen=20)
        
        # Session metrics
        self.posture_corrections_given = 0
        self.interruptions = 0
        self.last_posture_correction = 0.0
        self.last_quality_check = 0.0
        
        # Adaptation state
        self.current_meditation_type = config.meditation_type
        self.adaptation_history: List[Dict[str, Any]] = []
    
    def start(self) -> bool:
        """Start the session"""
        if self.state != SessionState.IDLE:
            return False
        
        self.state = SessionState.STARTING
        self.start_time = datetime.now()
        self.state = SessionState.ACTIVE
        
        self.manager._log(f"Started session {self.config.session_id}")
        return True
    
    def pause(self) -> bool:
        """Pause the session"""
        if self.state != SessionState.ACTIVE:
            return False
        
        self.state = SessionState.PAUSED
        self.pause_time = datetime.now()
        self.interruptions += 1
        
        return True
    
    def resume(self) -> bool:
        """Resume the session"""
        if self.state != SessionState.PAUSED:
            return False
        
        if self.pause_time:
            pause_duration = (datetime.now() - self.pause_time).total_seconds()
            self.total_paused_duration += pause_duration
            self.pause_time = None
        
        self.state = SessionState.ACTIVE
        return True
    
    def end(self, user_rating: Optional[float] = None) -> SessionMetrics:
        """End the session"""
        self.state = SessionState.ENDING
        
        end_time = datetime.now()
        if self.start_time:
            total_duration = (end_time - self.start_time).total_seconds()
            active_duration = total_duration - self.total_paused_duration
        else:
            total_duration = active_duration = 0.0
        
        # Calculate completion rate
        completion_rate = min(1.0, active_duration / max(self.config.planned_duration, 1.0))
        
        # Calculate average posture score
        avg_posture = float(np.mean(self.posture_scores)) if self.posture_scores else 0.0
        
        # Calculate engagement score (placeholder)
        engagement_score = self._calculate_engagement_score()
        
        # Create metrics
        metrics = SessionMetrics(
            session_id=self.config.session_id,
            start_time=self.start_time or datetime.now(),
            end_time=end_time,
            duration=active_duration,
            planned_duration=self.config.planned_duration,
            completion_rate=completion_rate,
            average_posture_score=avg_posture,
            posture_corrections_given=self.posture_corrections_given,
            engagement_score=engagement_score,
            user_rating=user_rating,
            meditation_type=self.current_meditation_type,
            interruptions=self.interruptions,
            quality_issues=self._get_quality_issues()
        )
        
        self.state = SessionState.COMPLETED
        
        # Save session data if enabled
        if self.config.save_session_data:
            self._save_session_data(metrics)
        
        return metrics
    
    def process_data(self, data: RealTimeData) -> bool:
        """Process real-time data"""
        if self.state != SessionState.ACTIVE:
            return False
        
        # Add to buffer
        self.data_buffer.append(data)
        
        # Process posture if available
        if data.video_frame is not None and self.config.enable_posture_detection:
            self._process_posture_data(data)
        
        # Process audio if available
        if data.audio_chunk is not None:
            self._process_audio_data(data)
        
        # Update engagement level
        if data.engagement_level is not None:
            self._update_engagement(data.engagement_level)
        
        return True
    
    def background_processing(self):
        """Background processing for active session"""
        current_time = time.time()
        
        # Check if posture correction is needed
        if (self.config.enable_posture_detection and 
            current_time - self.last_posture_correction >= self.config.posture_correction_interval):
            self._check_posture_correction()
            self.last_posture_correction = current_time
        
        # Quality monitoring
        if current_time - self.last_quality_check >= self.config.quality_check_interval:
            self._check_quality()
            self.last_quality_check = current_time
        
        # Adaptation check
        self._check_adaptation_needs()
    
    # In session_manager.py - Replace the _process_posture_data method:

    # In session_manager.py - around line 400-430
    def _process_posture_data(self, data: RealTimeData):
        """Process posture detection data - CORRECTED VERSION"""
        try:
            if data.video_frame is None:
                return
                
            # Add vision encoder if not already present
            if not hasattr(self, 'vision_encoder'):
                from Encoders.vision_encoder import VisionEncoder
                self.vision_encoder = VisionEncoder(use_torch=False)
                
            # Process frame through vision encoder first
            vision_result = self.vision_encoder.process_frame(data.video_frame)
            
            if vision_result['pose_detected']:
                # Now use proper pose embeddings
                record = {
                    "embeddings": {"pose": vision_result['pose_embedding']}
                }
                
                frame_result = self.manager.posture_detector.process_record(record)
                posture_score = frame_result.get("posture_score", 0.5)
                
                self.posture_scores.append(posture_score)
                data.posture_score = posture_score
            else:
                # Handle case when pose is not detected
                print("nope")
                
        except Exception as e:
            self.manager._log(f"Posture processing error: {e}", "error")
            print("nope")


    def _process_audio_data(self, data: RealTimeData):
        """Process audio data"""
        # Placeholder for audio processing
        # Could include voice activity detection, emotion recognition, etc.
        pass
    
    def _update_engagement(self, engagement_level: float):
        """Update engagement tracking"""
        # Placeholder for engagement tracking
        pass
    
    def _check_posture_correction(self):
        """Check if posture correction is needed"""
        if len(self.posture_scores) < 3:
            return
        
        recent_scores = list(self.posture_scores)[-3:]
        avg_recent_score = np.mean(recent_scores)
        
        if avg_recent_score < 0.6:  # Threshold for correction
            correction = self._generate_posture_correction(avg_recent_score)
            if correction:
                self.posture_corrections_given += 1
                self.manager._trigger_callback('posture_correction', self.config.session_id, correction)
    
    def _generate_posture_correction(self, posture_score: float) -> Optional[str]:
        """Generate posture correction message"""
        if posture_score < 0.3:
            return "Please straighten your back and relax your shoulders"
        elif posture_score < 0.6:
            return "Gently adjust your posture - sit tall and breathe deeply"
        else:
            return None
    
    def _check_quality(self):
        """Check session quality"""
        # Collect recent data for quality assessment
        recent_data = list(self.data_buffer)[-5:] if len(self.data_buffer) >= 5 else list(self.data_buffer)
        
        quality_issues = []
        
        # Check posture quality
        if self.posture_scores:
            avg_posture = np.mean(list(self.posture_scores)[-10:])
            if avg_posture < 0.4:
                quality_issues.append("Poor posture detected")
        
        # Check for interruptions
        if self.interruptions > 2:
            quality_issues.append("Multiple interruptions")
        
        if quality_issues:
            self.manager._trigger_callback('quality_alert', self.config.session_id, quality_issues)
    
    def _check_adaptation_needs(self):
        """Check if session adaptation is needed"""
        if not self.posture_scores or len(self.data_buffer) < 10:
            return
        
        # Simple adaptation logic
        avg_posture = np.mean(list(self.posture_scores)[-10:])
        adaptation_needed = False
        adaptation_reason = ""
        
        if avg_posture < 0.3:
            adaptation_needed = True
            adaptation_reason = "Persistent posture issues"
        elif self.interruptions > 3:
            adaptation_needed = True
            adaptation_reason = "Frequent interruptions"
        
        if adaptation_needed:
            self._adapt_session(adaptation_reason)
    
    def _adapt_session(self, reason: str):
        """Adapt session based on current state"""
        adaptation = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "original_type": self.current_meditation_type,
            "action": "No adaptation implemented"  # Placeholder
        }
        
        self.adaptation_history.append(adaptation)
        self.manager._trigger_callback('adaptation_triggered', self.config.session_id, adaptation)
    
    def _calculate_engagement_score(self) -> float:
        """Calculate overall engagement score"""
        # Placeholder implementation
        base_score = 0.7
        
        # Adjust based on completion rate
        if hasattr(self, 'completion_rate'):
            base_score *= (0.5 + 0.5 * getattr(self, 'completion_rate', 0.7))
        
        # Adjust based on posture
        if self.posture_scores:
            avg_posture = np.mean(self.posture_scores)
            base_score *= (0.7 + 0.3 * avg_posture)
        
        return min(1.0, base_score)
    
    def _get_quality_issues(self) -> List[str]:
        """Get list of quality issues encountered"""
        issues = []
        
        if self.posture_corrections_given > 5:
            issues.append("Frequent posture corrections needed")
        
        if self.interruptions > 2:
            issues.append("Multiple session interruptions")
        
        return issues
    
    def _save_session_data(self, metrics: SessionMetrics):
        """Save session data to file"""
        try:
            output_dir = Path("session_data")
            output_dir.mkdir(exist_ok=True)
            
            # Save metrics
            metrics_file = output_dir / f"{self.config.session_id}_metrics.json"
            with metrics_file.open("w") as f:
                json.dump(asdict(metrics), f, indent=2, default=str)
            
            # Save detailed data
            session_data = {
                "config": asdict(self.config),
                "posture_scores": list(self.posture_scores),
                "adaptation_history": self.adaptation_history,
                "data_points": len(self.data_buffer)
            }
            
            data_file = output_dir / f"{self.config.session_id}_data.json"
            with data_file.open("w") as f:
                json.dump(session_data, f, indent=2, default=str)
                
        except Exception as e:
            self.manager._log(f"Error saving session data: {e}", "error")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current session status"""
        current_time = datetime.now()
        
        if self.start_time:
            elapsed = (current_time - self.start_time).total_seconds() - self.total_paused_duration
            progress = min(1.0, elapsed / max(self.config.planned_duration, 1.0))
        else:
            elapsed = 0.0
            progress = 0.0
        
        return {
            "session_id": self.config.session_id,
            "state": self.state.value,
            "progress": progress,
            "elapsed_time": elapsed,
            "posture_score": float(np.mean(list(self.posture_scores)[-3:])) if len(self.posture_scores) >= 3 else None,
            "corrections_given": self.posture_corrections_given,
            "interruptions": self.interruptions,
            "meditation_type": self.current_meditation_type
        }


# Example usage functions
def create_example_session_manager():
    """Create example session manager with callbacks"""
    manager = SessionManager(enable_logging=True)
    
    def on_session_started(session_id):
        print(f"Session {session_id} started!")
    
    def on_posture_correction(session_id, correction):
        print(f"Posture correction for {session_id}: {correction}")
    
    def on_session_completed(session_id, metrics):
        print(f"Session {session_id} completed with {metrics.completion_rate:.1%} completion")
    
    manager.register_callback('session_started', on_session_started)
    manager.register_callback('posture_correction', on_posture_correction)
    manager.register_callback('session_completed', on_session_completed)
    
    return manager


def run_example_session():
    """Run an example meditation session"""
    manager = create_example_session_manager()
    
    try:
        # Create session config
        config = SessionConfig(
            session_id="example_001",
            user_id="user_123",
            meditation_type="mindfulness",
            planned_duration=300.0,  # 5 minutes
            enable_posture_detection=True,
            posture_correction_interval=15.0
        )
        
        # Create and start session
        session_id = manager.create_session(config)
        success = manager.start_session(session_id)
        
        if success:
            print("Session started successfully!")
            
            # Simulate some real-time data
            for i in range(10):
                data = RealTimeData(
                    timestamp=time.time(),
                    video_frame=np.random.random((224, 224, 3)),
                    posture_score=0.7 + 0.3 * np.random.random() - 0.15,
                    engagement_level=0.8
                )
                
                manager.process_real_time_data(session_id, data)
                time.sleep(1)
            
            # End session
            metrics = manager.end_session(session_id, user_rating=4.5)
            if metrics:
                print(f"Session completed: {metrics.completion_rate:.1%}")
        
    finally:
        manager.cleanup()


if __name__ == "__main__":
    run_example_session()