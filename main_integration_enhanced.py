
import time
import logging
from typing import Dict, Any, Optional, Callable
import numpy as np
import threading
from dataclasses import dataclass

# Import enhanced components
try:
    from posture_detector_enhanced import PostureDetector, EnhancedPostureDetector
    from vision_encoder_enhanced import VisionEncoder
    from session_manager_enhanced import SessionManager, RealTimeData
    from real_time_processor_enhanced import RealTimeProcessor, ProcessingConfig
    from progress_tracker_enhanced import ProgressTracker
    from posture_feedback_renderer import PostureFeedbackRenderer
except ImportError as e:
    print(f"Import warning: {e}")
    print("Some enhanced components may not be available")

@dataclass
class MeditationConfig:
    """Configuration for meditation module"""
    enable_posture_detection: bool = True
    enable_visual_feedback: bool = True
    enable_progress_tracking: bool = True
    target_fps: int = 30
    frame_width: int = 640
    frame_height: int = 480
    session_timeout_seconds: int = 3600  # 1 hour max
    auto_save_interval_seconds: int = 300  # 5 minutes

class MeditationModuleIntegration:
    """Enhanced meditation module integration with live posture detection"""

    def __init__(self, user_id: str, config: Optional[MeditationConfig] = None):
        self.user_id = user_id
        self.config = config or MeditationConfig()
        self.logger = self._setup_logging()

        # Initialize enhanced components
        self.enhanced_posture_detector = EnhancedPostureDetector(meditation_specific=True)
        self.vision_encoder = VisionEncoder()
        self.posture_feedback_renderer = PostureFeedbackRenderer()

        # Initialize managers
        self.session_manager = SessionManager(enhanced_posture_detector=self.enhanced_posture_detector)
        self.progress_tracker = ProgressTracker()

        # Initialize real-time processor
        processing_config = ProcessingConfig(
            enable_video=self.config.enable_posture_detection,
            target_fps=self.config.target_fps,
            frame_width=self.config.frame_width,
            frame_height=self.config.frame_height,
            enable_visual_feedback=self.config.enable_visual_feedback,
            enable_posture_analysis=True
        )

        self.real_time_processor = RealTimeProcessor(
            config=processing_config,
            enhanced_posture_detector=self.enhanced_posture_detector,
            vision_encoder=self.vision_encoder
        )

        # State management
        self.current_session_id = None
        self.is_active = False
        self.auto_save_thread = None
        self._stop_auto_save = threading.Event()

        # Setup callbacks
        self._setup_callbacks()

        self.logger.info(f"Meditation module initialized for user: {user_id}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the module"""
        logger = logging.getLogger(f"meditation_module_{self.user_id}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_callbacks(self):
        """Setup callbacks for real-time feedback and session management"""

        # Session manager callbacks
        def on_session_end(session_id: str, session_stats: Dict[str, Any], session_summary: Dict[str, Any]):
            self.logger.info(f"📊 Session {session_id} ended")
            self.logger.info(f"   Average posture score: {session_stats.get('average_posture_score', 0):.1%}")
            self.logger.info(f"   Session duration: {session_stats.get('session_duration', 0):.1f}s")
            self.logger.info(f"   Improvement trend: {session_stats.get('improvement_trend', 0):+.3f}")

            # Add to progress tracker
            if self.config.enable_progress_tracking:
                self.progress_tracker.add_session_data(session_id, session_stats)

            # Trigger custom callback if set
            if hasattr(self, 'on_session_complete'):
                self.on_session_complete(session_id, session_stats, session_summary)

        def on_detailed_posture_feedback(session_id: str, detailed_feedback: Dict[str, Any]):
            score = detailed_feedback.get('score', 0)
            if score < 0.4:  # Critical posture issue
                recommendations = detailed_feedback.get('recommendations', [])
                if recommendations:
                    self.logger.warning(f"⚠️ Posture alert: {recommendations[0]}")

            # Trigger custom callback if set
            if hasattr(self, 'on_posture_feedback'):
                self.on_posture_feedback(session_id, detailed_feedback)

        def on_real_time_feedback(session_id: str, feedback_message: str):
            self.logger.info(f"💬 {feedback_message}")

            # Trigger custom callback if set
            if hasattr(self, 'on_real_time_guidance'):
                self.on_real_time_guidance(session_id, feedback_message)

        # Register callbacks
        self.session_manager.register_global_callback('session_end', on_session_end)
        self.session_manager.register_global_callback('detailed_posture_feedback', on_detailed_posture_feedback)
        self.session_manager.register_global_callback('real_time_feedback', on_real_time_feedback)

        # Real-time processor callbacks
        def on_frame_processed(result: Dict[str, Any]):
            # Process frame result for session
            if self.current_session_id and result.get('posture_analysis'):
                session = self.session_manager.get_session(self.current_session_id)
                if session:
                    # Create real-time data
                    real_time_data = RealTimeData(
                        video_frame=result.get('original_frame'),
                        timestamp=result.get('timestamp', time.time()),
                        session_id=self.current_session_id
                    )

                    # Process through session
                    session.process_real_time_data(real_time_data)

        self.real_time_processor.register_callback('frame_processed', on_frame_processed)

    def start_meditation_session(self, meditation_type: str = "general", video_source: int = 0) -> str:
        """Start a new meditation session with live posture detection"""
        try:
            if self.is_active:
                self.logger.warning("Session already active. Stop current session first.")
                return self.current_session_id

            # Generate session ID
            session_id = f"meditation_{int(time.time())}_{self.user_id}"

            # Create session
            session = self.session_manager.create_session(session_id)
            session.start()

            # Start real-time processing
            if not self.real_time_processor.start(video_source):
                self.logger.error("Failed to start real-time processing")
                self.session_manager.end_session(session_id)
                return None

            # Set state
            self.current_session_id = session_id
            self.is_active = True

            # Start auto-save thread
            self._start_auto_save()

            self.logger.info(f"🧘 Meditation session started: {session_id}")
            self.logger.info(f"   Type: {meditation_type}")
            self.logger.info(f"   Video source: {video_source}")
            self.logger.info("   Live posture detection active")

            return session_id

        except Exception as e:
            self.logger.error(f"Failed to start meditation session: {e}")
            return None

    def stop_meditation_session(self) -> Optional[Dict[str, Any]]:
        """Stop the current meditation session"""
        try:
            if not self.is_active or not self.current_session_id:
                self.logger.warning("No active session to stop")
                return None

            session_id = self.current_session_id

            # Stop real-time processing
            self.real_time_processor.stop()

            # Stop auto-save
            self._stop_auto_save_thread()

            # End session
            session_result = self.session_manager.end_session(session_id)

            # Reset state
            self.is_active = False
            self.current_session_id = None

            self.logger.info(f"🏁 Meditation session ended: {session_id}")

            return session_result

        except Exception as e:
            self.logger.error(f"Error stopping meditation session: {e}")
            return None

    def get_live_feedback(self) -> Optional[Dict[str, Any]]:
        """Get current live feedback"""
        try:
            if not self.is_active:
                return None

            # Get latest processing result
            latest_result = self.real_time_processor.get_latest_result()
            if not latest_result:
                return None

            # Get current session data
            session = self.session_manager.get_session(self.current_session_id)
            if not session:
                return None

            current_posture = session.get_current_posture_data()

            return {
                'session_id': self.current_session_id,
                'posture_score': current_posture.score,
                'recommendations': current_posture.recommendations[:3],  # Top 3
                'body_part_scores': current_posture.body_part_scores,
                'stability_score': current_posture.stability_score,
                'visual_feedback_frame': latest_result.get('visual_feedback_frame'),
                'performance_metrics': self.real_time_processor.get_performance_metrics()
            }

        except Exception as e:
            self.logger.error(f"Error getting live feedback: {e}")
            return None

    def get_session_statistics(self) -> Optional[Dict[str, Any]]:
        """Get current session statistics"""
        try:
            if not self.current_session_id:
                return None

            session = self.session_manager.get_session(self.current_session_id)
            if not session:
                return None

            return session.get_session_statistics()

        except Exception as e:
            self.logger.error(f"Error getting session statistics: {e}")
            return None

    def get_progress_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get progress analytics"""
        try:
            if not self.config.enable_progress_tracking:
                return {'error': 'Progress tracking disabled'}

            return self.progress_tracker.get_progress_analytics(days)

        except Exception as e:
            self.logger.error(f"Error getting progress analytics: {e}")
            return {'error': str(e)}

    def pause_session(self):
        """Pause the current session"""
        if self.is_active:
            self.real_time_processor.pause()
            self.logger.info("Session paused")

    def resume_session(self):
        """Resume the current session"""
        if self.is_active:
            self.real_time_processor.resume()
            self.logger.info("Session resumed")

    def set_callback(self, event_name: str, callback: Callable):
        """Set custom callback for events"""
        setattr(self, f'on_{event_name}', callback)
        self.logger.info(f"Custom callback set for: {event_name}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'user_id': self.user_id,
                'is_active': self.is_active,
                'current_session_id': self.current_session_id,
                'config': {
                    'posture_detection_enabled': self.config.enable_posture_detection,
                    'visual_feedback_enabled': self.config.enable_visual_feedback,
                    'progress_tracking_enabled': self.config.enable_progress_tracking,
                    'target_fps': self.config.target_fps
                },
                'components': {
                    'enhanced_posture_detector': self.enhanced_posture_detector is not None,
                    'vision_encoder': self.vision_encoder is not None,
                    'session_manager': self.session_manager is not None,
                    'progress_tracker': self.progress_tracker is not None,
                    'real_time_processor': self.real_time_processor is not None
                },
                'real_time_processor_status': self.real_time_processor.get_system_status() if self.real_time_processor else None,
                'active_sessions': self.session_manager.get_active_sessions() if self.session_manager else [],
                'total_sessions_today': len(self.progress_tracker.get_recent_sessions(1)) if self.progress_tracker else 0
            }

            return status

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

    def export_session_data(self, session_id: str = None, format: str = 'json') -> str:
        """Export session data"""
        try:
            target_session_id = session_id or self.current_session_id
            if not target_session_id:
                return '{"error": "No session specified"}'

            session = self.session_manager.get_session(target_session_id)
            if session:
                return session.export_session_data(format)
            else:
                # Try historical data
                historical = self.session_manager.get_session_history(target_session_id)
                if historical:
                    return json.dumps(historical, indent=2)
                else:
                    return '{"error": "Session not found"}'

        except Exception as e:
            return f'{{"error": "{str(e)}"}}'

    def export_progress_data(self, format: str = 'json') -> str:
        """Export all progress data"""
        try:
            if not self.config.enable_progress_tracking:
                return '{"error": "Progress tracking disabled"}'

            return self.progress_tracker.export_progress_data(format)

        except Exception as e:
            return f'{{"error": "{str(e)}"}}'

    def _start_auto_save(self):
        """Start auto-save thread"""
        if self.config.auto_save_interval_seconds > 0:
            self._stop_auto_save.clear()
            self.auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
            self.auto_save_thread.start()

    def _stop_auto_save_thread(self):
        """Stop auto-save thread"""
        self._stop_auto_save.set()
        if self.auto_save_thread and self.auto_save_thread.is_alive():
            self.auto_save_thread.join(timeout=5.0)

    def _auto_save_loop(self):
        """Auto-save loop"""
        while not self._stop_auto_save.wait(self.config.auto_save_interval_seconds):
            try:
                if self.is_active and self.current_session_id:
                    # Save current session state
                    session_data = self.export_session_data()
                    # In a real implementation, you would save to persistent storage
                    self.logger.debug(f"Auto-saved session data: {len(session_data)} chars")
            except Exception as e:
                self.logger.error(f"Auto-save error: {e}")

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.is_active:
                self.stop_meditation_session()

            self._stop_auto_save_thread()

            if self.real_time_processor:
                self.real_time_processor.stop()

            if self.vision_encoder:
                self.vision_encoder.cleanup()

            self.logger.info("Meditation module cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except:
            pass


# Example usage and testing functions
def create_demo_integration(user_id: str = "demo_user") -> MeditationModuleIntegration:
    """Create a demo meditation integration"""

    config = MeditationConfig(
        enable_posture_detection=True,
        enable_visual_feedback=True,
        enable_progress_tracking=True,
        target_fps=30,
        frame_width=640,
        frame_height=480
    )

    integration = MeditationModuleIntegration(user_id, config)

    # Set up demo callbacks
    def demo_session_complete(session_id: str, stats: Dict[str, Any], summary: Dict[str, Any]):
        print(f"\n🎉 Demo Session Complete!")
        print(f"Session ID: {session_id}")
        print(f"Duration: {stats.get('session_duration', 0):.1f} seconds")
        print(f"Average Posture Score: {stats.get('average_posture_score', 0):.1%}")
        print(f"Best Score: {stats.get('best_posture_score', 0):.1%}")
        print(f"Improvement: {stats.get('improvement_trend', 0):+.3f}")

    def demo_posture_feedback(session_id: str, feedback: Dict[str, Any]):
        score = feedback.get('score', 0)
        recommendations = feedback.get('recommendations', [])

        if score < 0.5 and recommendations:
            print(f"📢 Posture guidance: {recommendations[0]}")

    def demo_real_time_guidance(session_id: str, message: str):
        print(f"🗣️ Real-time: {message}")

    integration.set_callback('session_complete', demo_session_complete)
    integration.set_callback('posture_feedback', demo_posture_feedback)
    integration.set_callback('real_time_guidance', demo_real_time_guidance)

    return integration

if __name__ == "__main__":
    # Demo usage
    print("🧘 Meditation Module Integration Demo")
    print("=====================================")

    # Create demo integration
    meditation = create_demo_integration("test_user_123")

    # Print system status
    status = meditation.get_system_status()
    print(f"\nSystem Status:")
    print(f"Components loaded: {sum(1 for v in status['components'].values() if v)}/{len(status['components'])}")
    print(f"Configuration: {status['config']}")

    print("\n✅ Integration ready!")
    print("\nTo use:")
    print("1. session_id = meditation.start_meditation_session()")
    print("2. feedback = meditation.get_live_feedback()")
    print("3. result = meditation.stop_meditation_session()")
    print("4. analytics = meditation.get_progress_analytics()")
