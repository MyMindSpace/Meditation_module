# main_integration.py
"""
Main Integration Script

Demonstrates the complete meditation module working together:
- Real-time session management
- Multi-modal processing
- Progress tracking
- Personalized recommendations
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Import all components
from session_manager import SessionManager, SessionConfig, RealTimeData
from real_time_processor import RealTimeProcessor, ProcessingConfig, StreamData
from progress_tracker import ProgressTracker, ProgressMetric
from Core_engine.meditation_selector import MeditationSelectorModule

# Import encoders and processors
from Encoders.vision_encoder import VisionEncoder
from Encoders.audio_encoder import AudioEncoder
from Encoders.user_profile_encoder import UserProfileEncoder
from preprocessing_unit.video_preprocessor import VideoPreprocessor
from preprocessing_unit.audio_preprocessor import AudioPreprocessor
from preprocessing_unit.user_preprocessor import UserDataPreprocessor
from preprocessing_unit.diagnosis_processor import DiagnosisProcessor


class MeditationModuleIntegration:
    """
    Complete meditation module integration
    Orchestrates all components for a full meditation experience
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # Initialize core components
        self.session_manager = SessionManager(enable_logging=True)
        self.progress_tracker = ProgressTracker(user_id=user_id)
        self.meditation_selector = MeditationSelectorModule(use_ml=True)  # Use rule-based for simplicity
        
        # Initialize processing components
        processing_config = ProcessingConfig(
            enable_video=True,
            enable_audio=True,
            video_fps=10.0,  # Lower FPS for demo
            max_queue_size=5
        )
        self.real_time_processor = RealTimeProcessor(processing_config)
        
        # Initialize encoders
        self.vision_encoder = VisionEncoder(use_torch=False)  # Use basic version
        self.audio_encoder = AudioEncoder()
        self.user_profile_encoder = UserProfileEncoder()
        
        # Initialize preprocessors
        self.video_preprocessor = VideoPreprocessor()
        self.audio_preprocessor = AudioPreprocessor()
        self.user_preprocessor = UserDataPreprocessor()
        self.diagnosis_processor = DiagnosisProcessor()
        
        # Setup callbacks
        self._setup_callbacks()
        
        # User profile and session data
        self.user_profile = self._load_or_create_user_profile()
        self.current_session_id = None
        
        # Real-time data buffer
        self.session_data_buffer = []
    
    def _setup_callbacks(self):
        """Setup callbacks between components"""
        
        # Session manager callbacks
        def on_session_started(session_id):
            print(f"🧘 Session {session_id} started")
            self.current_session_id = session_id
            
        def on_posture_correction(session_id, correction):
            print(f"📏 Posture correction: {correction}")
            
        def on_session_completed(session_id, metrics):
            print(f"✅ Session {session_id} completed")
            self._process_session_completion(session_id, metrics)
            
        self.session_manager.register_callback('session_started', on_session_started)
        self.session_manager.register_callback('posture_correction', on_posture_correction)
        self.session_manager.register_callback('session_completed', on_session_completed)
        
        # Real-time processor callbacks
        def on_output_ready(output):
            if self.current_session_id:
                self._process_real_time_output(output)
                
        def on_performance_update(stats):
            if stats['frames_processed'] % 50 == 0:  # Log every 50 frames
                print(f"📊 Processed {stats['frames_processed']} frames, "
                      f"avg time: {stats['average_processing_time']*1000:.1f}ms")
        
        self.real_time_processor.register_callback('output_ready', on_output_ready)
        self.real_time_processor.register_callback('performance_update', on_performance_update)
    
    def _load_or_create_user_profile(self) -> Dict[str, Any]:
        """Load existing user profile or create new one"""
        profile_file = Path(f"user_profiles/{self.user_id}.json")
        
        if profile_file.exists():
            with profile_file.open('r') as f:
                return json.load(f)
        else:
            # Create default profile
            return {
                "user_id": self.user_id,
                "profile": {
                    "age": 30,
                    "meditation_experience": "beginner",
                    "goals": ["stress reduction", "better focus"],
                    "availability": {
                        "morning": True,
                        "afternoon": False,
                        "evening": True
                    },
                    "preferred_duration_minutes": 15
                },
                "sessions": []
            }
    
    def create_sample_data(self):
        """Create sample data files for testing"""
        # Create directories
        Path("preprocess_input").mkdir(exist_ok=True)
        Path("preprocess_output").mkdir(exist_ok=True)
        Path("user_profiles").mkdir(exist_ok=True)
        
        # Sample user feedback
        user_feedback = {
            "userId": self.user_id,
            "feedback": "I'm feeling very stressed and anxious today. I also have some difficulty focusing on my work. The meditation session helped me feel more calm and centered.",
            "recent_sessions": [
                {
                    "sessionId": "session_001",
                    "timestamp": "2024-09-01 09:00:00",
                    "feedbackText": "Great session, felt very relaxed afterwards",
                    "rating": 4.5,
                    "completed": True
                },
                {
                    "sessionId": "session_002", 
                    "timestamp": "2024-09-02 09:15:00",
                    "feedbackText": "Had trouble focusing today, mind was wandering",
                    "rating": 3.0,
                    "completed": True
                }
            ]
        }
        
        with open("preprocess_input/user_feedback.json", 'w') as f:
            json.dump(user_feedback, f, indent=2)
        
        # Sample diagnosis data
        diagnosis_data = {
            "user_id": self.user_id,
            "mental_disorder": "generalized anxiety disorder",
            "severity": "moderate",
            "symptoms": [
                "excessive worry",
                "restlessness", 
                "difficulty concentrating"
            ],
            "description": "Patient reports feeling anxious and overwhelmed with work stress. Has difficulty concentrating and experiences restlessness. Seeks meditation for stress management."
        }
        
        with open("preprocess_input/diagnosis_data.json", 'w') as f:
            json.dump(diagnosis_data, f, indent=2)
        
        # Sample diary entry
        diary_entry = {
            "user_id": self.user_id,
            "date": "2024-09-08",
            "entry": "Today I want to work on my breathing technique and try to improve my posture during meditation. I've been feeling stressed about upcoming deadlines.",
            "intent": "stress relief and posture improvement"
        }
        
        with open("preprocess_input/diary_entry.json", 'w') as f:
            json.dump(diary_entry, f, indent=2)
        
        # Save user profile
        profile_file = Path(f"user_profiles/{self.user_id}.json")
        profile_file.parent.mkdir(exist_ok=True)
        with profile_file.open('w') as f:
            json.dump(self.user_profile, f, indent=2)
        
        print("✅ Sample data files created")
    
    def preprocess_all_data(self):
        """Run all preprocessing steps"""
        print("🔄 Starting preprocessing pipeline...")
        
        # Process user feedback
        user_feedback_result = self.user_preprocessor.process_user_feedback(
            json.load(open("preprocess_input/user_feedback.json"))
        )
        
        with open("preprocess_output/user_feedback_processed.json", 'w') as f:
            json.dump(user_feedback_result, f, indent=2)
        
        # Process diagnosis
        diagnosis_result = self.diagnosis_processor.process_diagnosis_entry(
            json.load(open("preprocess_input/diagnosis_data.json"))
        )
        
        with open("preprocess_output/diagnosis_processed.json", 'w') as f:
            json.dump(diagnosis_result, f, indent=2)
        
        # Generate user profile encoding
        user_encoding = self.user_profile_encoder.process_user_data(self.user_profile)
        
        with open("preprocess_output/user_profile_encoded.json", 'w') as f:
            json.dump(user_encoding, f, indent=2)
        
        print("✅ Preprocessing completed")
    
    def get_personalized_recommendations(self) -> Dict[str, Any]:
        """Get personalized meditation recommendations"""
        print("🎯 Generating personalized recommendations...")
        
        # Use processed data for recommendations
        recommendations = self.meditation_selector.select_meditation(
            "preprocess_output/user_feedback_processed.json",
            "preprocess_output/diagnosis_processed.json"
        )
        
        return recommendations
    
    def start_meditation_session(self, meditation_type: str, duration_minutes: float = 15.0):
        """Start a complete meditation session"""
        print(f"🚀 Starting {meditation_type} meditation session ({duration_minutes} min)")
        
        # Create session configuration
        session_config = SessionConfig(
            session_id=f"session_{int(time.time())}",
            user_id=self.user_id,
            meditation_type=meditation_type,
            planned_duration=duration_minutes * 60,  # Convert to seconds
            enable_posture_detection=True,
            posture_correction_interval=20.0,
            quality_check_interval=10.0
        )
        
        # Create and start session
        session_id = self.session_manager.create_session(session_config)
        success = self.session_manager.start_session(session_id)
        
        if not success:
            print("❌ Failed to start session")
            return None
        
        # Start real-time processing
        self.real_time_processor.start_processing()
        
        return session_id
    
    def simulate_session_data(self, session_id: str, duration_seconds: float):
        """Simulate real-time session data"""
        print("📡 Simulating real-time session data...")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            
            # Simulate video frame (posture gradually improves)
            progress = (current_time - start_time) / duration_seconds
            base_posture = 0.6 + 0.3 * progress + 0.1 * np.random.random()
            video_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Simulate audio chunk
            audio_chunk = np.random.randn(8000).astype(np.float32) * 0.1  # 0.5s of audio
            
            # Create real-time data
            rt_data = RealTimeData(
                timestamp=current_time,
                video_frame=video_frame,
                audio_chunk=audio_chunk,
                posture_score=base_posture,
                engagement_level=0.7 + 0.2 * np.random.random()
            )
            
            # Process with session manager
            self.session_manager.process_real_time_data(session_id, rt_data)
            
            # Also send to real-time processor
            stream_data = StreamData(
                timestamp=current_time,
                video_frame=video_frame,
                audio_chunk=audio_chunk
            )
            self.real_time_processor.add_stream_data(stream_data)
            
            frame_count += 1
            time.sleep(0.1)  # 10 FPS simulation
        
        print(f"📊 Simulated {frame_count} frames of data")
    
    def _process_real_time_output(self, output):
        """Process real-time processor output"""
        # Store session data for later analysis
        session_data = {
            'timestamp': output.timestamp,
            'processing_time': output.processing_time,
            'quality_metrics': output.quality_metrics
        }
        
        if output.decisions and 'adjustments' in output.decisions:
            adjustments = output.decisions['adjustments']
            session_data['adjustments'] = adjustments
        
        self.session_data_buffer.append(session_data)
    
    def _process_session_completion(self, session_id: str, metrics):
        """Process completed session"""
        print("📈 Processing session completion...")
        
        # Convert metrics to dict for progress tracker
        session_metrics = {
            'duration': metrics.duration,
            'planned_duration': metrics.planned_duration,
            'completion_rate': metrics.completion_rate,
            'average_posture_score': metrics.average_posture_score,
            'engagement_score': metrics.engagement_score,
            'meditation_type': metrics.meditation_type,
            'interruptions': metrics.interruptions,
            'user_rating': metrics.user_rating
        }
        
        # Simulate user feedback
        user_feedback = {
            'feedback_text': 'The session helped me feel more centered and focused. Good posture reminders.',
            'rating': np.random.uniform(3.5, 5.0)
        }
        
        # Add to progress tracker
        self.progress_tracker.add_session_data(session_id, session_metrics, user_feedback)
        
        # Update user profile with session
        session_record = {
            'session_id': session_id,
            'timestamp': metrics.start_time.isoformat(),
            'meditation_type': metrics.meditation_type,
            'duration': metrics.duration,
            'completion_rate': metrics.completion_rate,
            'rating': user_feedback['rating']
        }
        
        self.user_profile['sessions'].append(session_record)
        self._save_user_profile()
        
        # Clear session data buffer
        self.session_data_buffer = []
    
    def _save_user_profile(self):
        """Save updated user profile"""
        profile_file = Path(f"user_profiles/{self.user_id}.json")
        with profile_file.open('w') as f:
            json.dump(self.user_profile, f, indent=2)
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Get comprehensive progress report"""
        print("📊 Generating progress report...")
        return self.progress_tracker.get_progress_summary()
    
    def end_session(self, user_rating: float = 4.0):
        """End current session"""
        if self.current_session_id:
            print(f"🏁 Ending session {self.current_session_id}")
            
            # Stop real-time processing
            self.real_time_processor.stop_processing()
            
            # End session with rating
            metrics = self.session_manager.end_session(self.current_session_id, user_rating)
            self.current_session_id = None
            
            return metrics
        
        return None
    
    def cleanup(self):
        """Cleanup all components"""
        print("🧹 Cleaning up...")
        
        if self.current_session_id:
            self.end_session()
        
        self.session_manager.cleanup()
        self.real_time_processor.stop_processing()


def run_complete_demo():
    """Run complete meditation module demonstration"""
    print("🌟 Starting Complete Meditation Module Demo\n")
    
    # Initialize integration
    user_id = "demo_user_123"
    integration = MeditationModuleIntegration(user_id)
    
    try:
        # Step 1: Create sample data
        print("=== Step 1: Data Setup ===")
        integration.create_sample_data()
        
        # Step 2: Preprocess data
        print("\n=== Step 2: Data Preprocessing ===")
        integration.preprocess_all_data()
        
        # Step 3: Get recommendations
        print("\n=== Step 3: Personalized Recommendations ===")
        recommendations = integration.get_personalized_recommendations()
        print(f"🎯 Recommended meditation: {recommendations['recommendations'][0]['meditation_type']}")
        print(f"📝 Rationale: {recommendations['recommendations'][0]['rationale']}")
        
        # Step 4: Start meditation session
        print("\n=== Step 4: Meditation Session ===")
        selected_meditation = recommendations['recommendations'][0]['meditation_type']
        session_id = integration.start_meditation_session(selected_meditation, duration_minutes=2.0)  # Short demo
        
        if session_id:
            # Step 5: Simulate session
            print("\n=== Step 5: Real-time Processing ===")
            integration.simulate_session_data(session_id, duration_seconds=30.0)  # 30 second demo
            
            # Step 6: End session
            print("\n=== Step 6: Session Completion ===")
            metrics = integration.end_session(user_rating=4.5)
            
            if metrics:
                print(f"✅ Session completed with {metrics.completion_rate:.1%} completion rate")
                print(f"📏 Average posture score: {metrics.average_posture_score:.2f}")
                print(f"⭐ User rating: {metrics.user_rating}")
        
        # Step 7: Progress report
        print("\n=== Step 7: Progress Analysis ===")
        progress_report = integration.get_progress_report()
        print(f"📈 Total sessions: {progress_report['total_sessions']}")
        
        if progress_report['current_scores']:
            print("Current scores:")
            for metric, score in progress_report['current_scores'].items():
                print(f"  • {metric.replace('_', ' ').title()}: {score:.2f}")
        
        if progress_report['recommendations']:
            print("Recommendations:")
            for rec in progress_report['recommendations'][:3]:
                print(f"  • {rec}")
        
        print("\n🎉 Demo completed successfully!")
        
        # Show file outputs
        print("\n📁 Generated files:")
        output_files = [
            "preprocess_output/user_feedback_processed.json",
            "preprocess_output/diagnosis_processed.json",
            "preprocess_output/user_profile_encoded.json",
            "meditation_recommendations_output.json",
            f"user_profiles/{user_id}.json",
            f"progress_data_{user_id}.json"
        ]
        
        for filepath in output_files:
            if Path(filepath).exists():
                print(f"  ✅ {filepath}")
            else:
                print(f"  ⚠️  {filepath} (may not exist)")
    
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        integration.cleanup()


def run_individual_component_tests():
    """Test individual components"""
    print("🔧 Running Individual Component Tests\n")
    
    # Test meditation selector
    print("=== Testing Meditation Selector ===")
    selector = MeditationSelectorModule(use_ml=False)
    
    # Create test data
    feedback_data = {"feedback": "I'm feeling stressed and need to relax"}
    diagnosis_data = {
        "mental_disorder": "anxiety disorder",
        "symptoms": ["worry", "restlessness"]
    }
    
    result = selector.select_meditation_direct(
        feedback_data["feedback"], 
        diagnosis_data
    )
    
    print(f"✅ Meditation Selector: {len(result['recommendations'])} recommendations")
    
    # Test progress tracker
    print("\n=== Testing Progress Tracker ===")
    tracker = ProgressTracker("test_user")
    
    # Add sample session
    session_metrics = {
        'duration': 600,
        'completion_rate': 0.9,
        'average_posture_score': 0.8,
        'engagement_score': 0.85,
        'meditation_type': 'mindfulness',
        'interruptions': 1
    }
    
    tracker.add_session_data("test_session", session_metrics)
    summary = tracker.get_progress_summary()
    
    print(f"✅ Progress Tracker: {summary['total_sessions']} sessions tracked")
    
    # Test user profile encoder
    print("\n=== Testing User Profile Encoder ===")
    encoder = UserProfileEncoder()
    
    user_data = {
        "profile": {
            "age": 28,
            "meditation_experience": "intermediate",
            "goals": ["stress reduction"]
        },
        "sessions": [
            {
                "date": "2024-09-01",
                "rating": 4.0,
                "posture_score": 0.7
            }
        ]
    }
    
    encoding = encoder.process_user_data(user_data)
    print(f"✅ User Profile Encoder: {len(encoding['embedding'])}D embedding")
    
    print("\n🎉 All component tests completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_individual_component_tests()
    else:
        run_complete_demo()