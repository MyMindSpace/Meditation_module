#!/usr/bin/env python3
"""
Enhanced Live Capture with Real-time Posture Detection
Integrates posture analysis into the live meditation pipeline
"""

import argparse
import datetime
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf

# Import our working posture detector
from working_posture_detection import WorkingPostureDetector, PostureConfig, PostureThresholds


class LiveCaptureWithPosture:
    """Enhanced live capture with real-time posture detection and feedback"""
    
    def __init__(self, output_dir: str = "preprocess_input"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Audio settings
        self.audio_sample_rate = 22050
        self.audio_channels = 1
        self.audio_dtype = np.float32
        
        # Video settings
        self.video_fps = 30
        self.video_resolution = (640, 480)
        self.video_codec = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Recording state
        self.is_recording = False
        self.audio_data = []
        self.video_writer = None
        
        # Posture detection
        self.posture_detector = None
        self.posture_data = []
        self.posture_feedback_active = True
        
        # Session tracking
        self.session_start_time = None
        self.frame_count = 0
        self.posture_scores = []
        
    def initialize_posture_detection(self):
        """Initialize the posture detection system"""
        print("Initializing posture detection system...")
        
        config = PostureConfig(
            show_skeleton=True,
            show_score=True,
            show_corrections=True,
            show_detailed_metrics=True,
            smoothing_window=10,
            confidence_threshold=0.7
        )
        
        thresholds = PostureThresholds(
            excellent_threshold=0.85,
            good_threshold=0.70,
            fair_threshold=0.55,
            poor_threshold=0.40
        )
        
        self.posture_detector = WorkingPostureDetector(config, thresholds)
        print("Posture detection system ready!")
    
    def list_audio_devices(self):
        """List available audio input devices"""
        print("Available audio input devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} (max inputs: {device['max_input_channels']})")
    
    def list_video_devices(self):
        """List available video capture devices"""
        print("Available video capture devices:")
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"  {i}: Camera {i}")
                cap.release()
    
    def audio_callback(self, indata, frames, time, status):
        """Audio recording callback"""
        if status:
            print(f"Audio recording status: {status}")
        if self.is_recording:
            self.audio_data.append(indata.copy())
    
    def capture_with_posture_feedback(self, 
                                    duration: int = 30, 
                                    audio_device: Optional[int] = None,
                                    video_device: int = 0,
                                    show_posture_feedback: bool = True) -> Tuple[str, str, str]:
        """
        Capture audio and video with real-time posture feedback
        
        Args:
            duration: Recording duration in seconds
            audio_device: Audio input device index (None for default)
            video_device: Video input device index
            show_posture_feedback: Whether to show real-time posture feedback
            
        Returns:
            Tuple of (audio_file_path, video_file_path, posture_data_path)
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_time = time.time()
        
        # Setup file paths
        audio_filename = f"live_audio_{timestamp}.wav"
        audio_path = self.output_dir / audio_filename
        
        video_filename = f"live_video_{timestamp}.mp4"
        video_path = self.output_dir / video_filename
        
        posture_filename = f"posture_data_{timestamp}.json"
        posture_path = self.output_dir / posture_filename
        
        # Initialize posture detection
        if show_posture_feedback:
            self.initialize_posture_detection()
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_device)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video device {video_device}")
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.video_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Initialize video writer
        self.video_writer = cv2.VideoWriter(
            str(video_path), 
            self.video_codec, 
            self.video_fps, 
            self.video_resolution
        )
        
        # Setup audio recording
        self.audio_data = []
        self.is_recording = True
        
        print(f"Starting meditation session with posture feedback for {duration} seconds...")
        print("Controls:")
        print("   - 'q' - Quit session early")
        print("   - 's' - Save screenshot")
        print("   - 'p' - Toggle posture feedback")
        print("   - 'c' - Calibrate posture")
        print("   - 'h' - Show help")
        print("\nFocus on your posture and breathing...")
        
        # Start audio recording
        try:
            with sd.InputStream(
                device=audio_device,
                channels=self.audio_channels,
                samplerate=self.audio_sample_rate,
                dtype=self.audio_dtype,
                callback=self.audio_callback
            ):
                start_time = time.time()
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    self.frame_count = frame_count
                    
                    # Process frame with posture detection
                    if self.posture_detector and self.posture_feedback_active:
                        annotated_frame, feedback_data = self.posture_detector.process_frame(frame)
                        
                        # Store posture data for later analysis
                        if feedback_data['landmarks_detected']:
                            posture_record = {
                                'timestamp': time.time() - start_time,
                                'frame_number': frame_count,
                                'posture_score': feedback_data['posture_score'],
                                'raw_score': feedback_data.get('raw_score', feedback_data['posture_score']),
                                'feedback_level': feedback_data['feedback_level'],
                                'metrics': feedback_data.get('metrics', {}),
                                'landmarks_detected': True
                            }
                            self.posture_data.append(posture_record)
                            self.posture_scores.append(feedback_data['posture_score'])
                        else:
                            # No detection
                            posture_record = {
                                'timestamp': time.time() - start_time,
                                'frame_number': frame_count,
                                'posture_score': 0.0,
                                'feedback_level': 'no_detection',
                                'landmarks_detected': False
                            }
                            self.posture_data.append(posture_record)
                        
                        display_frame = annotated_frame
                    else:
                        # No posture detection, just show raw frame
                        display_frame = frame.copy()
                        cv2.putText(display_frame, "Posture Detection: OFF", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Add session info
                    elapsed_time = time.time() - start_time
                    remaining_time = max(0, duration - elapsed_time)
                    
                    # Session timer
                    timer_text = f"Session: {int(elapsed_time//60):02d}:{int(elapsed_time%60):02d} / {int(duration//60):02d}:{int(duration%60):02d}"
                    cv2.putText(display_frame, timer_text, (10, display_frame.shape[0] - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Progress bar
                    progress = min(1.0, elapsed_time / duration)
                    bar_width = 300
                    bar_height = 20
                    bar_x = (display_frame.shape[1] - bar_width) // 2
                    bar_y = display_frame.shape[0] - 30
                    
                    # Background
                    cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (40, 40, 40), -1)
                    # Progress
                    progress_width = int(bar_width * progress)
                    cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
                    # Border
                    cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
                    
                    # Save frame to video
                    self.video_writer.write(frame)  # Save original frame, not annotated
                    
                    # Display frame
                    cv2.imshow('Meditation Session with Posture Feedback', display_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Session stopped by user")
                        break
                    elif key == ord('s'):
                        timestamp_screenshot = int(time.time())
                        screenshot_filename = f"meditation_session_{timestamp_screenshot}.jpg"
                        cv2.imwrite(screenshot_filename, display_frame)
                        print(f"Screenshot saved: {screenshot_filename}")
                    elif key == ord('p'):
                        self.posture_feedback_active = not self.posture_feedback_active
                        status = "ON" if self.posture_feedback_active else "OFF"
                        print(f"Posture feedback: {status}")
                    elif key == ord('c'):
                        if self.posture_detector:
                            print("Posture calibration triggered")
                            # Could implement calibration logic here
                    elif key == ord('h'):
                        self.show_help()
                    
                    # Check if duration reached
                    if elapsed_time >= duration:
                        print("Session duration completed")
                        break
                        
        except Exception as e:
            print(f"❌ Error during recording: {e}")
            raise
        finally:
            # Cleanup
            self.is_recording = False
            cap.release()
            cv2.destroyAllWindows()
            
            if self.video_writer:
                self.video_writer.release()
        
        # Save audio data
        if self.audio_data:
            audio_array = np.concatenate(self.audio_data, axis=0)
            sf.write(str(audio_path), audio_array, self.audio_sample_rate)
            print(f"Audio saved: {audio_path}")
        else:
            print("No audio data captured")
            audio_path = None
        
        # Save posture data
        if self.posture_data:
            posture_summary = {
                'session_info': {
                    'timestamp': timestamp,
                    'duration_seconds': duration,
                    'total_frames': len(self.posture_data),
                    'posture_detection_enabled': self.posture_detector is not None,
                    'session_start_time': self.session_start_time
                },
                'posture_statistics': {
                    'average_score': np.mean(self.posture_scores) if self.posture_scores else 0.0,
                    'max_score': np.max(self.posture_scores) if self.posture_scores else 0.0,
                    'min_score': np.min(self.posture_scores) if self.posture_scores else 0.0,
                    'detection_rate': len([p for p in self.posture_data if p['landmarks_detected']]) / len(self.posture_data) if self.posture_data else 0.0
                },
                'frame_data': self.posture_data
            }
            
            with open(posture_path, 'w', encoding='utf-8') as f:
                json.dump(posture_summary, f, indent=2, ensure_ascii=False)
            print(f"Posture data saved: {posture_path}")
            
            # Also convert to posture_scores.json format for pipeline compatibility
            try:
                from convert_posture_data import convert_posture_data_to_scores
                convert_posture_data_to_scores(posture_path, "preprocess_output/posture_scores.json")
                print("Posture data converted to posture_scores.json format")
            except ImportError:
                print("Note: convert_posture_data.py not found - manual conversion needed")
            except Exception as e:
                print(f"Warning: Could not convert to posture_scores.json: {e}")
        else:
            print("No posture data captured")
            posture_path = None
        
        print(f"Video saved: {video_path}")
        
        # Print session summary
        self.print_session_summary()
        
        return str(audio_path) if audio_path else None, str(video_path), str(posture_path) if posture_path else None
    
    def show_help(self):
        """Show help information"""
        help_text = """
🧘 MEDITATION SESSION CONTROLS:

📋 Basic Controls:
   • 'q' - Quit session early
   • 's' - Save screenshot
   • 'h' - Show this help

🧘 Posture Controls:
   • 'p' - Toggle posture feedback on/off
   • 'c' - Calibrate posture (future feature)

📊 What You'll See:
   • Real-time posture score and feedback
   • Detailed posture metrics (shoulders, head, spine, hips)
   • Detection mode (FULL BODY / UPPER BODY)
   • Session timer and progress bar
   • Performance indicators (FPS)

🎯 Posture Feedback Levels:
   • 🌟 Excellent (85%+) - Perfect posture!
   • ✓ Good (70-85%) - Great posture
   • ⚠️ Fair (55-70%) - Minor adjustments needed
   • ❌ Poor (<55%) - Major corrections needed

💡 Tips for Better Posture:
   • Sit with your back straight
   • Keep shoulders relaxed and level
   • Align your head over your shoulders
   • Keep your spine in a natural curve
   • Breathe deeply and relax
        """
        print(help_text)
    
    def print_session_summary(self):
        """Print a summary of the meditation session"""
        if not self.posture_data:
            print("\n📊 Session Summary: No posture data available")
            return
        
        total_frames = len(self.posture_data)
        detected_frames = len([p for p in self.posture_data if p['landmarks_detected']])
        detection_rate = (detected_frames / total_frames) * 100 if total_frames > 0 else 0
        
        if self.posture_scores:
            avg_score = np.mean(self.posture_scores)
            max_score = np.max(self.posture_scores)
            min_score = np.min(self.posture_scores)
            
            # Determine overall session quality
            if avg_score >= 0.85:
                quality = "🌟 Excellent"
            elif avg_score >= 0.70:
                quality = "✓ Good"
            elif avg_score >= 0.55:
                quality = "⚠️ Fair"
            else:
                quality = "❌ Needs Improvement"
        else:
            avg_score = max_score = min_score = 0.0
            quality = "❓ No Data"
        
        print(f"\nMEDITATION SESSION SUMMARY")
        print(f"=" * 50)
        print(f"Posture Statistics:")
        print(f"   - Overall Quality: {quality}")
        print(f"   - Average Score: {avg_score:.1%}")
        print(f"   - Best Score: {max_score:.1%}")
        print(f"   - Lowest Score: {min_score:.1%}")
        print(f"   - Detection Rate: {detection_rate:.1f}%")
        print(f"   - Total Frames: {total_frames}")
        print(f"   - Detected Frames: {detected_frames}")
        print(f"=" * 50)
        
        # Provide feedback based on performance
        if avg_score >= 0.85:
            print("Outstanding! Your posture was excellent throughout the session.")
        elif avg_score >= 0.70:
            print("Great job! Your posture was good with room for minor improvements.")
        elif avg_score >= 0.55:
            print("Good effort! Focus on maintaining better posture consistency.")
        else:
            print("Keep practicing! Consider posture exercises and mindfulness.")
        
        print(f"\nData saved for AI analysis in the meditation pipeline.")


def main():
    parser = argparse.ArgumentParser(description="Live Capture with Posture Detection for Meditation")
    parser.add_argument("--mode", choices=["audio", "video", "both"], default="both",
                        help="Capture mode: audio only, video only, or both")
    parser.add_argument("--duration", type=int, default=30,
                        help="Recording duration in seconds")
    parser.add_argument("--output-dir", type=str, default="preprocess_input",
                        help="Output directory for captured files")
    parser.add_argument("--audio-device", type=int, default=None,
                        help="Audio input device index")
    parser.add_argument("--video-device", type=int, default=0,
                        help="Video input device index")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio and video devices")
    parser.add_argument("--create-librispeech", action="store_true",
                        help="Create LibriSpeech directory structure for audio")
    parser.add_argument("--no-posture", action="store_true",
                        help="Disable posture detection and feedback")
    
    args = parser.parse_args()
    
    capture = LiveCaptureWithPosture(args.output_dir)
    
    if args.list_devices:
        capture.list_audio_devices()
        print()
        capture.list_video_devices()
        return
    
    try:
        if args.mode == "both":
            audio_file, video_file, posture_file = capture.capture_with_posture_feedback(
                duration=args.duration,
                audio_device=args.audio_device,
                video_device=args.video_device,
                show_posture_feedback=not args.no_posture
            )
            
            # Create LibriSpeech structure if requested
            if args.create_librispeech and audio_file:
                # This would need to be implemented similar to the original live_capture.py
                print(f"LibriSpeech structure creation for: {audio_file}")
            
        else:
            print(f"Mode '{args.mode}' not fully implemented yet. Use 'both' for full functionality.")
            return 1
        
        print("\nCapture complete! Files are ready for the meditation pipeline.")
        print(f"Files location: {args.output_dir}")
        
        if audio_file:
            print(f"Audio: {audio_file}")
        if video_file:
            print(f"Video: {video_file}")
        if posture_file:
            print(f"Posture Data: {posture_file}")
        
    except Exception as e:
        print(f"Error during capture: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
