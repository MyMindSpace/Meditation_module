"""
Live Audio and Video Capture Module for Meditation Pipeline

This module captures live audio and video input from the user and saves them
in the required format for the meditation preprocessing pipeline.

Requirements:
- pip install opencv-python sounddevice soundfile numpy
"""

import argparse
import datetime
import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf


class LiveCapture:
    """Captures live audio and video from user input devices"""
    
    def __init__(self, output_dir: str = "preprocess_input"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Audio settings
        self.audio_sample_rate = 22050  # Standard rate used by librosa
        self.audio_channels = 1  # Mono
        self.audio_dtype = np.float32
        
        # Video settings
        self.video_fps = 30
        self.video_resolution = (640, 480)  # Width, Height
        self.video_codec = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Recording state
        self.is_recording = False
        self.audio_data = []
        self.video_writer = None
        
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
        for i in range(10):  # Check first 10 device indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"  {i}: Camera {i}")
                cap.release()
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio recording"""
        if status:
            print(f"Audio recording status: {status}")
        if self.is_recording:
            self.audio_data.append(indata.copy())
    
    def capture_audio_video(self, 
                          duration: int = 30, 
                          audio_device: Optional[int] = None,
                          video_device: int = 0) -> Tuple[str, str]:
        """
        Capture audio and video simultaneously
        
        Args:
            duration: Recording duration in seconds
            audio_device: Audio input device index (None for default)
            video_device: Video input device index
            
        Returns:
            Tuple of (audio_file_path, video_file_path)
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup audio recording
        audio_filename = f"live_audio_{timestamp}.wav"
        audio_path = self.output_dir / audio_filename
        
        # Setup video recording
        video_filename = f"live_video_{timestamp}.mp4"
        video_path = self.output_dir / video_filename
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_device)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video device {video_device}")
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.video_fps)
        
        # Initialize video writer
        self.video_writer = cv2.VideoWriter(
            str(video_path), 
            self.video_codec, 
            self.video_fps, 
            self.video_resolution
        )
        
        print(f"Starting recording for {duration} seconds...")
        print("Press 'q' to stop recording early or wait for automatic stop.")
        
        # Start recording
        self.is_recording = True
        self.audio_data = []
        
        # Start audio recording in a separate thread
        def record_audio():
            with sd.InputStream(
                device=audio_device,
                channels=self.audio_channels,
                samplerate=self.audio_sample_rate,
                dtype=self.audio_dtype,
                callback=self.audio_callback
            ):
                sd.sleep(duration * 1000)  # Sleep in milliseconds
        
        audio_thread = threading.Thread(target=record_audio)
        audio_thread.start()
        
        # Record video
        start_time = time.time()
        frame_count = 0
        
        try:
            while self.is_recording and (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Resize frame if needed
                if frame.shape[:2][::-1] != self.video_resolution:
                    frame = cv2.resize(frame, self.video_resolution)
                
                # Write frame
                self.video_writer.write(frame)
                frame_count += 1
                
                # Display frame (optional)
                cv2.imshow('Recording... Press Q to stop', frame)
                
                # Check for early exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Recording stopped by user")
                    break
                    
        except KeyboardInterrupt:
            print("Recording interrupted by user")
        
        finally:
            # Stop recording
            self.is_recording = False
            
            # Cleanup video
            cap.release()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
            
            # Wait for audio thread to finish
            audio_thread.join()
            
            # Save audio data
            if self.audio_data:
                audio_array = np.concatenate(self.audio_data, axis=0)
                sf.write(str(audio_path), audio_array, self.audio_sample_rate)
                print(f"Audio saved: {audio_path}")
            else:
                print("No audio data recorded")
                audio_path = None
            
            print(f"Video saved: {video_path}")
            print(f"Recorded {frame_count} frames")
            
        return str(audio_path) if audio_path else None, str(video_path)
    
    def capture_audio_only(self, 
                          duration: int = 30, 
                          audio_device: Optional[int] = None) -> str:
        """
        Capture audio only
        
        Args:
            duration: Recording duration in seconds
            audio_device: Audio input device index (None for default)
            
        Returns:
            Audio file path
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"live_audio_{timestamp}.wav"
        audio_path = self.output_dir / audio_filename
        
        print(f"Recording audio for {duration} seconds...")
        print("Press Ctrl+C to stop recording early")
        
        try:
            # Record audio
            audio_data = sd.rec(
                int(duration * self.audio_sample_rate),
                samplerate=self.audio_sample_rate,
                channels=self.audio_channels,
                dtype=self.audio_dtype,
                device=audio_device
            )
            sd.wait()  # Wait until recording is finished
            
            # Save audio
            sf.write(str(audio_path), audio_data, self.audio_sample_rate)
            print(f"Audio saved: {audio_path}")
            
        except KeyboardInterrupt:
            print("Recording interrupted by user")
            
        return str(audio_path)
    
    def create_librispeech_structure(self, audio_file: str) -> str:
        """
        Create LibriSpeech-like directory structure for the audio file
        
        Args:
            audio_file: Path to the recorded audio file
            
        Returns:
            Path to created LibriSpeech directory
        """
        audio_path = Path(audio_file)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create LibriSpeech-like structure: speaker_id/chapter_id/
        speaker_id = f"live_user_{timestamp}"
        chapter_id = "session_001"
        
        libri_dir = self.output_dir / "LibriSpeech_Live" / speaker_id / chapter_id
        libri_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy audio file to LibriSpeech structure
        new_audio_path = libri_dir / f"{speaker_id}-{chapter_id}-0001.wav"
        
        # Convert to wav format if needed and copy
        import shutil
        shutil.copy2(audio_file, new_audio_path)
        
        # Create transcript file (placeholder)
        transcript_path = libri_dir / f"{speaker_id}-{chapter_id}.trans.txt"
        with open(transcript_path, 'w') as f:
            f.write(f"{speaker_id}-{chapter_id}-0001 LIVE MEDITATION SESSION AUDIO\\n")
        
        print(f"LibriSpeech structure created: {libri_dir}")
        return str(libri_dir.parent.parent)  # Return LibriSpeech_Live directory


def main():
    parser = argparse.ArgumentParser(description="Live Audio/Video Capture for Meditation Pipeline")
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
    
    args = parser.parse_args()
    
    capture = LiveCapture(args.output_dir)
    
    if args.list_devices:
        capture.list_audio_devices()
        print()
        capture.list_video_devices()
        return
    
    try:
        if args.mode == "audio":
            audio_file = capture.capture_audio_only(args.duration, args.audio_device)
            if args.create_librispeech and audio_file:
                capture.create_librispeech_structure(audio_file)
                
        elif args.mode == "video":
            # Video only mode would need separate implementation
            print("Video-only mode not implemented yet")
            
        elif args.mode == "both":
            audio_file, video_file = capture.capture_audio_video(
                args.duration, args.audio_device, args.video_device
            )
            if args.create_librispeech and audio_file:
                capture.create_librispeech_structure(audio_file)
            
        print("\\nCapture complete! Files are ready for the meditation pipeline.")
        print(f"Audio files location: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during capture: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
