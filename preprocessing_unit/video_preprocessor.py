# preprocessing_unit/video_preprocessor.py
"""
Video Preprocessor (VP)

Handles video preprocessing tasks:
- Frame extraction and sampling
- Resizing and normalization
- Quality assessment
- Temporal consistency checks
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional imports with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class VideoPreprocessor:
    """
    Video preprocessor for meditation module
    Handles frame extraction, resizing, and quality assessment
    """
    
    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 target_fps: Optional[float] = None,
                 max_frames: Optional[int] = None,
                 quality_threshold: float = 0.3):
        
        self.target_size = target_size
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.quality_threshold = quality_threshold
        
        # Frame quality metrics thresholds
        self.brightness_range = (30, 225)  # Acceptable brightness range
        self.contrast_threshold = 20  # Minimum contrast
        self.blur_threshold = 100  # Minimum focus score
    
    def load_video(self, video_path: str) -> Tuple[Optional[cv2.VideoCapture], Dict[str, Any]]:
        """Load video and extract metadata"""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for video processing")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, {"error": f"Cannot open video file: {video_path}"}
        
        # Extract video metadata
        metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        }
        
        return cap, metadata
    
    def assess_frame_quality(self, frame: np.ndarray) -> Dict[str, float]:
        """Assess quality metrics for a single frame"""
        if frame is None or frame.size == 0:
            return {
                "brightness": 0.0,
                "contrast": 0.0,
                "sharpness": 0.0,
                "quality_score": 0.0
            }
        
        # Convert to grayscale for analysis
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Brightness (mean intensity)
        brightness = float(np.mean(gray))
        
        # Contrast (standard deviation of intensity)
        contrast = float(np.std(gray))
        
        # Sharpness (Laplacian variance - edge content)
        if CV2_AVAILABLE:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = float(laplacian.var())
        else:
            # Simple gradient-based sharpness
            grad_x = np.diff(gray, axis=1)
            grad_y = np.diff(gray, axis=0)
            sharpness = float(np.mean(grad_x**2) + np.mean(grad_y**2))
        
        # Overall quality score (0-1)
        brightness_score = 1.0 if self.brightness_range[0] <= brightness <= self.brightness_range[1] else 0.5
        contrast_score = min(1.0, contrast / 50.0)  # Normalize contrast
        sharpness_score = min(1.0, sharpness / 500.0)  # Normalize sharpness
        
        quality_score = (brightness_score + contrast_score + sharpness_score) / 3.0
        
        return {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "quality_score": quality_score
        }
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target size"""
        if frame is None or frame.size == 0:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
        
        if CV2_AVAILABLE:
            resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        elif PIL_AVAILABLE:
            # Fallback to PIL
            if len(frame.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(frame)
            
            resized_pil = pil_image.resize(self.target_size, Image.LANCZOS)
            resized = np.array(resized_pil)
            
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        else:
            # Manual resize (basic)
            h, w = frame.shape[:2]
            target_h, target_w = self.target_size
            
            # Simple nearest neighbor resize
            resized = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
            scale_h = h / target_h
            scale_w = w / target_w
            
            for i in range(target_h):
                for j in range(target_w):
                    orig_i = int(i * scale_h)
                    orig_j = int(j * scale_w)
                    orig_i = min(orig_i, h - 1)
                    orig_j = min(orig_j, w - 1)
                    
                    if len(frame.shape) == 3:
                        resized[i, j] = frame[orig_i, orig_j]
                    else:
                        resized[i, j] = [frame[orig_i, orig_j]] * 3
        
        return resized
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame values to [0, 1] range"""
        if frame.dtype == np.uint8:
            return frame.astype(np.float32) / 255.0
        elif frame.dtype == np.uint16:
            return frame.astype(np.float32) / 65535.0
        else:
            # Already float, ensure [0, 1] range
            return np.clip(frame, 0.0, 1.0)
    
    def extract_frames(self, video_path: str, 
                      sample_rate: Optional[int] = None,
                      start_time: float = 0.0,
                      end_time: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract and preprocess frames from video"""
        cap, metadata = self.load_video(video_path)
        
        if cap is None:
            return np.array([]), metadata
        
        try:
            fps = metadata["fps"]
            total_frames = metadata["frame_count"]
            
            # Calculate sampling parameters
            if sample_rate is None:
                if self.target_fps is not None:
                    sample_rate = max(1, int(fps / self.target_fps))
                else:
                    sample_rate = 1  # Use all frames
            
            # Calculate frame range
            start_frame = int(start_time * fps)
            if end_time is not None:
                end_frame = int(end_time * fps)
            else:
                end_frame = total_frames
            
            end_frame = min(end_frame, total_frames)
            
            # Limit number of frames if specified
            if self.max_frames is not None:
                frame_range = end_frame - start_frame
                if frame_range > self.max_frames * sample_rate:
                    sample_rate = max(1, frame_range // self.max_frames)
            
            # Extract frames
            frames = []
            quality_scores = []
            frame_indices = []
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if (current_frame - start_frame) % sample_rate == 0:
                    # Assess quality
                    quality_metrics = self.assess_frame_quality(frame)
                    
                    # Skip low quality frames
                    if quality_metrics["quality_score"] >= self.quality_threshold:
                        # Resize and normalize
                        resized_frame = self.resize_frame(frame)
                        normalized_frame = self.normalize_frame(resized_frame)
                        
                        frames.append(normalized_frame)
                        quality_scores.append(quality_metrics)
                        frame_indices.append(current_frame)
                        
                        # Check if we've reached max frames
                        if self.max_frames is not None and len(frames) >= self.max_frames:
                            break
                
                current_frame += 1
            
            # Convert to numpy array
            if frames:
                frame_array = np.array(frames)  # Shape: [T, H, W, C]
            else:
                frame_array = np.zeros((0, *self.target_size, 3), dtype=np.float32)
            
            # Update metadata
            processing_info = {
                "original_fps": fps,
                "sample_rate": sample_rate,
                "extracted_frames": len(frames),
                "frame_indices": frame_indices,
                "average_quality": float(np.mean([q["quality_score"] for q in quality_scores])) if quality_scores else 0.0,
                "quality_scores": quality_scores,
                "start_frame": start_frame,
                "end_frame": current_frame,
                "target_size": self.target_size,
                "quality_threshold": self.quality_threshold
            }
            
            metadata.update(processing_info)
            
            return frame_array, metadata
            
        finally:
            cap.release()
    
    def detect_scene_changes(self, frames: np.ndarray, threshold: float = 0.3) -> List[int]:
        """Detect scene changes in frame sequence"""
        if frames.shape[0] < 2:
            return []
        
        scene_changes = []
        
        for i in range(1, frames.shape[0]):
            # Calculate frame difference
            diff = np.mean(np.abs(frames[i] - frames[i-1]))
            
            if diff > threshold:
                scene_changes.append(i)
        
        return scene_changes
    
    def temporal_consistency_check(self, frames: np.ndarray) -> Dict[str, float]:
        """Check temporal consistency of frame sequence"""
        if frames.shape[0] < 3:
            return {
                "motion_smoothness": 1.0,
                "brightness_stability": 1.0,
                "overall_consistency": 1.0
            }
        
        # Motion smoothness (frame differences should be relatively stable)
        frame_diffs = []
        for i in range(1, frames.shape[0]):
            diff = np.mean(np.abs(frames[i] - frames[i-1]))
            frame_diffs.append(diff)
        
        motion_smoothness = 1.0 / (1.0 + np.std(frame_diffs)) if frame_diffs else 1.0
        
        # Brightness stability
        brightness_values = [np.mean(frame) for frame in frames]
        brightness_stability = 1.0 / (1.0 + np.std(brightness_values)) if brightness_values else 1.0
        
        # Overall consistency score
        overall_consistency = (motion_smoothness + brightness_stability) / 2.0
        
        return {
            "motion_smoothness": float(motion_smoothness),
            "brightness_stability": float(brightness_stability),
            "overall_consistency": float(overall_consistency)
        }
    
    def process_video_file(self, video_path: str, 
                          output_path: Optional[str] = None) -> Dict[str, Any]:
        """Process complete video file"""
        try:
            # Extract frames
            frames, metadata = self.extract_frames(video_path)
            
            if frames.size == 0:
                return {
                    "file": video_path,
                    "status": "failed",
                    "error": "No frames extracted",
                    "metadata": metadata
                }
            
            # Additional analysis
            scene_changes = self.detect_scene_changes(frames)
            consistency = self.temporal_consistency_check(frames)
            
            # Save frames if output path specified
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, frames)
            
            result = {
                "file": video_path,
                "status": "success",
                "frames_shape": frames.shape,
                "scene_changes": scene_changes,
                "temporal_consistency": consistency,
                "metadata": metadata
            }
            
            if output_path:
                result["output_file"] = str(output_path)
            
            return result
            
        except Exception as e:
            return {
                "file": video_path,
                "status": "error",
                "error": str(e),
                "metadata": {}
            }
    
    def process_frames_array(self, frames: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process array of frames (preprocessing only)"""
        if frames.size == 0:
            return frames, {"status": "empty_input"}
        
        processed_frames = []
        quality_scores = []
        
        for i, frame in enumerate(frames):
            # Assess quality
            quality_metrics = self.assess_frame_quality(frame)
            
            # Skip low quality frames
            if quality_metrics["quality_score"] >= self.quality_threshold:
                # Resize and normalize
                if frame.shape[:2] != self.target_size:
                    resized_frame = self.resize_frame(frame)
                else:
                    resized_frame = frame
                
                normalized_frame = self.normalize_frame(resized_frame)
                processed_frames.append(normalized_frame)
                quality_scores.append(quality_metrics)
        
        if processed_frames:
            result_frames = np.array(processed_frames)
        else:
            result_frames = np.zeros((0, *self.target_size, 3), dtype=np.float32)
        
        # Analysis
        scene_changes = self.detect_scene_changes(result_frames) if len(processed_frames) > 1 else []
        consistency = self.temporal_consistency_check(result_frames)
        
        metadata = {
            "status": "success",
            "original_frames": len(frames),
            "processed_frames": len(processed_frames),
            "average_quality": float(np.mean([q["quality_score"] for q in quality_scores])) if quality_scores else 0.0,
            "scene_changes": scene_changes,
            "temporal_consistency": consistency,
            "target_size": self.target_size
        }
        
        return result_frames, metadata


def main():
    """CLI interface for video preprocessor"""
    parser = argparse.ArgumentParser(description="Video Preprocessor for meditation module")
    parser.add_argument("--input", type=str, required=True,
                       help="Input video file or directory")
    parser.add_argument("--output-dir", type=str, default="preprocess_output/video_frames/",
                       help="Output directory for processed frames")
    parser.add_argument("--metadata-file", type=str, default="preprocess_output/video_metadata.json",
                       help="Output JSON file for metadata")
    parser.add_argument("--target-size", type=int, nargs=2, default=[224, 224],
                       help="Target frame size (height width)")
    parser.add_argument("--sample-rate", type=int, default=5,
                       help="Frame sampling rate (take every Nth frame)")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum number of frames to extract per video")
    parser.add_argument("--quality-threshold", type=float, default=0.3,
                       help="Minimum quality threshold for frame selection")
    parser.add_argument("--target-fps", type=float, default=None,
                       help="Target FPS for frame extraction")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor(
        target_size=tuple(args.target_size),
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        quality_threshold=args.quality_threshold
    )
    
    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    if input_path.is_file():
        # Single video file
        output_file = output_dir / f"{input_path.stem}_frames.npy"
        result = preprocessor.process_video_file(str(input_path), str(output_file))
        results.append(result)
        
    elif input_path.is_dir():
        # Directory of video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(input_path.glob(f"**/*{ext}"))
            video_files.extend(input_path.glob(f"**/*{ext.upper()}"))
        
        for video_file in sorted(video_files):
            output_file = output_dir / f"{video_file.stem}_frames.npy"
            result = preprocessor.process_video_file(str(video_file), str(output_file))
            results.append(result)
            
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    # Save metadata
    metadata_path = Path(args.metadata_file)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Video preprocessing complete. Processed {len(results)} files.")
    print(f"Frames saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Summary statistics
    successful = [r for r in results if r.get("status") == "success"]
    if successful:
        total_frames = sum(r.get("frames_shape", [0])[0] for r in successful)
        avg_quality = np.mean([r["metadata"].get("average_quality", 0) for r in successful])
        
        print(f"\nSummary:")
        print(f"Successful videos: {len(successful)}/{len(results)}")
        print(f"Total frames extracted: {total_frames}")
        print(f"Average frame quality: {avg_quality:.3f}")


if __name__ == "__main__":
    main()