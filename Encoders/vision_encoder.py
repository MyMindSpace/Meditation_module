# Encoders/vision_encoder.py
"""
Vision Encoder (VE)

Processes video frames to extract:
- Visual features using ResNet50/EfficientNet
- Pose keypoints using MediaPipe
- Posture embeddings for PDM
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional imports with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class VisionEncoder:
    """
    Vision Encoder for meditation module
    Extracts visual features and pose embeddings from video frames
    """
    
    def __init__(self, 
                 use_torch: bool = True,
                 model_name: str = "resnet50",
                 pose_model_complexity: int = 1):
        
        self.use_torch = use_torch and TORCH_AVAILABLE
        self.model_name = model_name
        
        # Initialize visual feature extractor
        if self.use_torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._init_visual_model()
            
        # Initialize pose detection
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=pose_model_complexity,
                enable_segmentation=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        else:
            self.mp_pose = None
            self.pose = None
            
        # Transform for visual features
        if self.use_torch:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _init_visual_model(self):
        """Initialize the visual feature extraction model"""
        if self.model_name == "resnet50":
            self.visual_model = models.resnet50(pretrained=True)
            self.visual_model.fc = nn.Identity()  # Remove final layer
            self.feature_dim = 2048
        elif self.model_name == "efficientnet":
            self.visual_model = models.efficientnet_b0(pretrained=True)
            self.visual_model.classifier = nn.Identity()
            self.feature_dim = 1280
        else:
            # Fallback to ResNet50
            self.visual_model = models.resnet50(pretrained=True)
            self.visual_model.fc = nn.Identity()
            self.feature_dim = 2048
            
        self.visual_model.to(self.device)
        self.visual_model.eval()
    
    def _extract_visual_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract deep visual features from frame"""
        if not self.use_torch:
            # Simple fallback: histogram features
            return self._extract_histogram_features(frame)
            
        # Convert BGR to RGB if needed
        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
            
        # Transform and extract features
        with torch.no_grad():
            tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            features = self.visual_model(tensor)
            return features.cpu().numpy().flatten()
    
    def _extract_histogram_features(self, frame: np.ndarray) -> np.ndarray:
        """Fallback histogram-based features"""
        # Convert to different color spaces and compute histograms
        features = []
        
        # RGB histograms
        for i in range(3):
            hist = cv2.calcHist([frame], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
            
        # HSV histograms  
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
            
        return np.array(features, dtype=np.float32)
    
    def _extract_pose_keypoints(self, frame: np.ndarray) -> Tuple[List[List[float]], bool]:
        """Extract pose keypoints using MediaPipe"""
        if not MEDIAPIPE_AVAILABLE or self.pose is None:
            return [], False
            
        # Convert BGR to RGB for MediaPipe
        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
            
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return [], False
            
        # Extract keypoints with confidence
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([
                float(landmark.x),
                float(landmark.y), 
                float(getattr(landmark, 'z', 0.0)),
                float(landmark.visibility)
            ])
            
        return keypoints, True
    
    def _compute_pose_embedding(self, keypoints: List[List[float]]) -> np.ndarray:
        """Compute pose embedding from keypoints"""
        if not keypoints or len(keypoints) < 33:  # MediaPipe has 33 landmarks
            return np.zeros(23, dtype=np.float32)  # Expected pose embedding size
            
        kp = np.array(keypoints)[:33, :3]  # Use x, y, z coordinates
        visibility = np.array(keypoints)[:33, 3]
        
        # Compute geometric features
        features = []
        
        # Distance features (limb lengths)
        # Left shoulder to left wrist
        d_ls_lw = float(np.linalg.norm(kp[11] - kp[15]))
        # Right shoulder to right wrist  
        d_rs_rw = float(np.linalg.norm(kp[12] - kp[16]))
        # Left hip to left ankle
        d_lh_la = float(np.linalg.norm(kp[23] - kp[27]))
        # Right hip to right ankle
        d_rh_ra = float(np.linalg.norm(kp[24] - kp[28]))
        
        features.extend([d_ls_lw, d_rs_rw, d_lh_la, d_rh_ra])
        
        # Angle features (joint angles)
        def angle_between_vectors(v1, v2):
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        # Left elbow angle
        if len(kp) > 15:
            v1 = kp[13] - kp[11]  # shoulder to elbow
            v2 = kp[15] - kp[13]  # elbow to wrist
            th_lelbow = angle_between_vectors(v1, v2)
        else:
            th_lelbow = 0.0
            
        # Right elbow angle
        if len(kp) > 16:
            v1 = kp[14] - kp[12]  # shoulder to elbow
            v2 = kp[16] - kp[14]  # elbow to wrist  
            th_relbow = angle_between_vectors(v1, v2)
        else:
            th_relbow = 0.0
            
        # Left knee angle
        if len(kp) > 27:
            v1 = kp[25] - kp[23]  # hip to knee
            v2 = kp[27] - kp[25]  # knee to ankle
            th_lknee = angle_between_vectors(v1, v2)
        else:
            th_lknee = 0.0
            
        # Right knee angle  
        if len(kp) > 28:
            v1 = kp[26] - kp[24]  # hip to knee
            v2 = kp[28] - kp[26]  # knee to ankle
            th_rknee = angle_between_vectors(v1, v2)
        else:
            th_rknee = 0.0
            
        features.extend([th_lelbow, th_relbow, th_lknee, th_rknee])
        
        # Shoulder slope
        if len(kp) > 12:
            shoulder_vec = kp[12] - kp[11]  # right shoulder - left shoulder
            slope_sh = float(np.arctan2(shoulder_vec[1], shoulder_vec[0]))
        else:
            slope_sh = 0.0
            
        features.append(slope_sh)
        
        # Visibility statistics
        vis_mean = float(np.mean(visibility))
        vis_min = float(np.min(visibility))
        features.extend([vis_mean, vis_min])
        
        # Convert to numpy array (11 features so far)
        mean_features = np.array(features, dtype=np.float32)
        
        # Compute same features for std (use small values as placeholder)
        std_features = np.abs(mean_features) * 0.1 + 0.01
        
        # Stability measure (L2 norm of std)
        stability = float(np.linalg.norm(std_features))
        
        # Combine: 11 mean + 11 std + 1 stability = 23 features
        pose_embedding = np.concatenate([mean_features, std_features, [stability]])
        
        return pose_embedding.astype(np.float32)
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame to extract visual and pose features"""
        if frame is None or frame.size == 0:
            return {
                "visual_features": [],
                "pose_keypoints": [],
                "pose_embedding": [],
                "pose_detected": False
            }
            
        # Extract visual features
        visual_features = self._extract_visual_features(frame)
        
        # Extract pose keypoints and embedding
        keypoints, pose_detected = self._extract_pose_keypoints(frame)
        pose_embedding = self._compute_pose_embedding(keypoints) if pose_detected else np.zeros(23)
        
        return {
            "visual_features": visual_features.tolist() if isinstance(visual_features, np.ndarray) else visual_features,
            "pose_keypoints": keypoints,
            "pose_embedding": pose_embedding.tolist(),
            "pose_detected": pose_detected
        }
    
    def process_video_file(self, video_path: str, sample_rate: int = 5) -> List[Dict[str, Any]]:
        """Process entire video file, sampling every N frames"""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for video processing")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        results = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % sample_rate == 0:
                    result = self.process_frame(frame)
                    result["frame_number"] = frame_count
                    result["timestamp"] = frame_count / cap.get(cv2.CAP_PROP_FPS)
                    results.append(result)
                    
                frame_count += 1
                
        finally:
            cap.release()
            
        return results
    
    def process_frames_array(self, frames: np.ndarray) -> List[Dict[str, Any]]:
        """Process array of frames"""
        if frames.ndim != 4:  # [T, H, W, C]
            raise ValueError("Expected 4D array [T, H, W, C]")
            
        results = []
        for i, frame in enumerate(frames):
            # Ensure frame is in correct format
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
                
            result = self.process_frame(frame)
            result["frame_number"] = i
            results.append(result)
            
        return results


def load_frames_from_npy(file_path: str) -> np.ndarray:
    """Load frames from .npy file"""
    frames = np.load(file_path)
    
    # Normalize and ensure correct format
    if frames.max() > 1.5:
        frames = frames.astype(np.float32) / 255.0
    
    # Ensure 4D: [T, H, W, C]
    if frames.ndim == 3:
        frames = frames[..., None]  # Add channel dim
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)  # Convert to RGB
        
    return frames


def main():
    """CLI interface for vision encoder"""
    parser = argparse.ArgumentParser(description="Vision Encoder for meditation module")
    parser.add_argument("--input", type=str, required=True,
                       help="Input video file or .npy frames file")
    parser.add_argument("--output", type=str, default="preprocess_output/vision_encoded.json",
                       help="Output JSON file")
    parser.add_argument("--sample-rate", type=int, default=5,
                       help="Sample every N frames (for video files)")
    parser.add_argument("--model", type=str, default="resnet50",
                       choices=["resnet50", "efficientnet"],
                       help="Visual feature extraction model")
    parser.add_argument("--no-torch", action="store_true",
                       help="Disable PyTorch and use fallback features")
    
    args = parser.parse_args()
    
    # Initialize encoder
    encoder = VisionEncoder(
        use_torch=not args.no_torch,
        model_name=args.model
    )
    
    # Process input
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    if input_path.suffix == ".npy":
        # Process frames from numpy array
        frames = load_frames_from_npy(str(input_path))
        results = encoder.process_frames_array(frames)
    else:
        # Process video file
        results = encoder.process_video_file(str(input_path), args.sample_rate)
    
    # Prepare output with aggregated embeddings
    output_data = []
    if results:
        # Aggregate features across all frames
        all_visual = [r["visual_features"] for r in results if r["visual_features"]]
        all_pose = [r["pose_embedding"] for r in results if r["pose_embedding"]]
        
        if all_visual:
            visual_mean = np.mean(all_visual, axis=0).tolist()
            visual_std = np.std(all_visual, axis=0).tolist()
        else:
            visual_mean = visual_std = []
            
        if all_pose:
            pose_mean = np.mean(all_pose, axis=0).tolist() 
            pose_std = np.std(all_pose, axis=0).tolist()
        else:
            pose_mean = pose_std = []
        
        # Create combined embedding
        combined = []
        if visual_mean and pose_mean:
            # Normalize and combine
            visual_norm = np.array(visual_mean) / (np.linalg.norm(visual_mean) + 1e-8)
            pose_norm = np.array(pose_mean) / (np.linalg.norm(pose_mean) + 1e-8)
            combined = np.concatenate([visual_norm * 0.7, pose_norm * 0.3]).tolist()
        elif pose_mean:
            combined = pose_mean
        elif visual_mean:
            combined = visual_mean
            
        output_data.append({
            "file": str(input_path),
            "total_frames": len(results),
            "pose_detection_rate": sum(1 for r in results if r["pose_detected"]) / len(results),
            "embeddings": {
                "visual": visual_mean,
                "visual_std": visual_std,
                "pose": pose_mean,
                "pose_std": pose_std,
                "combined": combined
            },
            "frame_details": results[:10]  # Include first 10 frames for debugging
        })
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"Vision encoding complete. Results saved to {output_path}")
    if results:
        print(f"Processed {len(results)} frames")
        pose_rate = sum(1 for r in results if r["pose_detected"]) / len(results)
        print(f"Pose detection rate: {pose_rate:.2%}")


if __name__ == "__main__":
    main()