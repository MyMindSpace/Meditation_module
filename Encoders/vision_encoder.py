import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False


@dataclass
class PoseStats:
    had_detections: bool
    frames_with_pose: int
    total_frames: int
    embedding: np.ndarray


class VisionEncoder:
    """
    Vision Encoder (VE)

    Input: Preprocessed video frames stored as numpy arrays (e.g., *_frames.npy)

    Processing:
      - CNN (ResNet50/EfficientNet) to extract high-level visual features
      - Keypoint detection (MediaPipe Pose) for posture analysis
      - Pose embedding from normalized distances/angles aggregated over frames

    Output: Numerical embeddings for posture and visual features
    """

    def __init__(self,
                 cnn_backbone: str = "resnet50",
                 frame_size: int = 224,
                 max_frames: int = 32,
                 device: Optional[str] = None):
        self.cnn_backbone = cnn_backbone
        self.frame_size = int(frame_size)
        self.max_frames = int(max_frames)
        self.device = self._resolve_device(device) if TORCH_AVAILABLE else None

        self.cnn: Optional[nn.Module] = None
        self.cnn_out_dim: int = 0
        self.preprocess = None

        if TORCH_AVAILABLE:
            self._init_cnn()

        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(static_image_mode=True,
                                          model_complexity=1,
                                          enable_segmentation=False,
                                          min_detection_confidence=0.5)
        else:
            self.mp_pose = None
            self.pose = None

    # ---------- Device & CNN ----------
    def _resolve_device(self, user_device: Optional[str]) -> torch.device:
        if user_device:
            return torch.device(user_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _init_cnn(self) -> None:
        # Build feature extractor by removing final classification layer
        if self.cnn_backbone.lower() == "resnet50":
            try:
                # Torchvision 0.13+
                weights = models.ResNet50_Weights.DEFAULT  # type: ignore[attr-defined]
                model = models.resnet50(weights=weights)
            except Exception:
                model = models.resnet50(pretrained=True)
            # Remove classifier
            modules = list(model.children())[:-1]  # upto avgpool
            self.cnn = nn.Sequential(*modules).to(self.device).eval()
            self.cnn_out_dim = 2048
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.frame_size, self.frame_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            # Default to resnet50 if unknown
            self._init_cnn_fallback()

    def _init_cnn_fallback(self) -> None:
        # Same as resnet50
        try:
            weights = models.ResNet50_Weights.DEFAULT  # type: ignore[attr-defined]
            model = models.resnet50(weights=weights)
        except Exception:
            model = models.resnet50(pretrained=True)
        modules = list(model.children())[:-1]
        self.cnn = nn.Sequential(*modules).to(self.device).eval()
        self.cnn_out_dim = 2048
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.frame_size, self.frame_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ---------- Loading Frames ----------
    def load_frames(self, path: Path) -> np.ndarray:
        arr = np.load(str(path))
        arr = np.array(arr, dtype=np.float32, copy=False)
        # Expect [T, H, W, C]. Try to adapt common variants.
        if arr.ndim == 3:
            # [T, H, W] -> add channel
            arr = arr[..., None]
        elif arr.ndim == 4:
            pass
        else:
            # Unsupported, try to flatten last dim to channels
            arr = arr.reshape((-1,) + arr.shape[-3:])

        T, *rest = arr.shape
        # Possible variants: [H, W, C, T] or [C, T, H, W] or [T, C, H, W]
        if T <= 4 and arr.shape[-1] > 4:
            # Likely [C, H, W, T] or [H, W, C, T]
            arr = np.moveaxis(arr, -1, 0)  # move T to front
        if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            # Might be [T, C, H, W] -> [T, H, W, C]
            arr = np.moveaxis(arr, 1, -1)

        # Ensure channels last and 3 channels
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]

        # Clip to [0,1] if it looks like 0..255
        if arr.max() > 1.5:
            arr = arr / 255.0

        # Replace NaNs/Infs
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _sample_indices(self, total: int, max_frames: int) -> np.ndarray:
        if total <= max_frames:
            return np.arange(total, dtype=np.int64)
        # Uniform sampling
        return np.linspace(0, total - 1, num=max_frames).astype(np.int64)

    # ---------- CNN Feature Extraction ----------
    def extract_cnn_features(self, frames: np.ndarray) -> np.ndarray:
        """
        Returns a single vector by mean pooling per-frame CNN features.
        Fallback to color statistics if PyTorch is unavailable.
        """
        T = frames.shape[0]
        indices = self._sample_indices(T, self.max_frames)
        sampled = frames[indices]  # [t, H, W, 3]

        if not TORCH_AVAILABLE or self.cnn is None or self.preprocess is None:
            # Fallback: color stats + luminance histogram
            # Convert to grayscale luminance
            gray = 0.299 * sampled[..., 0] + 0.587 * sampled[..., 1] + 0.114 * sampled[..., 2]
            hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0), density=True)
            means = np.mean(sampled, axis=(0, 1, 2))  # RGB means
            stds = np.std(sampled, axis=(0, 1, 2))
            vec = np.concatenate([means, stds, hist.astype(np.float32)], axis=0)
            return vec.astype(np.float32)

        # Torch path
        with torch.no_grad():
            batch_tensors: List[torch.Tensor] = []
            for img in sampled:
                # img: H W 3 in [0,1]
                tensor = self.preprocess(img).unsqueeze(0)  # 1x3xHxW
                batch_tensors.append(tensor)
            batch = torch.cat(batch_tensors, dim=0).to(self.device)
            feats = self.cnn(batch)  # [t, 2048, 1, 1]
            feats = feats.reshape(feats.size(0), feats.size(1))  # [t, 2048]
            pooled = feats.mean(dim=0)  # [2048]
            return pooled.detach().cpu().numpy().astype(np.float32)

    # ---------- Pose Detection & Embedding ----------
    def _pose_keypoints_frame(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        if not MEDIAPIPE_AVAILABLE or self.pose is None:
            return None
        # frame_rgb: HxWx3 in [0,1]
        # Mediapipe expects uint8 0..255 RGB
        img = np.clip(frame_rgb * 255.0, 0, 255).astype(np.uint8)
        result = self.pose.process(img)
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks.landmark
        # 33 landmarks with x,y,visibility
        points = np.array([[p.x, p.y, p.z if hasattr(p, 'z') else 0.0, p.visibility] for p in lm], dtype=np.float32)
        return points  # [33, 4]

    def extract_pose_embedding(self, frames: np.ndarray) -> PoseStats:
        if not MEDIAPIPE_AVAILABLE or self.pose is None:
            return PoseStats(False, 0, int(frames.shape[0]), np.zeros(32, dtype=np.float32))

        T = frames.shape[0]
        indices = self._sample_indices(T, min(self.max_frames, 48))
        sampled = frames[indices]

        keypoints_list: List[np.ndarray] = []
        for img in sampled:
            pts = self._pose_keypoints_frame(img)
            if pts is not None:
                keypoints_list.append(pts)

        if not keypoints_list:
            return PoseStats(False, 0, int(T), np.zeros(32, dtype=np.float32))

        # Build per-frame normalized distances/angles
        def torso_scale(pts: np.ndarray) -> float:
            # Use shoulder width + hip width average as scale
            # Indices from MediaPipe Pose: 11 L-shoulder, 12 R-shoulder, 23 L-hip, 24 R-hip
            def dist(a: int, b: int) -> float:
                return float(np.linalg.norm(pts[a, :2] - pts[b, :2]))
            s = dist(11, 12)
            h = dist(23, 24)
            return max(0.5 * (s + h), 1e-6)

        def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            # angle at b for a-b-c in radians
            v1 = a - b
            v2 = c - b
            n1 = np.linalg.norm(v1) + 1e-8
            n2 = np.linalg.norm(v2) + 1e-8
            cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            return float(np.arccos(cosang))

        feats_accum: List[np.ndarray] = []
        for pts in keypoints_list:
            s = torso_scale(pts)
            # Distances (normalized): wrists to shoulders, ankles to hips
            d_ls_lw = np.linalg.norm(pts[11, :2] - pts[15, :2]) / s  # L shoulder - L wrist
            d_rs_rw = np.linalg.norm(pts[12, :2] - pts[16, :2]) / s  # R shoulder - R wrist
            d_lh_la = np.linalg.norm(pts[23, :2] - pts[27, :2]) / s  # L hip - L ankle
            d_rh_ra = np.linalg.norm(pts[24, :2] - pts[28, :2]) / s  # R hip - R ankle

            # Angles (radians): elbows, knees
            th_lelbow = angle(pts[11, :2], pts[13, :2], pts[15, :2])
            th_relbow = angle(pts[12, :2], pts[14, :2], pts[16, :2])
            th_lknee = angle(pts[23, :2], pts[25, :2], pts[27, :2])
            th_rknee = angle(pts[24, :2], pts[26, :2], pts[28, :2])

            # Shoulder slope (posture forward/lean proxy)
            slope_sh = np.arctan2(pts[12, 1] - pts[11, 1], pts[12, 0] - pts[11, 0])

            # Visibility stats for robustness
            vis_mean = float(np.mean(pts[:, 3]))
            vis_min = float(np.min(pts[:, 3]))

            f = np.array([
                d_ls_lw, d_rs_rw, d_lh_la, d_rh_ra,
                th_lelbow, th_relbow, th_lknee, th_rknee,
                slope_sh,
                vis_mean, vis_min,
            ], dtype=np.float32)
            feats_accum.append(f)

        F = np.stack(feats_accum, axis=0)
        mean = np.mean(F, axis=0)
        std = np.std(F, axis=0)
        # Simple stability metric: std magnitude
        stability = float(np.linalg.norm(std))
        emb = np.concatenate([mean, std, np.array([stability], dtype=np.float32)], axis=0)
        return PoseStats(True, int(F.shape[0]), int(frames.shape[0]), emb.astype(np.float32))

    # ---------- Public API ----------
    def process_video_frames_file(self, file_path: Path) -> Dict[str, Any]:
        frames = self.load_frames(file_path)  # [T, H, W, 3] in [0,1]
        cnn_vec = self.extract_cnn_features(frames)
        pose = self.extract_pose_embedding(frames)

        # Build embeddings
        visual_vec = cnn_vec
        pose_vec = pose.embedding

        combined = np.concatenate([visual_vec.astype(np.float32), pose_vec.astype(np.float32)], axis=0)

        return {
            "file": str(file_path).replace("\\", "/"),
            "features": {
                "pose": {
                    "had_detections": pose.had_detections,
                    "frames_with_pose": pose.frames_with_pose,
                    "total_frames": pose.total_frames,
                },
                "meta": {
                    "torch_available": TORCH_AVAILABLE,
                    "mediapipe_available": MEDIAPIPE_AVAILABLE,
                    "cnn_backbone": self.cnn_backbone if TORCH_AVAILABLE else "fallback_stats",
                },
            },
            "embeddings": {
                "visual": visual_vec.tolist(),
                "pose": pose_vec.tolist(),
                "combined": combined.tolist(),
            },
            "embedding_dimensions": {
                "visual": int(visual_vec.shape[0]),
                "pose": int(pose_vec.shape[0]),
                "combined": int(combined.shape[0]),
            },
        }

    # Add to your VisionEncoder class
    def detect_meditation_posture(self, frame):
        """Enhanced posture detection for meditation"""
        
        # Current face landmarks + body pose detection
        pose_landmarks = self.mp_pose.process(frame)
        face_landmarks = self.mp_face_mesh.process(frame)
        
        posture_metrics = {
            'head_alignment': self._calculate_head_alignment(face_landmarks),
            'shoulder_level': self._check_shoulder_alignment(pose_landmarks),
            'spine_straightness': self._assess_spine_curvature(pose_landmarks),
            'sitting_stability': self._check_base_stability(pose_landmarks),
            'hand_position': self._verify_hand_placement(pose_landmarks)
        }
        
        return self._compute_overall_posture_score(posture_metrics)



# ---------- CLI Utilities ----------
def find_frame_arrays(root: Path, pattern: str = "*_frames.npy") -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob(pattern))


def save_json(output_path: Path, data: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Vision Encoder (VE)")
    parser.add_argument("--input-dir", type=str, default="preprocess_output/video_frames",
                        help="Directory containing *_frames.npy files")
    parser.add_argument("--glob", type=str, default="*_frames.npy",
                        help="Glob pattern to match frame arrays recursively")
    parser.add_argument("--output", type=str, default="preprocess_output/vision_encoded.json",
                        help="Path to save encoded JSON results")
    parser.add_argument("--cnn-backbone", type=str, default="resnet50",
                        help="CNN backbone name (resnet50)")
    parser.add_argument("--frame-size", type=int, default=224,
                        help="Resize frames for CNN input")
    parser.add_argument("--max-frames", type=int, default=32,
                        help="Max frames to sample per video")
    args = parser.parse_args()

    encoder = VisionEncoder(cnn_backbone=args.cnn_backbone,
                            frame_size=args.frame_size,
                            max_frames=args.max_frames)

    input_root = Path(args.input_dir)
    files = find_frame_arrays(input_root, args.glob)
    if not files:
        raise FileNotFoundError(f"No files matched in {input_root} with pattern {args.glob}")

    results: List[Dict[str, Any]] = []
    for i, f in enumerate(files):
        try:
            result = encoder.process_video_frames_file(f)
            results.append(result)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    out_path = Path(args.output)
    save_json(out_path, results)
    print(f"Saved {len(results)} records to {out_path}")

    if results:
        dims = results[0]["embedding_dimensions"]
        print("\nEmbedding dimensions:")
        for k, v in dims.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


