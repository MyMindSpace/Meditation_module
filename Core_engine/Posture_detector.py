import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False


def _squash(x: float, k: float) -> float:
    x = max(float(x), 0.0)
    return 1.0 - np.exp(-x / max(k, 1e-6))


class LSTMPostureHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)  # [B,1] in [0,1]


class PostureDetector:
    """
    Posture Detection Model (PDM)

    Input: Pose embeddings and visual features from the Vision Encoder.
    Processing:
      - Heuristic posture quality scoring from pose embedding statistics
      - Optional LSTM head over sequences of pose embeddings
      - Optional keypoint re-extraction from frames for real-time tracking
    Output:
      - Posture quality score [0,1]
      - (Optional) keypoint locations for sampled frames
    """

    def __init__(self, use_lstm: bool = False, lstm_checkpoint: Optional[str] = None):
        self.use_lstm = bool(use_lstm and TORCH_AVAILABLE)
        self.pose_emb_dim = 23  # from VisionEncoder: 11 mean + 11 std + 1 stability

        if self.use_lstm:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = LSTMPostureHead(input_dim=self.pose_emb_dim).to(self.device)
            if lstm_checkpoint:
                state = torch.load(lstm_checkpoint, map_location=self.device)
                # Support both full state_dict or wrapped
                state_dict = state.get("state_dict", state)
                self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
        else:
            self.device = None
            self.model = None

        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(static_image_mode=True,
                                          model_complexity=1,
                                          enable_segmentation=False,
                                          min_detection_confidence=0.5)
        else:
            self.mp_pose = None
            self.pose = None

    # ---------- Heuristic scoring ----------
    def _score_from_pose_embedding(self, pose_emb: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Expect pose_emb layout from VisionEncoder:
          mean features (11): [d_ls_lw, d_rs_rw, d_lh_la, d_rh_ra, th_lelbow, th_relbow, th_lknee, th_rknee, slope_sh, vis_mean, vis_min]
          std features  (11) in same order
          stability (1): L2 norm of std vector
        """
        v = np.array(pose_emb, dtype=np.float32).reshape(-1)
        if v.size < self.pose_emb_dim:
            # Pad if shorter
            pad = np.zeros(self.pose_emb_dim - v.size, dtype=np.float32)
            v = np.concatenate([v, pad], axis=0)

        mean = v[:11]
        std = v[11:22]
        stability = float(v[22])

        # Targets: elbows/knees close to ~pi/2 (1.57 rad), shoulder slope ~0
        target_angle = 1.57
        angle_scale = 0.5
        th_lelbow, th_relbow, th_lknee, th_rknee = mean[4], mean[5], mean[6], mean[7]
        angs = np.array([th_lelbow, th_relbow, th_lknee, th_rknee], dtype=np.float32)
        angle_err = np.abs(angs - target_angle)
        angle_score = np.clip(1.0 - (angle_err / angle_scale), 0.0, 1.0).mean()

        slope = float(mean[8])  # radians, 0 is level shoulders
        slope_score = float(np.exp(-abs(slope) / 0.3))  # sharp drop after ~17 degrees

        stability_score = 1.0 - _squash(stability, k=0.5)

        # Visibility reduces confidence
        vis_mean = float(mean[9])
        visibility_factor = np.clip((vis_mean - 0.3) / 0.5, 0.0, 1.0)

        score = 0.4 * angle_score + 0.3 * slope_score + 0.3 * stability_score
        score *= visibility_factor
        score = float(np.clip(score, 0.0, 1.0))

        dbg = {
            "angle_score": float(angle_score),
            "slope_score": float(slope_score),
            "stability_score": float(stability_score),
            "visibility_factor": float(visibility_factor),
        }
        return score, dbg

    # ---------- Optional sequence scoring via LSTM ----------
    def score_sequence(self, pose_embeddings: List[np.ndarray]) -> float:
        if not self.use_lstm or self.model is None:
            # Heuristic average
            if not pose_embeddings:
                return 0.0
            scores = [self._score_from_pose_embedding(e)[0] for e in pose_embeddings]
            return float(np.mean(scores))

        X = np.stack([np.pad(e.reshape(-1), (0, max(0, self.pose_emb_dim - e.size)), mode='constant')[:self.pose_emb_dim]
                       for e in pose_embeddings], axis=0)
        with torch.no_grad():
            t = torch.from_numpy(X).float().unsqueeze(0).to(self.device)  # [1, T, D]
            y = self.model(t)  # [1,1]
            return float(y.squeeze(0).squeeze(0).detach().cpu().item())

    # ---------- Keypoint re-extraction (optional) ----------
    def reextract_keypoints(self, frames_path: Path, sample_every: int = 5) -> List[List[List[float]]]:
        if not MEDIAPIPE_AVAILABLE or self.pose is None or not frames_path.exists():
            return []
        arr = np.load(str(frames_path))
        arr = np.array(arr, dtype=np.float32, copy=False)
        # Normalize to [T, H, W, C]
        if arr.ndim == 3:
            arr = arr[..., None]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.max() > 1.5:
            arr = arr / 255.0

        keypoints: List[List[List[float]]] = []
        for i in range(0, arr.shape[0], max(1, int(sample_every))):
            img = np.clip(arr[i] * 255.0, 0, 255).astype(np.uint8)
            result = self.pose.process(img)
            if not result.pose_landmarks:
                keypoints.append([])
                continue
            lm = result.pose_landmarks.landmark
            pts = [[float(p.x), float(p.y), float(getattr(p, 'z', 0.0)), float(p.visibility)] for p in lm]
            keypoints.append(pts)
        return keypoints

    # ---------- Public API ----------
    def process_record(self, record: Dict[str, Any], reextract: bool = False, sample_every: int = 5) -> Dict[str, Any]:
        pose_vec = np.array(record.get("embeddings", {}).get("pose", []), dtype=np.float32)
        score, dbg = self._score_from_pose_embedding(pose_vec)

        out: Dict[str, Any] = {
            "file": record.get("file"),
            "posture_score": score,
            "debug": dbg,
        }

        if reextract and record.get("file"):
            frames_path = Path(record["file"])
            if frames_path.exists():
                kps = self.reextract_keypoints(frames_path, sample_every=sample_every)
                out["keypoints"] = kps
        return out


# ---------- IO ----------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Add these methods to the PostureDetector class in Core_engine/Posture_detector.py
# Around line 200+ (after existing methods)

def _fallback_posture_detection(self, frame):
    """Fallback when MediaPipe fails"""
    # Simple motion-based posture detection
    if hasattr(self, 'prev_frame'):
        # Calculate frame difference
        diff = cv2.absdiff(frame, self.prev_frame)
        movement = np.mean(diff)
        
        # High movement = poor posture stability
        stability_score = max(0.0, 1.0 - movement / 50.0)
        
        # Simple brightness check for head position
        top_region = frame[:frame.shape[0]//3, :]
        brightness = np.mean(top_region)
        head_score = 0.7 if 50 < brightness < 200 else 0.3
        
        combined_score = (stability_score + head_score) / 2.0
        self.prev_frame = frame.copy()
        
        return {
            'posture_score': combined_score,
            'debug': {
                'method': 'fallback',
                'movement': float(movement),
                'brightness': float(brightness)
            }
        }
    else:
        self.prev_frame = frame.copy()
        return {'posture_score': 0.5, 'debug': {'method': 'fallback_init'}}

def process_live_frame(self, frame):
    """Process live camera frame with fallback handling"""
    try:
        # Try MediaPipe first
        if self.pose is not None:
            # Process through vision pipeline
            # This should integrate with your vision encoder
            pass
        else:
            # Use fallback method
            return self._fallback_posture_detection(frame)
    except Exception as e:
        print(f"Live frame processing error: {e}")
        return self._fallback_posture_detection(frame)



# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Posture Detection Model (PDM)")
    parser.add_argument("--vision-json", type=str, default="preprocess_output/vision_encoded.json",
                        help="Path to vision-encoded JSON file")
    parser.add_argument("--output", type=str, default="preprocess_output/posture_scores.json",
                        help="Where to save posture scores")
    parser.add_argument("--use-lstm", action="store_true", help="Use LSTM head (requires checkpoint)")
    parser.add_argument("--lstm-checkpoint", type=str, default="",
                        help="Path to LSTM checkpoint (.pt/.pth)")
    parser.add_argument("--reextract-keypoints", action="store_true",
                        help="Re-extract keypoints from frames for sampled frames")
    parser.add_argument("--sample-every", type=int, default=5,
                        help="Sample every N frames when re-extracting keypoints")
    args = parser.parse_args()

    det = PostureDetector(use_lstm=args.use_lstm, lstm_checkpoint=(args.lstm_checkpoint or None))

    vision_path = Path(args.vision_json)
    if not vision_path.exists():
        raise FileNotFoundError(f"Vision JSON not found: {vision_path}")
    data = load_json(vision_path)

    if not isinstance(data, list):
        raise ValueError("Expected vision JSON to be a list of records")

    outputs: List[Dict[str, Any]] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        try:
            out = det.process_record(rec, reextract=args.reextract_keypoints, sample_every=max(1, args.sample_every))
            outputs.append(out)
        except Exception as e:
            print(f"Error processing record: {e}")

    out_path = Path(args.output)
    save_json(out_path, outputs)
    print(f"Saved {len(outputs)} posture records to {out_path}")


if __name__ == "__main__":
    main()


