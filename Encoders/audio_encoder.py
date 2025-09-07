import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class VADResult:
    speech_mask: np.ndarray
    speech_ratio: float
    segment_count: int
    mean_segment_length_frames: float


class AudioEncoder:
    """
    Audio Encoder (AE)

    Input: Preprocessed audio features (e.g., MFCC numpy arrays).

    Processing:
      - Mel-frequency analysis: compute MFCC stats and deltas to capture tonal qualities
      - Voice Activity Detection (VAD): detect speech frames from energy proxy
      - Emotion Recognition (heuristic): infer calm/neutral/stressed from prosody proxies

    Output: Numerical embeddings for audio content and emotion.
    """

    def __init__(self, target_mfcc_dim: int = 20, vad_smooth_window: int = 5):
        self.target_mfcc_dim = int(target_mfcc_dim)
        self.vad_smooth_window = max(int(vad_smooth_window), 1)

    # ---------- IO ----------
    def load_mfcc(self, path: Path) -> np.ndarray:
        """Load MFCC numpy array and normalize orientation to [frames, n_mfcc]."""
        arr = np.load(str(path))
        arr = np.array(arr, dtype=np.float32, copy=False)
        if arr.ndim == 1:
            # Single vector -> make it 1xN
            arr = arr[None, :]
        if arr.shape[0] < arr.shape[1]:
            # Heuristic: assume [frames, n_mfcc] already
            frames, coeffs = arr.shape
        else:
            # If rows >= cols, try transposing when it seems [n_mfcc, frames]
            if arr.shape[0] <= 60 and arr.shape[1] > arr.shape[0]:
                arr = arr.T
        # Ensure finite
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    # ---------- Feature Processing ----------
    def _pad_or_trim_mfcc(self, mfcc: np.ndarray) -> np.ndarray:
        """
        Ensure MFCC has self.target_mfcc_dim coefficients by trimming or zero-padding coefficients.
        Input shape: [frames, n_mfcc]. Output: [frames, target_mfcc_dim]
        """
        frames, n_mfcc = mfcc.shape
        if n_mfcc == self.target_mfcc_dim:
            return mfcc
        if n_mfcc > self.target_mfcc_dim:
            return mfcc[:, : self.target_mfcc_dim]
        # Pad
        pad_width = self.target_mfcc_dim - n_mfcc
        pad_block = np.zeros((frames, pad_width), dtype=mfcc.dtype)
        return np.concatenate([mfcc, pad_block], axis=1)

    def _compute_deltas(self, x: np.ndarray) -> np.ndarray:
        """Compute simple deltas along time axis with edge padding."""
        if x.shape[0] < 2:
            return np.zeros_like(x)
        dx = np.diff(x, axis=0)
        # Pad first row to keep same frames
        dx = np.vstack([dx[0:1, :], dx])
        return dx

    def _energy_proxy(self, mfcc: np.ndarray) -> np.ndarray:
        """
        Use L2 norm across coefficients as frame-level energy proxy.
        Shape: [frames]
        """
        energy = np.linalg.norm(mfcc, axis=1)
        return energy.astype(np.float32)

    def _smooth_boolean(self, mask: np.ndarray, window: int) -> np.ndarray:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smoothed = np.convolve(mask.astype(np.float32), kernel, mode="same")
        return (smoothed >= 0.5).astype(np.bool_)

    def run_vad(self, mfcc: np.ndarray) -> VADResult:
        energy = self._energy_proxy(mfcc)
        if energy.size == 0:
            return VADResult(
                speech_mask=np.zeros((0,), dtype=np.bool_),
                speech_ratio=0.0,
                segment_count=0,
                mean_segment_length_frames=0.0,
            )

        median = float(np.median(energy))
        std = float(np.std(energy))
        # Adaptive threshold: median + 0.5*std (fallback to 1.2*median)
        thr = median + 0.5 * std
        thr = max(thr, 1.2 * median)
        raw_mask = energy >= thr
        mask = self._smooth_boolean(raw_mask, self.vad_smooth_window)

        # Segment stats
        if mask.size == 0:
            speech_ratio = 0.0
            segment_count = 0
            mean_len = 0.0
        else:
            speech_ratio = float(np.mean(mask.astype(np.float32)))
            # Run-length encode
            changes = np.diff(mask.astype(np.int32), prepend=0)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0] - 1
            if mask[-1]:
                ends = np.append(ends, mask.size - 1)
            if len(starts) == 0:
                segment_count = 0
                mean_len = 0.0
            else:
                segment_lengths = (ends - starts + 1).astype(np.float32)
                segment_count = int(segment_lengths.size)
                mean_len = float(np.mean(segment_lengths))

        return VADResult(mask, speech_ratio, segment_count, mean_len)

    def mel_frequency_analysis(self, mfcc: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute tonal/phonetic descriptors from MFCCs and their deltas.
        Returns coefficient-level stats arrays.
        """
        mfcc = self._pad_or_trim_mfcc(mfcc)
        deltas = self._compute_deltas(mfcc)
        d_deltas = self._compute_deltas(deltas)

        def stats_over_time(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            mean = np.mean(mat, axis=0)
            std = np.std(mat, axis=0)
            return mean.astype(np.float32), std.astype(np.float32)

        mean, std = stats_over_time(mfcc)
        d_mean, d_std = stats_over_time(deltas)
        dd_mean, dd_std = stats_over_time(d_deltas)
        return {
            "mfcc_mean": mean,
            "mfcc_std": std,
            "delta_mean": d_mean,
            "delta_std": d_std,
            "delta2_mean": dd_mean,
            "delta2_std": dd_std,
        }

    def _global_descriptors(self, mfcc: np.ndarray, vad: VADResult) -> Dict[str, float]:
        energy = self._energy_proxy(mfcc)
        if energy.size == 0:
            return {
                "energy_mean": 0.0,
                "energy_std": 0.0,
                "energy_cv": 0.0,
                "brightness_ratio": 0.0,
            }
        energy_mean = float(np.mean(energy))
        energy_std = float(np.std(energy))
        energy_cv = float(energy_std / (energy_mean + 1e-8))

        # Spectral brightness proxy: share of energy in higher-order MFCCs
        # Use coefficients 8..target as "brighter" band
        mfcc_t = self._pad_or_trim_mfcc(mfcc)
        high = np.linalg.norm(mfcc_t[:, 8:], axis=1)
        total = np.linalg.norm(mfcc_t, axis=1) + 1e-8
        brightness_ratio = float(np.mean(high / total))

        return {
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "energy_cv": energy_cv,
            "brightness_ratio": brightness_ratio,
            "speech_ratio": float(vad.speech_ratio),
            "speech_segments": float(vad.segment_count),
            "mean_segment_len_frames": float(vad.mean_segment_length_frames),
        }

    # ---------- Emotion (heuristic) ----------
    def infer_emotion(self, global_desc: Dict[str, float]) -> Tuple[str, np.ndarray]:
        """
        Heuristic mapping to {calm, neutral, stressed} using energy variability and brightness.
        Returns (label, probabilities[3]).
        """
        energy_cv = float(global_desc.get("energy_cv", 0.0))
        bright = float(global_desc.get("brightness_ratio", 0.0))
        speech_ratio = float(global_desc.get("speech_ratio", 0.0))

        # Baseline priors
        probs = np.array([0.33, 0.34, 0.33], dtype=np.float32)  # calm, neutral, stressed

        # If very little speech, bias to neutral
        if speech_ratio < 0.2:
            probs = np.array([0.25, 0.6, 0.15], dtype=np.float32)
        else:
            # Emphasize stressed on higher variability/brightness
            stressed_score = 0.0
            stressed_score += np.clip((energy_cv - 0.4) / 0.4, 0.0, 1.0)
            stressed_score += np.clip((bright - 0.45) / 0.35, 0.0, 1.0)
            stressed_score /= 2.0

            calm_score = 1.0 - stressed_score

            # Map to probabilities with some neutral buffer
            probs = np.array([
                0.2 + 0.6 * calm_score,
                0.3,
                0.2 + 0.6 * stressed_score,
            ], dtype=np.float32)
            # Normalize
            probs = probs / float(probs.sum() + 1e-8)

        label_idx = int(np.argmax(probs))
        label = ["calm", "neutral", "stressed"][label_idx]
        return label, probs

    # ---------- Public API ----------
    def process_mfcc_file(self, file_path: Path) -> Dict[str, Any]:
        mfcc = self.load_mfcc(file_path)
        stats = self.mel_frequency_analysis(mfcc)
        vad = self.run_vad(mfcc)
        g = self._global_descriptors(mfcc, vad)
        emotion_label, emotion_probs = self.infer_emotion(g)

        # Build embedding vector
        # Concatenate coefficient stats and global features
        embedding_parts: List[np.ndarray] = [
            stats["mfcc_mean"],
            stats["mfcc_std"],
            stats["delta_mean"],
            stats["delta_std"],
            stats["delta2_mean"],
            stats["delta2_std"],
            np.array([
                g.get("speech_ratio", 0.0),
                g.get("speech_segments", 0.0),
                g.get("mean_segment_len_frames", 0.0),
                g.get("energy_mean", 0.0),
                g.get("energy_std", 0.0),
                g.get("energy_cv", 0.0),
                g.get("brightness_ratio", 0.0),
            ], dtype=np.float32),
        ]

        audio_embedding = np.concatenate(embedding_parts, axis=0).astype(np.float32)

        return {
            "file": str(file_path).replace("\\", "/"),
            "features": {
                "vad": {
                    "speech_ratio": g.get("speech_ratio", 0.0),
                    "segment_count": int(g.get("speech_segments", 0.0)),
                    "mean_segment_length_frames": g.get("mean_segment_len_frames", 0.0),
                },
                "global": g,
                "emotion": {
                    "label": emotion_label,
                    "probs": emotion_probs.tolist(),  # [calm, neutral, stressed]
                },
            },
            "embeddings": {
                "audio": audio_embedding.tolist(),
                "emotion": emotion_probs.tolist(),
            },
            "embedding_dimensions": {
                "audio": int(audio_embedding.shape[0]),
                "emotion": int(emotion_probs.shape[0]),
                "mfcc_dim": int(self.target_mfcc_dim),
            },
        }


# ---------- Directory Processing & CLI ----------
def find_mfcc_files(root: Path, pattern: str = "*_mfcc.npy") -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob(pattern))


def save_json(output_path: Path, data: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Audio Encoder (AE)")
    parser.add_argument("--input-dir", type=str, default="preprocess_output/audio_features",
                        help="Directory containing MFCC .npy files")
    parser.add_argument("--glob", type=str, default="*_mfcc.npy",
                        help="Glob pattern to match MFCC files recursively")
    parser.add_argument("--output", type=str, default="preprocess_output/audio_encoded.json",
                        help="Path to save encoded JSON results")
    parser.add_argument("--target-mfcc-dim", type=int, default=20,
                        help="Number of MFCC coefficients to use (trim/pad)")
    parser.add_argument("--vad-smooth-window", type=int, default=5,
                        help="Smoothing window size (frames) for VAD mask")
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    files = find_mfcc_files(input_root, args.glob)
    if not files:
        raise FileNotFoundError(f"No files matched in {input_root} with pattern {args.glob}")

    encoder = AudioEncoder(target_mfcc_dim=args.target_mfcc_dim, vad_smooth_window=args.vad_smooth_window)

    results: List[Dict[str, Any]] = []
    for i, f in enumerate(files):
        try:
            result = encoder.process_mfcc_file(f)
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


