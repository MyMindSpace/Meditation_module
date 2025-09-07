import math
from typing import Dict, Any, Tuple

import numpy as np


class MultiModalFusion:
    """
    Multi-Modal Fusion (MF)

    - Inputs: outputs from MSM (meditation selector), PDM (posture), TTSM, ARM
    - Processing: attention-like weighting using provided confidences; produce fused feature vector
    - Output: fused feature set and combined confidence
    """

    def __init__(self) -> None:
        pass

    def _safe_vec(self, arr: Any) -> np.ndarray:
        try:
            v = np.array(arr, dtype=np.float32).reshape(-1)
        except Exception:
            v = np.zeros(0, dtype=np.float32)
        return v

    def _weight(self, conf: float) -> float:
        conf = float(conf if conf is not None else 0.0)
        return 1.0 / (1.0 + math.exp(-4.0 * (conf - 0.5)))  # squash to (0,1)

    def fuse(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            inputs: {
              'msm': {'embedding': [...], 'confidence': float},
              'pdm': {'embedding': [...], 'confidence': float},
              'tts': {'embedding': [...], 'confidence': float},
              'arm': {'embedding': [...], 'confidence': float},
            }

        Returns: {
          'fused_embedding': [...],
          'weights': {'msm': w, 'pdm': w, 'tts': w, 'arm': w},
          'combined_confidence': float
        }
        """
        keys = ['msm', 'pdm', 'tts', 'arm']
        vectors = {}
        weights = {}
        for k in keys:
            v = self._safe_vec((inputs.get(k) or {}).get('embedding'))
            c = float((inputs.get(k) or {}).get('confidence') or 0.0)
            vectors[k] = v
            weights[k] = self._weight(c)

        # Pad vectors to same length
        max_len = max([len(v) for v in vectors.values()] + [0])
        def pad(v: np.ndarray) -> np.ndarray:
            if v.size == 0:
                return np.zeros(max_len, dtype=np.float32)
            if v.size == max_len:
                return v
            out = np.zeros(max_len, dtype=np.float32)
            out[:v.size] = v
            return out

        stacked = []
        ws = []
        for k in keys:
            stacked.append(pad(vectors[k]))
            ws.append(weights[k])
        if not stacked:
            fused = np.zeros(0, dtype=np.float32)
        else:
            W = np.array(ws, dtype=np.float32)
            if W.sum() <= 1e-8:
                W = np.ones_like(W) / float(len(W))
            else:
                W = W / W.sum()
            M = np.stack(stacked, axis=0)  # [K, D]
            fused = (W[:, None] * M).sum(axis=0)

        combined_conf = float(np.clip(sum(ws) / max(len(ws), 1), 0.0, 1.0))
        return {
            'fused_embedding': fused.tolist(),
            'weights': weights,
            'combined_confidence': combined_conf,
        }


