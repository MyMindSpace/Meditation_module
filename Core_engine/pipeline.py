import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core_engine.fusion import MultiModalFusion
from Core_engine.quality_monitor import QualityMonitor
from Core_engine.decision_manager import DecisionManager


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _hash_embed(text: str, dim: int = 32) -> np.ndarray:
    # Simple deterministic hash to vector
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec
    h = abs(hash(text))
    rng = np.random.default_rng(h % (2**32))
    vec = rng.standard_normal(dim).astype(np.float32)
    # L2 normalize
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec


def load_msm(meditation_json: Path) -> Dict[str, Any]:
    data = _read_json(meditation_json) or {}
    # Expect structure from meditation_selector
    if isinstance(data, dict) and "recommendations" in data:
        recs = data.get("recommendations", [])
        top = recs[0] if recs else {}
        label = str(top.get("meditation_type") or "")
        conf = float(top.get("confidence_score") or 0.0)
        emb = _hash_embed(label, dim=32)
        return {"embedding": emb.tolist(), "confidence": conf, "label": label}
    return {"embedding": [], "confidence": 0.0}


def load_pdm(posture_json: Path, vision_json: Path) -> Dict[str, Any]:
    # Prefer pose/visual embedding from vision encoder; use posture score as confidence
    vision = _read_json(vision_json)
    pose_vec = []
    conf = 0.0
    if isinstance(vision, list) and vision:
        emb = (vision[0].get("embeddings") or {})
        combined = emb.get("combined") or []
        if combined:
            pose_vec = combined
    posture = _read_json(posture_json)
    if isinstance(posture, list) and posture:
        conf = float(posture[0].get("posture_score") or 0.0)
    return {"embedding": pose_vec, "confidence": conf}


def load_arm(audio_json: Path) -> Dict[str, Any]:
    data = _read_json(audio_json)
    if isinstance(data, list) and data:
        emb = (data[0].get("embeddings") or {}).get("audio") or []
        # Confidence from emotion distribution
        probs = (data[0].get("embeddings") or {}).get("emotion") or []
        conf = float(max(probs) if probs else 0.0)
        return {"embedding": emb, "confidence": conf}
    return {"embedding": [], "confidence": 0.0}


def main() -> None:
    # Default expected outputs from earlier steps
    base = Path("preprocess_output")
    paths = {
        "msm": Path("meditation_recommendations_output.json"),
        "pdm": base / "posture_scores.json",
        "vision": base / "vision_encoded.json",
        "audio": base / "audio_encoded.json",
    }

    msm = load_msm(paths["msm"])  # meditation selector
    pdm = load_pdm(paths["pdm"], paths["vision"])  # posture/vision
    arm = load_arm(paths["audio"])  # audio response proxy

    # TTS placeholder (no embedding available here)
    tts = {"embedding": [], "confidence": 0.0}

    fusion = MultiModalFusion()
    fused = fusion.fuse({"msm": msm, "pdm": pdm, "tts": tts, "arm": arm})

    qm = QualityMonitor(threshold=0.5)
    qrep = qm.assess({"msm": msm, "pdm": pdm, "tts": tts, "arm": arm})
    qfb = qm.recommend_fallback(qrep)

    dm = DecisionManager()
    decision = dm.decide(fused, qrep)

    out = {
        "inputs": {"msm": msm, "pdm": pdm, "arm": arm, "tts": tts},
        "fusion": fused,
        "quality": qrep,
        "fallbacks": qfb,
        "decision": decision,
    }

    out_path = base / "fused_decision.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved fused decision to {out_path}")


if __name__ == "__main__":
    main()


