from typing import Dict, Any


class QualityMonitor:
    """
    Quality Monitor (QM)

    - Tracks model confidences
    - Applies thresholds and flags low-confidence outputs
    - Suggests fallbacks
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    def assess(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Assess confidences for MSM, PDM, TTSM, ARM.

        Returns a dict with per-model ok flags and overall status.
        """
        report: Dict[str, Any] = {"threshold": self.threshold, "models": {}, "overall_ok": True}
        for name, payload in model_outputs.items():
            conf = float((payload or {}).get("confidence") or 0.0)
            ok = conf >= self.threshold
            report["models"][name] = {"confidence": conf, "ok": ok}
            if not ok:
                report["overall_ok"] = False
        return report

    def recommend_fallback(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest fallbacks for low-confidence models."""
        fallbacks = {}
        for name, metrics in report.get("models", {}).items():
            if not metrics.get("ok"):
                if name == "msm":
                    fallbacks[name] = "Use rule-based recommendation only"
                elif name == "pdm":
                    fallbacks[name] = "Lower sensitivity, skip posture coaching"
                elif name == "tts":
                    fallbacks[name] = "Use neutral voice settings"
                elif name == "arm":
                    fallbacks[name] = "Skip sentiment; rely on diagnosis feedback"
        return {"fallbacks": fallbacks, "has_fallbacks": bool(fallbacks)}


