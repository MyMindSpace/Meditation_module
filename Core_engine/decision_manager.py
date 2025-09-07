from typing import Dict, Any


class DecisionManager:
    """
    Decision Manager (DM)

    Combines fused features and quality report to produce high-level actions.
    """

    def __init__(self) -> None:
        pass

    def decide(self, fused: Dict[str, Any], quality: Dict[str, Any]) -> Dict[str, Any]:
        actions: Dict[str, Any] = {"adjustments": {}, "notes": []}

        overall_ok = bool(quality.get("overall_ok", False))
        actions["overall_ok"] = overall_ok

        # Default: choose meditation recommendation if available
        weights = fused.get("weights", {})
        if weights:
            # Example policy: if posture is low weight, downplay posture coaching
            if float(weights.get("pdm", 0.0)) < 0.2:
                actions["adjustments"]["posture_coaching"] = "minimize"
            else:
                actions["adjustments"]["posture_coaching"] = "enable"

        # Quality-based fallbacks
        for name, metrics in quality.get("models", {}).items():
            if not metrics.get("ok"):
                if name == "msm":
                    actions["adjustments"]["recommendation_mode"] = "rule_only"
                if name == "tts":
                    actions["adjustments"]["voice_style"] = "neutral"
                if name == "arm":
                    actions["adjustments"]["sentiment_usage"] = "skip"

        actions["notes"].append("Decision computed from fused features and quality report")
        return actions


