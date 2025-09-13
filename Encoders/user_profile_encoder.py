import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import hashlib
import math
from datetime import datetime, timezone


class UserProfileEncoder:
    def encode_feedback(self, feedback: Dict[str, Any], dim: int = 32) -> np.ndarray:
        """
        Encode feedback into a fixed-size vector.
        Supports numeric ratings, sentiment, and hash-based embedding for text.
        """
        vec = np.zeros(dim, dtype=np.float32)
        if not isinstance(feedback, dict):
            return vec
        # Numeric features
        rating = feedback.get("rating")
        sentiment = feedback.get("sentiment")
        if rating is not None:
            try:
                vec[0] = float(rating)
            except Exception:
                pass
        if sentiment is not None:
            try:
                vec[1] = float(sentiment)
            except Exception:
                pass
        # Text feature (hash-based)
        text = feedback.get("text")
        if text:
            idx = self._stable_hash_index(text, dim)
            vec[idx] += 1.0
        # Normalize
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec
    """
    User Profile Encoder (UE)

    Input: Aggregated user data records with optional sections:
        - preferences: user choices like content types, modalities, goals, tags
        - progress: usage progress metrics
        - behavior: temporal usage patterns, histograms

    Processing:
        - Preference Embeddings: hash-based fixed-size vector for categorical preferences
        - Progress Vectors: normalized numeric features capturing progress/recency
        - Behavioral Patterns: normalized distributions and summary stats for habits

    Output:
        - A comprehensive numerical embedding per user profile
    """

    def __init__(self,
                 preference_dim: int = 256,
                 time_of_day_bins: Tuple[str, ...] = ("morning", "afternoon", "evening", "night"),
                 weekday_bins: Tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
        self.preference_dim = preference_dim
        self.time_of_day_bins = time_of_day_bins
        self.weekday_bins = weekday_bins

    # ---------- Helpers ----------
    def _stable_hash_index(self, text: str, dim: int) -> int:
        text = (text or "").strip().lower()
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        return int(h, 16) % dim

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    # ---------- Preference Embedding ----------
    def encode_preferences(self, preferences: Dict[str, Any]) -> np.ndarray:
        """
        Hash-based multi-hot embedding for categorical preferences.

        Supported keys (if present): content_types, modalities, goals, tags, languages, difficulties
        Any string values under these keys contribute to the vector via hashing.
        """
        embedding = np.zeros(self.preference_dim, dtype=np.float32)

        if not isinstance(preferences, dict):
            return embedding

        preference_keys = [
            "content_types", "modalities", "goals", "tags", "languages", "difficulties"
        ]

        for key in preference_keys:
            values = preferences.get(key)
            if values is None:
                continue
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, (list, tuple)):
                continue
            for value in values:
                if not isinstance(value, str):
                    value = str(value)
                idx = self._stable_hash_index(f"{key}:{value}", self.preference_dim)
                embedding[idx] += 1.0

        # L2 normalize to keep scale consistent
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        return embedding

    # ---------- Progress Vector ----------
    def encode_progress(self, progress: Dict[str, Any]) -> np.ndarray:
        """
        Numeric progress features normalized/clipped to [0,1].

        Considers (if available):
        - sessions_completed, avg_session_duration_sec, streak_days, completion_rate
        - level, points, last_active_ts (ISO8601)
        """
        if not isinstance(progress, dict):
            progress = {}

        def get_number(name: str, default: float = 0.0) -> float:
            val = progress.get(name, default)
            try:
                return float(val)
            except Exception:
                return default

        # Raw values
        sessions_completed = get_number("sessions_completed")
        avg_session_duration_sec = get_number("avg_session_duration_sec")
        streak_days = get_number("streak_days")
        completion_rate = get_number("completion_rate")  # expected 0..1; clamp later
        level = get_number("level")
        points = get_number("points")

        # Recency from last_active_ts
        last_active_ts = progress.get("last_active_ts")
        recency_days = 365.0  # default old
        if isinstance(last_active_ts, str) and last_active_ts:
            try:
                last_dt = datetime.fromisoformat(last_active_ts.replace("Z", "+00:00"))
                delta = self._now_utc() - last_dt.astimezone(timezone.utc)
                recency_days = max(delta.total_seconds() / 86400.0, 0.0)
            except Exception:
                pass

        # Normalizations (soft saturations)
        def squash(x: float, k: float) -> float:
            # 1 - exp(-x/k) scaled to [0,1) for x>=0
            x = max(x, 0.0)
            return 1.0 - math.exp(-x / max(k, 1e-6))

        sessions_completed_n = squash(sessions_completed, 100.0)
        avg_session_duration_n = squash(avg_session_duration_sec / 60.0, 60.0)  # minutes vs 60-min scale
        streak_days_n = squash(streak_days, 30.0)
        completion_rate_n = float(np.clip(completion_rate, 0.0, 1.0))
        level_n = squash(level, 50.0)
        points_n = squash(points, 10000.0)
        recency_days_n = 1.0 - squash(recency_days, 30.0)  # recent -> closer to 1

        return np.array([
            sessions_completed_n,
            avg_session_duration_n,
            streak_days_n,
            completion_rate_n,
            level_n,
            points_n,
            recency_days_n,
        ], dtype=np.float32)

    # ---------- Behavioral Patterns ----------
    def encode_behavior(self, behavior: Dict[str, Any]) -> np.ndarray:
        """
        Habit/engagement features:
        - time_of_day histogram over [morning, afternoon, evening, night]
        - weekday histogram over [Mon..Sun]
        - session length stats: mean, std, p25, p50, p75 (minutes, squashed)
        - cadence_days stats (intervals between sessions): mean, std (days, squashed)
        """
        if not isinstance(behavior, dict):
            behavior = {}

        # Time of day histogram
        tod_hist = behavior.get("time_of_day_hist", {}) or {}
        tod_vec = np.zeros(len(self.time_of_day_bins), dtype=np.float32)
        total_tod = 0.0
        for i, name in enumerate(self.time_of_day_bins):
            count = float(tod_hist.get(name, 0.0) or 0.0)
            tod_vec[i] = max(count, 0.0)
            total_tod += tod_vec[i]
        if total_tod > 0:
            tod_vec = tod_vec / total_tod

        # Weekday histogram
        wd_hist = behavior.get("days_active_hist", {}) or {}
        wd_vec = np.zeros(len(self.weekday_bins), dtype=np.float32)
        total_wd = 0.0
        for i, name in enumerate(self.weekday_bins):
            count = float(wd_hist.get(name, 0.0) or 0.0)
            wd_vec[i] = max(count, 0.0)
            total_wd += wd_vec[i]
        if total_wd > 0:
            wd_vec = wd_vec / total_wd

        # Session length statistics (seconds -> minutes)
        session_lengths = behavior.get("session_lengths_sec", []) or []
        sl = np.array([max(float(x), 0.0) for x in session_lengths], dtype=np.float32)
        sl_minutes = sl / 60.0 if sl.size > 0 else np.array([], dtype=np.float32)
        if sl_minutes.size == 0:
            sl_mean = sl_std = sl_p25 = sl_p50 = sl_p75 = 0.0
        else:
            sl_mean = float(np.mean(sl_minutes))
            sl_std = float(np.std(sl_minutes))
            sl_p25 = float(np.percentile(sl_minutes, 25))
            sl_p50 = float(np.percentile(sl_minutes, 50))
            sl_p75 = float(np.percentile(sl_minutes, 75))

        def squash_minutes(x: float) -> float:
            # 1 - exp(-x/60) caps around long sessions
            return 1.0 - math.exp(-max(x, 0.0) / 60.0)

        sl_stats_vec = np.array([
            squash_minutes(sl_mean),
            squash_minutes(sl_std),
            squash_minutes(sl_p25),
            squash_minutes(sl_p50),
            squash_minutes(sl_p75),
        ], dtype=np.float32)

        # Cadence (days between sessions)
        cadence_days = behavior.get("cadence_days", []) or []
        cd = np.array([max(float(x), 0.0) for x in cadence_days], dtype=np.float32)
        if cd.size == 0:
            cd_mean = cd_std = 0.0
        else:
            cd_mean = float(np.mean(cd))
            cd_std = float(np.std(cd))

        def squash_days(x: float) -> float:
            # 1 - exp(-x/30) so weekly/monthly cadence map into (0,1)
            return 1.0 - math.exp(-max(x, 0.0) / 30.0)

        cd_stats_vec = np.array([
            squash_days(cd_mean),
            squash_days(cd_std),
        ], dtype=np.float32)

        return np.concatenate([tod_vec, wd_vec, sl_stats_vec, cd_stats_vec], axis=0)

    # ---------- Public API ----------
    def process_user_profile(self, record: Dict[str, Any]) -> Dict[str, Any]:
        user_id = record.get("userId")
        preferences = record.get("preferences", {}) or {}
        progress = record.get("progress", {}) or {}
        behavior = record.get("behavior", {}) or {}
        feedback = record.get("feedback", {}) or {}

        pref_vec = self.encode_preferences(preferences)
        prog_vec = self.encode_progress(progress)
        beh_vec = self.encode_behavior(behavior)
        feed_vec = self.encode_feedback(feedback)

        combined = np.concatenate([pref_vec, prog_vec, beh_vec, feed_vec], axis=0)

        return {
            "userId": user_id,
            "embeddings": {
                "preferences": pref_vec.tolist(),
                "progress": prog_vec.tolist(),
                "behavior": beh_vec.tolist(),
                "feedback": feed_vec.tolist(),
                "combined": combined.tolist(),
            },
            "embedding_dimensions": {
                "preferences": int(pref_vec.shape[0]),
                "progress": int(prog_vec.shape[0]),
                "behavior": int(beh_vec.shape[0]),
                "feedback": int(feed_vec.shape[0]),
                "combined": int(combined.shape[0]),
            },
            "features": {
                "preferences": preferences,
                "progress": progress,
                "behavior": behavior,
                "feedback": feedback,
            }
        }


# ---------- IO Utilities ----------
def load_json_any(input_path: Path) -> Any:
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_user_profiles(input_path: Path) -> List[Dict[str, Any]]:
    data = load_json_any(input_path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Accept dict of userId -> profile
        profiles = []
        for user_id, payload in data.items():
            if isinstance(payload, dict):
                payload = dict(payload)
                payload.setdefault("userId", user_id)
                profiles.append(payload)
        return profiles
    raise ValueError("Input JSON must be a list of user profiles or a dict keyed by userId")


def save_encoded_profiles(output_path: Path, data: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------- Optional Auto Aggregation from preprocess_output ----------
def discover_candidate_profile_jsons(root: Path) -> List[Path]:
    if not root.exists():
        return []
    candidates: List[Path] = []
    for p in root.rglob("*.json"):
        name = p.name.lower()
        if "profile" in name or "aggregat" in name or "user" in name:
            candidates.append(p)
    return candidates


def try_aggregate_from_jsons(json_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Heuristic aggregation: merge dicts by userId when structure looks like user profiles.
    Records without a userId are skipped.
    """
    aggregated: Dict[str, Dict[str, Any]] = {}
    for p in json_paths:
        try:
            data = load_json_any(p)
        except Exception:
            continue

        # Normalize to list of dicts
        if isinstance(data, dict):
            items = []
            # if mapping userId->payload
            if all(isinstance(k, str) for k in data.keys()):
                for uid, payload in data.items():
                    if isinstance(payload, dict):
                        payload = dict(payload)
                        payload.setdefault("userId", uid)
                        items.append(payload)
            else:
                items = [data]
        elif isinstance(data, list):
            items = [x for x in data if isinstance(x, dict)]
        else:
            items = []

        for item in items:
            user_id = item.get("userId")
            if not user_id:
                continue
            dst = aggregated.setdefault(user_id, {"userId": user_id, "preferences": {}, "progress": {}, "behavior": {}, "_sources": []})

            # Merge top-level known keys without overwriting existing scalar metrics unless missing
            for key in ("preferences", "progress", "behavior"):
                if key in item and isinstance(item[key], dict):
                    # shallow merge; lists from source are preferred if destination empty
                    for k, v in item[key].items():
                        if k not in dst[key] or not dst[key][k]:
                            dst[key][k] = v
            dst["_sources"].append(str(p))

    profiles = list(aggregated.values())
    # Attach source list under meta
    for prof in profiles:
        meta = prof.setdefault("meta", {})
        meta["preprocess_source_files"] = prof.pop("_sources", [])
    return profiles


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="User Profile Encoder (UE)")
    parser.add_argument("--input", type=str, default="preprocess_output/user_feedback_processed.json",
                        help="Path to processed user feedback JSON (list or dict keyed by userId)")
    parser.add_argument("--output", type=str, default="encoder_output/user_profiles_encoded.json",
                        help="Path to save encoded user profiles JSON")
    parser.add_argument("--auto-aggregate", action="store_true",
                        help="If set, attempt to discover and aggregate user profiles from preprocess_output")
    parser.add_argument("--preference-dim", type=int, default=256, help="Dimension of preference embedding")
    args = parser.parse_args()

    encoder = UserProfileEncoder(preference_dim=args.preference_dim)

    input_path = Path(args.input)
    output_path = Path(args.output)

    records: List[Dict[str, Any]]
    if args.auto_aggregate:
        root = input_path.parent if input_path.exists() else Path("preprocess_output")
        print(f"Auto-aggregating profiles from {root}...")
        candidates = discover_candidate_profile_jsons(root)
        records = try_aggregate_from_jsons(candidates)
        if not records:
            print("No profiles discovered. Falling back to --input if exists...")
            if input_path.exists():
                records = load_user_profiles(input_path)
            else:
                raise FileNotFoundError("No aggregated profiles found and input file missing.")
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        records = load_user_profiles(input_path)

    print(f"Encoding {len(records)} user profiles...")
    encoded_records: List[Dict[str, Any]] = []

    for i, record in enumerate(records):
        try:
            encoded = encoder.process_user_profile(record)
            encoded_records.append(encoded)
        except Exception as e:
            print(f"Error processing record {i+1}/{len(records)}: {e}")

    save_encoded_profiles(output_path, encoded_records)
    print(f"Saved {len(encoded_records)} encoded profiles to {output_path}")

    if encoded_records:
        dims = encoded_records[0]["embedding_dimensions"]
        print("\nEmbedding dimensions:")
        for k, v in dims.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


