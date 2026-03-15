"""
gender_counter.py
-----------------
Tracks per-face gender across frames with two separate count modes:

    Current  : faces visible right now (pass active_ids to get_current_counts).
    Lifetime : every unique ID ever confirmed, regardless of visibility.

This module has no dependency on OpenCV or any ML library and can be tested
without a camera.
"""

import time
from collections import Counter


class GenderCounter:
    """
    Parameters
    ----------
    cleanup_interval : float
        Seconds between automatic inactive-face cleanup sweeps.
    inactive_timeout : float
        Seconds after which a face with no update is considered gone.
    """

    def __init__(self,
                 cleanup_interval: float = 30.0,
                 inactive_timeout: float = 5.0):
        # face_id -> "Male" or "Female" (confirmed only)
        self.current_genders = {}
        self.last_seen       = {}

        # Lifetime unique sets
        self.unique_males   = set()
        self.unique_females = set()
        self.uncertain_ids  = set()

        self.cleanup_interval = cleanup_interval
        self.inactive_timeout = inactive_timeout
        self.last_cleanup     = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_gender(self, face_id: int, raw_gender: str) -> str:
        """
        Record or update gender for a tracked face.

        raw_gender may carry confidence markers ("Male?", "Female*") that
        are stripped before processing.  Returns the clean gender string
        or "Unknown" when the value cannot be confirmed.
        """
        self.last_seen[face_id] = time.time()
        clean = self._clean(raw_gender)

        if clean not in ("Male", "Female"):
            self.uncertain_ids.add(face_id)
            return "Unknown"

        self.uncertain_ids.discard(face_id)
        self.current_genders[face_id] = clean

        if clean == "Male":
            self.unique_males.add(face_id)
        else:
            self.unique_females.add(face_id)

        if time.time() - self.last_cleanup > self.cleanup_interval:
            self.cleanup_inactive()

        return clean

    def remove_face(self, face_id: int) -> None:
        """Remove a face from current tracking (lifetime sets preserved)."""
        self.current_genders.pop(face_id, None)
        self.last_seen.pop(face_id, None)
        self.uncertain_ids.discard(face_id)

    def cleanup_inactive(self) -> int:
        """Remove faces not updated within inactive_timeout. Returns count."""
        now   = time.time()
        stale = [fid for fid, ts in self.last_seen.items()
                 if now - ts > self.inactive_timeout]
        for fid in stale:
            self.remove_face(fid)
        self.last_cleanup = now
        return len(stale)

    def get_current_counts(self, active_ids=None) -> tuple:
        """
        Return (male_count, female_count) for currently visible faces.

        active_ids : iterable of face IDs on the current frame.
                     When None, counts all entries in current_genders.
        """
        if active_ids is None:
            c = Counter(self.current_genders.values())
            return c.get("Male", 0), c.get("Female", 0)
        active = set(active_ids)
        males   = sum(1 for fid in active
                      if self.current_genders.get(fid) == "Male")
        females = sum(1 for fid in active
                      if self.current_genders.get(fid) == "Female")
        return males, females

    def get_total_counts(self) -> tuple:
        return len(self.unique_males), len(self.unique_females)

    def get_stats_dict(self, active_ids=None) -> dict:
        cm, cf = self.get_current_counts(active_ids=active_ids)
        tm, tf = self.get_total_counts()
        return {
            "current_male":    cm,
            "current_female":  cf,
            "total_male":      tm,
            "total_female":    tf,
            "total_unique":    tm + tf,
            "uncertain":       len(self.uncertain_ids),
        }

    def reset(self) -> None:
        self.current_genders.clear()
        self.last_seen.clear()
        self.unique_males.clear()
        self.unique_females.clear()
        self.uncertain_ids.clear()
        self.last_cleanup = time.time()

    def __repr__(self) -> str:
        cm, cf = self.get_current_counts()
        tm, tf = self.get_total_counts()
        return f"GenderCounter(current={cm}M/{cf}F, total={tm}M/{tf}F)"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(gender_str: str) -> str:
        """Strip confidence markers and validate."""
        if not isinstance(gender_str, str):
            return "Unknown"
        cleaned = gender_str.replace("*", "").replace("?", "").strip()
        return cleaned if cleaned in ("Male", "Female") else "Unknown"
