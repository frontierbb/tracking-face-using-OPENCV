"""
face_tracker.py
---------------
Lightweight centroid-based face tracker.

Design
    Each detected bounding box is matched to an existing tracked face by
    finding the nearest centroid within a distance threshold.  Unmatched
    detections receive a new unique ID.  Faces that have not been updated
    for more than `max_age` seconds are removed from the active set.

    This is intentionally simple: no Kalman filter, no Hungarian algorithm.
    For a webcam use-case with frame_skip=6 and faces rarely moving faster
    than 80 px/frame, nearest-centroid matching is sufficient and adds
    negligible CPU cost.

Output contract
    update() takes a list of face dicts from FaceDetector and returns an
    ordered dict mapping face_id -> face_dict, where each face_dict is
    enriched with:
        "face_id"  : int   stable ID across frames
        "center"   : (cx, cy)
        "last_seen": float  time.time() of last update
"""

import time
import numpy as np
from collections import OrderedDict


class FaceTracker:
    """
    Parameters
    ----------
    max_distance : float
        Maximum centroid distance (pixels) to consider two detections the
        same face across frames.
    max_age : float
        Seconds after which an unseen face is removed from tracking.
    """

    def __init__(self, max_distance: float = 120.0, max_age: float = 3.0):
        self.max_distance = max_distance
        self.max_age      = max_age

        self._tracked   = OrderedDict()   # face_id -> face_dict
        self._next_id   = 1
        self._total_seen = 0              # lifetime count (for statistics)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: list) -> OrderedDict:
        """
        Match detections to existing tracks; create new tracks for
        unmatched detections; expire stale tracks.

        Parameters
        ----------
        detections : list of dict from FaceDetector.detect()
                     Each dict must contain "bbox": (x, y, w, h).

        Returns
        -------
        OrderedDict  face_id -> enriched face dict
        """
        now = time.time()
        used_ids = set()

        new_tracked = OrderedDict()

        for det in detections:
            x, y, w, h = det["bbox"]
            cx, cy = x + w // 2, y + h // 2

            best_id   = None
            best_dist = float("inf")

            for fid, fdata in self._tracked.items():
                if fid in used_ids:
                    continue
                ox, oy = fdata["center"]
                dist = np.hypot(cx - ox, cy - oy)
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_id   = fid

            if best_id is None:
                best_id = self._next_id
                self._next_id   += 1
                self._total_seen += 1

            used_ids.add(best_id)
            new_tracked[best_id] = {
                **det,
                "face_id":   best_id,
                "center":    (cx, cy),
                "last_seen": now,
            }

        # Carry forward recently-seen tracks that had no detection this frame
        # (face temporarily occluded or missed by detector).
        for fid, fdata in self._tracked.items():
            if fid not in used_ids:
                age = now - fdata["last_seen"]
                if age <= self.max_age:
                    new_tracked[fid] = fdata   # keep stale entry

        self._tracked = new_tracked
        return self._tracked

    @property
    def active_ids(self) -> list:
        return list(self._tracked.keys())

    @property
    def total_seen(self) -> int:
        return self._total_seen

    def reset(self) -> None:
        self._tracked    = OrderedDict()
        self._next_id    = 1
        self._total_seen = 0
