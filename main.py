"""
main.py
-------
Entry point for the Enhanced Human Detector (v3 - modular stack).

Pipeline per frame
    1. FaceDetector  (MediaPipe BlazeFace)   -> bounding boxes
    2. FaceTracker   (centroid matching)     -> stable face IDs
    3. LandmarkDetector (MediaPipe FaceMesh) -> 468-point landmarks
                                               (run every frame for smooth overlay)
    4. AttributePredictor (FairFace ONNX)   -> gender / age / ethnicity
                                               (run every N frames via cache)
    5. EmotionDetector (HSEmotion ONNX)     -> 8-class emotion
                                               (run every N frames via cache)
    6. GenderCounter                        -> current + lifetime gender stats
    7. Renderer                             -> draw boxes, landmarks, HUD

Module responsibility summary
    face_detector.py       : BlazeFace detection only
    landmark_detector.py   : Face Mesh 468-point detection only
    attribute_predictor.py : FairFace inference only (gender/age/ethnicity)
    emotion_detector.py    : HSEmotion inference only
    face_tracker.py        : ID assignment and staleness management
    gender_counter.py      : gender statistics (current + lifetime)
    model_manager.py       : model download and path resolution
    main.py                : orchestration, rendering, keyboard handling

Keyboard controls
    q  Quit
    s  Save screenshot
    r  Reset all state
    t  Print statistics to terminal
    l  Toggle landmark overlay

Performance tuning
    ANALYSIS_EVERY_N_FRAMES : int
        How many frames to skip between full inference runs.
        Higher = faster FPS, slower attribute refresh.
        Recommended: 6-10 on CPU.
    CACHE_SECONDS : float
        How long an inference result stays valid before being refreshed.
"""

import time
import cv2
import numpy as np
from collections import deque, Counter

from face_detector      import FaceDetector
from landmark_detector  import LandmarkDetector, EYE_LEFT_POINTS, EYE_RIGHT_POINTS
from attribute_predictor import AttributePredictor
from emotion_detector   import EmotionDetector
from face_tracker       import FaceTracker
from gender_counter     import GenderCounter
import model_manager


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_INDEX          = 0
FRAME_WIDTH           = 1280
FRAME_HEIGHT          = 720
TARGET_FPS            = 20

ANALYSIS_EVERY_N_FRAMES = 15      # run attribute + emotion inference every N frames
CACHE_SECONDS           = 4.0    # max age of cached analysis result
HISTORY_LENGTH          = 10     # frames of history for temporal smoothing
AGE_HISTORY_LENGTH      = 20     # longer window for age (slow-changing attribute)

COLOR_PRIMARY   = (255, 200, 0)
COLOR_TEXT      = (255, 255, 255)
COLOR_LABEL     = (180, 180, 180)
COLOR_LANDMARK  = (0, 220, 220)


# ---------------------------------------------------------------------------
# Temporal smoother
# ---------------------------------------------------------------------------

class TemporalSmoother:
    """
    Maintains per-face majority-vote histories for each attribute.
    Gender markers (* ?) are stripped before insertion so they do not
    split the vote.
    """

    def __init__(self):
        self.gender    = {}
        self.age       = {}
        self.ethnicity = {}
        self.emotion   = {}

    def update(self, face_id: int, result: dict) -> dict:
        if face_id not in self.gender:
            self.gender[face_id]    = deque(maxlen=HISTORY_LENGTH)
            self.age[face_id]       = deque(maxlen=AGE_HISTORY_LENGTH)
            self.ethnicity[face_id] = deque(maxlen=HISTORY_LENGTH)
            self.emotion[face_id]   = deque(maxlen=HISTORY_LENGTH)

        # Strip uncertainty markers before voting
        clean_gender = result["gender"].replace("*", "").replace("?", "").strip()

        self.gender[face_id].append(clean_gender)
        self.age[face_id].append(result["age"])
        self.ethnicity[face_id].append(result["ethnicity"])
        self.emotion[face_id].append(result["emotion"])

        return {
            "gender":    Counter(self.gender[face_id]).most_common(1)[0][0],
            "age":       Counter(self.age[face_id]).most_common(1)[0][0],
            "ethnicity": Counter(self.ethnicity[face_id]).most_common(1)[0][0],
            "emotion":   Counter(self.emotion[face_id]).most_common(1)[0][0],
        }

    def purge(self, active_ids: set) -> None:
        """Remove history for faces no longer tracked."""
        for store in (self.gender, self.age, self.ethnicity, self.emotion):
            gone = [fid for fid in store if fid not in active_ids]
            for fid in gone:
                del store[fid]


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class Renderer:
    """Draws all visual overlays onto the frame."""

    def __init__(self):
        self.show_landmarks = False

    def draw_face(self, frame: np.ndarray, face_id: int,
                  bbox: tuple, analysis: dict) -> None:
        """Draw bounding box corners and info panel for one face."""
        fh, fw = frame.shape[:2]
        x, y, w, h = bbox

        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))

        # Corner-bracket bounding box
        ln = int(w * 0.20)
        for dx, dy in [(0, 0), (w, 0), (0, h), (w, h)]:
            cx, cy = x + dx, y + dy
            sx = 1 if dx == 0 else -1
            sy = 1 if dy == 0 else -1
            cv2.line(frame, (cx, cy), (cx + sx * ln, cy), COLOR_PRIMARY, 2)
            cv2.line(frame, (cx, cy), (cx, cy + sy * ln), COLOR_PRIMARY, 2)

        # Info panel
        lines = [
            ("GENDER",  analysis.get("gender",    "...")),
            ("AGE",     analysis.get("age",        "...")),
            ("ETHNIC",  analysis.get("ethnicity",  "...")),
            ("EMOTION", analysis.get("emotion",    "...")),
        ]

        sx = x + w + 10
        sy = y
        bg_w, bg_h = 190, 100

        if sx + bg_w > fw:
            sx = x - bg_w - 10
        if sy + bg_h > fh:
            sy = fh - bg_h - 5

        overlay = frame.copy()
        cv2.rectangle(overlay, (sx, sy), (sx + bg_w, sy + bg_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"ID {face_id:02d}",
                    (sx + 5, sy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_PRIMARY, 1, cv2.LINE_AA)
        cv2.line(frame, (sx + 5, sy + 18), (sx + 45, sy + 18), COLOR_PRIMARY, 1)

        for i, (lbl, val) in enumerate(lines):
            ly = sy + 32 + i * 16
            cv2.putText(frame, f"{lbl}:", (sx + 5, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLOR_LABEL, 1, cv2.LINE_AA)
            cv2.putText(frame, str(val).upper(), (sx + 80, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLOR_TEXT,  1, cv2.LINE_AA)

    def draw_landmarks(self, frame: np.ndarray, lm_data: dict) -> None:
        """Draw 468-point landmark dots."""
        if lm_data is None:
            return
        pts = lm_data["points"].astype(int)
        for (px, py) in pts:
            if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                cv2.circle(frame, (px, py), 1, COLOR_LANDMARK, -1)

    def draw_hud(self, frame: np.ndarray, fps: float,
                 gender_counter: GenderCounter,
                 active_ids: list) -> np.ndarray:
        """Draw top bar with counts and FPS."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cm, cf = gender_counter.get_current_counts(active_ids=active_ids)
        tm, tf = gender_counter.get_total_counts()

        cv2.putText(frame,
                    f"Now: {cm}M / {cf}F  ({cm + cf} on screen)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame,
                    f"Total: {tm}M / {tf}F  ({tm + tf} unique)",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        fps_color = ((0, 255, 0) if fps >= 15
                     else (0, 165, 255) if fps >= 8
                     else (0, 0, 255))
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (w - 115, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)

        lm_status = "LM: ON" if self.show_landmarks else "LM: OFF"
        cv2.putText(frame, lm_status, (w - 115, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.putText(frame,
                    "q:Quit  s:Save  r:Reset  t:Stats  l:Landmarks",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (200, 200, 200), 1)
        return frame


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class HumanDetector:
    """
    Orchestrates the full detection pipeline.

    Initialises all sub-modules, runs the main capture loop and handles
    keyboard input.
    """

    def __init__(self):
        print("=" * 60)
        print("HUMAN DETECTOR V3  (MediaPipe + FairFace + HSEmotion)")
        print("=" * 60)

        # Download models before initialising inference sessions
        print("\nChecking / downloading models...")
        model_manager.ensure_all()

        print("\nLoading modules...")
        self.face_detector   = FaceDetector(min_confidence=0.65, model_selection=1)
        self.landmark_det    = LandmarkDetector(max_faces=10)
        self.attr_predictor  = AttributePredictor(confidence_threshold=0.55)
        self.emotion_det     = EmotionDetector(min_confidence=0.45)
        self.tracker         = FaceTracker(max_distance=120, max_age=3.0)
        self.gender_counter  = GenderCounter(cleanup_interval=30.0)
        self.smoother        = TemporalSmoother()
        self.renderer        = Renderer()

        # Analysis cache: face_id -> (smoothed_result, timestamp)
        self.analysis_cache: dict = {}

        # Statistics
        self.attr_stats  = {k: Counter() for k in ("gender", "age",
                                                     "ethnicity", "emotion")}
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0

        print("\nAll modules ready.\n")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

        if not cap.isOpened():
            print("Cannot open camera.")
            return

        print("Camera started. Press q to quit.\n")

        screenshot_count = 0
        fps_start        = time.time()
        fps_count        = 0
        last_cleanup     = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                self.frame_count += 1
                now = time.time()

                # ---- 1. Face detection (every ANALYSIS_EVERY_N_FRAMES) ----
                if self.frame_count % ANALYSIS_EVERY_N_FRAMES == 0:
                    detections = self.face_detector.detect(frame)
                    tracked    = self.tracker.update(detections)
                else:
                    tracked = self.tracker._tracked

                active_ids = list(tracked.keys())

                # ---- 2. Landmark detection (every frame for smooth overlay) ----
                bboxes   = [tracked[fid]["bbox"] for fid in active_ids]
                lm_list  = self.landmark_det.get_landmarks(frame, bboxes)
                lm_map   = {fid: lm for fid, lm in zip(active_ids, lm_list)}

                # ---- 3. Attribute + emotion inference (cached) ----
                if self.frame_count % ANALYSIS_EVERY_N_FRAMES == 0:
                    self._update_analyses(frame, tracked, now)

                # ---- 4. Render ----
                for fid in active_ids:
                    bbox     = tracked[fid]["bbox"]
                    cached   = self.analysis_cache.get(fid)
                    analysis = cached[0] if cached else {}

                    self.renderer.draw_face(frame, fid, bbox, analysis)

                    if self.renderer.show_landmarks:
                        self.renderer.draw_landmarks(frame, lm_map.get(fid))

                # ---- 5. HUD ----
                fps_count += 1
                if now - fps_start >= 1.0:
                    self.fps_history.append(fps_count / (now - fps_start))
                    fps_count = 0
                    fps_start = now

                avg_fps = (sum(self.fps_history) / len(self.fps_history)
                           if self.fps_history else 0.0)
                frame = self.renderer.draw_hud(frame, avg_fps,
                                               self.gender_counter, active_ids)

                # ---- 6. Periodic cleanup ----
                if now - last_cleanup > 30.0:
                    removed = self.gender_counter.cleanup_inactive()
                    self.smoother.purge(set(active_ids))
                    if removed:
                        print(f"Cleaned up {removed} inactive gender entries.")
                    last_cleanup = now

                cv2.imshow("Human Detector V3", frame)

                # ---- Keyboard ----
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._print_stats()
                    break
                elif key == ord("s"):
                    fname = f"screenshot_{screenshot_count}.jpg"
                    cv2.imwrite(fname, frame)
                    print(f"Saved {fname}")
                    screenshot_count += 1
                elif key == ord("r"):
                    self._reset()
                elif key == ord("t"):
                    self._print_stats()
                elif key == ord("l"):
                    self.renderer.show_landmarks = not self.renderer.show_landmarks

        except KeyboardInterrupt:
            print("\nInterrupted.")
            self._print_stats()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.face_detector.close()
            self.landmark_det.close()
            print("Goodbye.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_analyses(self, frame: np.ndarray,
                         tracked: dict, now: float) -> None:
        """
        Refresh stale cache entries with fresh inference results.
        Updates gender_counter exactly once per face per cycle.
        """
        for face_id, face_data in tracked.items():
            cached = self.analysis_cache.get(face_id)
            if cached and (now - cached[1]) < CACHE_SECONDS:
                continue   # still fresh

            bbox = face_data["bbox"]

            # Run both models
            attr   = self.attr_predictor.predict(frame, bbox)
            emo    = self.emotion_det.predict(frame, bbox)

            raw = {
                "gender":    attr["gender"],
                "age":       attr["age"],
                "ethnicity": attr["ethnicity"],
                "emotion":   emo["emotion"],
            }

            # Temporal smoothing
            smoothed = self.smoother.update(face_id, raw)

            # Update gender counter (exactly once per analysis cycle)
            self.gender_counter.update_gender(face_id, smoothed["gender"])

            # Update aggregate stats
            for k in ("gender", "age", "ethnicity", "emotion"):
                self.attr_stats[k][smoothed[k]] += 1

            self.analysis_cache[face_id] = (smoothed, now)

    def _reset(self) -> None:
        self.tracker.reset()
        self.gender_counter.reset()
        self.smoother        = TemporalSmoother()
        self.analysis_cache  = {}
        for c in self.attr_stats.values():
            c.clear()
        print("Reset complete.")

    def _print_stats(self) -> None:
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)

        stats = self.gender_counter.get_stats_dict(
            active_ids=self.tracker.active_ids
        )
        print(f"\nPeople (current)  : {stats['current_male']}M / "
              f"{stats['current_female']}F")
        print(f"People (lifetime) : {stats['total_male']}M / "
              f"{stats['total_female']}F  ({stats['total_unique']} unique)")
        if stats["uncertain"]:
            print(f"Uncertain         : {stats['uncertain']}")

        for label, counter in self.attr_stats.items():
            print(f"\n{label.capitalize()}:")
            for val, cnt in counter.most_common(8):
                print(f"  {val}: {cnt}")

        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    detector = HumanDetector()
    detector.run()


if __name__ == "__main__":
    main()
