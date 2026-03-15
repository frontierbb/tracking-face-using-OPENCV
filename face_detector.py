"""
face_detector.py
----------------
Face detection using MediaPipe BlazeFace.

Compatible with MediaPipe 0.10.x (including 0.10.32).
Uses direct module imports instead of mp.solutions which was
restructured in newer MediaPipe versions.
"""

import cv2
import numpy as np

try:
    from mediapipe.python.solutions import face_detection as mp_face_detection
    _MP_AVAILABLE = True
except ImportError:
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        _MP_AVAILABLE = True
    except Exception:
        _MP_AVAILABLE = False


class FaceDetector:
    """
    Wraps MediaPipe FaceDetection (BlazeFace).

    Parameters
    ----------
    min_confidence : float
        Detections below this threshold are discarded.
    model_selection : int
        0 = short-range (~2 m), 1 = full-range (~5 m, recommended).
    max_faces : int
        Hard cap on faces returned per frame.
    """

    def __init__(self,
                 min_confidence: float = 0.65,
                 model_selection: int = 1,
                 max_faces: int = 10):

        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe is not installed. Run: pip install mediapipe"
            )

        self.max_faces = max_faces
        self._detector = mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence,
        )
        print(f"[FaceDetector] MediaPipe BlazeFace ready "
              f"(model={model_selection}, min_conf={min_confidence}).")

    def detect(self, frame: np.ndarray) -> list:
        """
        Detect faces in frame (BGR uint8).
        Returns list of dict {"bbox": (x,y,w,h), "score": float},
        sorted by score descending.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        if not results.detections:
            return []

        faces = []
        for det in results.detections:
            score = det.score[0] if det.score else 0.0
            bb    = det.location_data.relative_bounding_box

            x  = max(0, int(bb.xmin * w))
            y  = max(0, int(bb.ymin * h))
            bw = int(bb.width  * w)
            bh = int(bb.height * h)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            if bw < 20 or bh < 20:
                continue

            faces.append({"bbox": (x, y, bw, bh), "score": score})

        faces.sort(key=lambda d: d["score"], reverse=True)
        return faces[: self.max_faces]

    def close(self) -> None:
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()