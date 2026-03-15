"""
landmark_detector.py
--------------------
468-point facial landmark detection using MediaPipe Face Mesh.

Compatible with MediaPipe 0.10.x (including 0.10.32).
Uses direct module imports instead of mp.solutions.
"""

import cv2
import numpy as np

try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    _MP_AVAILABLE = True
except ImportError:
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        _MP_AVAILABLE = True
    except Exception:
        _MP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Landmark index groups
# ---------------------------------------------------------------------------

EYE_LEFT_POINTS  = [33, 160, 158, 133, 153, 144]
EYE_RIGHT_POINTS = [362, 385, 387, 263, 373, 380]

EYEBROW_LEFT  = [70, 63, 105, 66, 107]
EYEBROW_RIGHT = [336, 296, 334, 293, 300]

MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
               291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
               308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

NOSE_TIP    = 1
NOSE_BRIDGE = 6
NOSTRIL_L   = 129
NOSTRIL_R   = 358
JAW_CHIN    = 152


class LandmarkDetector:
    """
    Wraps MediaPipe FaceMesh for 468-point landmark detection.

    Parameters
    ----------
    max_faces                : int
    min_detection_confidence : float
    min_tracking_confidence  : float
    refine_landmarks         : bool
        Adds iris landmarks (indices 468-477) when True.
    """

    def __init__(self,
                 max_faces: int = 10,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float  = 0.5,
                 refine_landmarks: bool = True):

        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe is not installed. Run: pip install mediapipe"
            )

        self._mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.refine = refine_landmarks
        print(f"[LandmarkDetector] MediaPipe Face Mesh ready "
              f"(max_faces={max_faces}, refine={refine_landmarks}).")

    def get_landmarks(self, frame: np.ndarray, face_bboxes: list) -> list:
        """
        Compute 468-point landmarks for each bounding box.

        Returns list of dict or None, same length as face_bboxes.
        Each dict: {"points": np.ndarray (468,2), "points_3d": np.ndarray (468,3)}
        """
        results  = []
        h_frame, w_frame = frame.shape[:2]

        for (x, y, w, h) in face_bboxes:
            pad = int(max(w, h) * 0.20)
            x1  = max(0, x - pad)
            y1  = max(0, y - pad)
            x2  = min(w_frame, x + w + pad)
            y2  = min(h_frame, y + h + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                results.append(None)
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            out = self._mesh.process(rgb)

            if not out.multi_face_landmarks:
                results.append(None)
                continue

            lm     = out.multi_face_landmarks[0]
            ch, cw = crop.shape[:2]

            pts_2d = np.array(
                [(lm.landmark[i].x * cw + x1,
                  lm.landmark[i].y * ch + y1)
                 for i in range(len(lm.landmark))],
                dtype=np.float32,
            )
            pts_3d = np.array(
                [(lm.landmark[i].x,
                  lm.landmark[i].y,
                  lm.landmark[i].z)
                 for i in range(len(lm.landmark))],
                dtype=np.float32,
            )

            results.append({"points": pts_2d, "points_3d": pts_3d})

        return results

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def ear(points: np.ndarray, indices: list) -> float:
    """Eye Aspect Ratio using 6 landmark indices."""
    p = points[indices]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return float((A + B) / (2.0 * C + 1e-6))


def mouth_aspect_ratio(points: np.ndarray) -> tuple:
    """Return (width, height, MAR) of the mouth."""
    outer  = points[MOUTH_OUTER]
    width  = float(np.linalg.norm(outer[0]  - outer[10]))
    height = float(np.linalg.norm(outer[14] - outer[3]))
    return width, height, height / (width + 1e-6)