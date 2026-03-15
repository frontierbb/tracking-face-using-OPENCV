"""
emotion_detector.py
-------------------
8-class facial emotion recognition via direct ONNX inference (no hsemotion-onnx package).

Model: enet_b0_8_best_afew.onnx (~15 MB, downloaded automatically by model_manager)
Source: github.com/HSE-asavchenko/face-emotion-recognition

Emotion classes (8): Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise
"""

import cv2
import numpy as np
import model_manager

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

EMOTION_LABELS = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise",
]

# Preprocessing constants (based on hsemotion-onnx training)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EmotionDetector:
    """
    Direct ONNX inference for HSEmotion.

    Parameters
    ----------
    min_confidence : float
        Predictions below this softmax probability are reported as Neutral.
    """

    def __init__(self, min_confidence: float = 0.45):
        if not _ORT_AVAILABLE:
            raise ImportError("onnxruntime is not installed. Run: pip install onnxruntime")

        self.min_confidence = min_confidence
        model_path = model_manager.get_path("emotion")

        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        print(f"[EmotionDetector] HSEmotion ONNX loaded ({model_path}).")

    def predict(self, frame: np.ndarray, bbox: tuple) -> dict:
        """
        Classify emotion for one face.

        Parameters
        ----------
        frame : BGR uint8 full frame
        bbox  : (x, y, w, h)

        Returns
        -------
        dict: emotion (str), confidence (float), scores (dict)
        """
        face = self._crop_and_preprocess(frame, bbox)
        if face is None:
            return self._neutral()

        try:
            outputs = self._session.run(None, {self._input_name: face})
            probs = self._softmax(outputs[0][0])  # Assume single output with shape (1, 8)

            max_idx = int(np.argmax(probs))
            confidence = float(probs[max_idx])
            emotion = EMOTION_LABELS[max_idx]

            if confidence < self.min_confidence:
                emotion = "Neutral"

            scores = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}

            return {
                "emotion": emotion,
                "confidence": confidence,
                "scores": scores,
            }
        except Exception as exc:
            print(f"[EmotionDetector] Inference error: {exc}")
            return self._neutral()

    def predict_batch(self, frame: np.ndarray, bboxes: list) -> list:
        return [self.predict(frame, bbox) for bbox in bboxes]

    def _crop_and_preprocess(self, frame: np.ndarray, bbox: tuple):
        """Crop with 10% padding, resize to 224x224, normalize to NCHW float32."""
        try:
            x, y, w, h = bbox
            fh, fw = frame.shape[:2]
            pad = int(max(w, h) * 0.10)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(fw, x + w + pad)
            y2 = min(fh, y + h + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
            normalized = (resized - _MEAN) / _STD
            return normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    @staticmethod
    def _neutral() -> dict:
        scores = {lbl: 0.0 for lbl in EMOTION_LABELS}
        scores["Neutral"] = 1.0
        return {"emotion": "Neutral", "confidence": 1.0, "scores": scores}