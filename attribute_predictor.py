"""
attribute_predictor.py
----------------------
Gender, age-group and ethnicity via FairFace ResNet-34 ONNX.

Source model
    github.com/yakhyo/fairface-onnx
    File: fairface.onnx (~85 MB, downloaded automatically by model_manager)

Output order (verified from yakhyo/fairface-onnx onnx_inference.py)
    outputs[0] : race logits   shape (1, 7)
    outputs[1] : gender logits shape (1, 2)
    outputs[2] : age logits    shape (1, 9)

Input preprocessing (FairFace training convention)
    - Crop with 30% padding
    - Resize to 224x224
    - Normalise: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] (ImageNet)
    - Channel order: RGB, layout: NCHW float32
"""

import cv2
import numpy as np
import model_manager

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Label maps  (must match FairFace training order)
# ---------------------------------------------------------------------------

RACE_LABELS = [
    "White", "Black", "Latino", "East Asian",
    "Southeast Asian", "Indian", "Middle Eastern",
]
GENDER_LABELS = ["Male", "Female"]
AGE_LABELS    = ["0-2", "3-9", "10-19", "20-29", "30-39",
                  "40-49", "50-59", "60-69", "70+"]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AttributePredictor:
    """
    FairFace ONNX inference for gender, age and ethnicity.

    Parameters
    ----------
    confidence_threshold : float
        Minimum softmax confidence.  Predictions below this receive a "?"
        marker in the gender label.
    """

    def __init__(self, confidence_threshold: float = 0.55):
        if not _ORT_AVAILABLE:
            raise ImportError(
                "onnxruntime is not installed. Run: pip install onnxruntime"
            )

        self.threshold = confidence_threshold
        model_path = model_manager.get_path("fairface")

        providers = ["CPUExecutionProvider"]
        self._session    = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        print(f"[AttributePredictor] FairFace ONNX loaded ({model_path}).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, frame: np.ndarray, bbox: tuple) -> dict:
        """
        Predict gender, age and ethnicity for one face.

        Parameters
        ----------
        frame : BGR uint8 full frame
        bbox  : (x, y, w, h)

        Returns
        -------
        dict: gender, age, ethnicity, gender_conf, age_conf, ethnicity_conf
        """
        blob = self._prepare_input(frame, bbox)
        if blob is None:
            return self._unknown()

        try:
            outputs = self._session.run(None, {self._input_name: blob})
        except Exception as exc:
            print(f"[AttributePredictor] Inference error: {exc}")
            return self._unknown()

        return self._parse_outputs(outputs)

    def predict_batch(self, frame: np.ndarray, bboxes: list) -> list:
        return [self.predict(frame, bbox) for bbox in bboxes]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_input(self, frame: np.ndarray, bbox: tuple):
        """
        Crop with 30% padding, resize 224x224, normalise to NCHW float32.
        30% padding matches FairFace training convention exactly.
        """
        try:
            x, y, w, h = bbox
            fh, fw = frame.shape[:2]

            pad_x = int(w * 0.30)
            pad_y = int(h * 0.30)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(fw, x + w + pad_x)
            y2 = min(fh, y + h + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
            normed  = (resized - _MEAN) / _STD
            return normed.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        except Exception as exc:
            print(f"[AttributePredictor] Preprocess error: {exc}")
            return None

    def _parse_outputs(self, outputs: list) -> dict:
        """
        Parse FairFace ONNX outputs.
        Order: outputs[0]=race, outputs[1]=gender, outputs[2]=age
        Verified from yakhyo/fairface-onnx onnx_inference.py source.
        """
        def softmax(x):
            e = np.exp(x - x.max())
            return e / e.sum()

        race_probs   = softmax(outputs[0][0])
        gender_probs = softmax(outputs[1][0])
        age_probs    = softmax(outputs[2][0])

        race_idx   = int(np.argmax(race_probs))
        gender_idx = int(np.argmax(gender_probs))
        age_idx    = int(np.argmax(age_probs))

        race_conf   = float(race_probs[race_idx])
        gender_conf = float(gender_probs[gender_idx])
        age_conf    = float(age_probs[age_idx])

        gender_lbl = GENDER_LABELS[gender_idx]
        if gender_conf < self.threshold:
            gender_lbl += "?"

        return {
            "gender":         gender_lbl,
            "age":            AGE_LABELS[age_idx],
            "ethnicity":      RACE_LABELS[race_idx],
            "gender_conf":    gender_conf,
            "age_conf":       age_conf,
            "ethnicity_conf": race_conf,
        }

    @staticmethod
    def _unknown() -> dict:
        return {
            "gender":         "Unknown",
            "age":            "Unknown",
            "ethnicity":      "Unknown",
            "gender_conf":    0.0,
            "age_conf":       0.0,
            "ethnicity_conf": 0.0,
        }