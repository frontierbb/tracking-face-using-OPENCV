# tracking-face-using-OPENCV - python 3.11
#  Enhanced Human Detector

Real-time human analysis from a laptop camera using **OpenCV** and **dlib**.  
Detects and displays **Gender · Age · Emotion · Ethnicity** for every face in frame,  
plus live and lifetime people counts.

---

Features

| Feature | Details |
|---|---|
|  Face Detection | dlib frontal detector (primary) · OpenCV Haar Cascade (fallback) |
|  Gender | Multi-feature heuristic · DNN model (optional, higher accuracy) |
|  Age | Range prediction `(0-2)` → `(60-100)` via DNN or heuristic |
|  Emotion | Happy · Sad · Angry · Surprised · Neutral — landmark ratios |
|  Ethnicity | Asian · White · Black · Brown · Middle Eastern |
|  People Count | Current frame count + lifetime unique persons |
|  Multi-angle | Frontal + left/right profile detection |
|  Performance | Frame-skip · analysis cache · temporal smoothing |

---

## 🖥️ Requirements

- Python **3.8 – 3.12**
- Laptop or external **webcam**
- (Optional but should install) ~45 MB download for age/gender DNN models

---

##  Python Version & Environment Setup

> **Read this section carefully before installing.**

### 🟢 Python ≤ 3.11 — Standard terminal (no venv needed)

Use your system Python directly from **VS Code integrated terminal** or any terminal:

```bash
# Install dependencies globally
pip install -r requirements.txt
```

> ✅ `/venv` entry in `.gitignore` is **not needed** — you may comment it out.

---

###  Python > 3.11 — Virtual environment required

Python 3.12+ enforces PEP 668 (externally managed environments).  
You **must** create a virtual environment first:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
#    Windows (PowerShell)
.\venv\Scripts\Activate.ps1
#    Windows (CMD)
venv\Scripts\activate.bat
#    macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> ✅ The `/venv` folder is already in `.gitignore` — **do not commit it**.

---

## 🛠️ Installing dlib (extra steps)

dlib requires **CMake** and a **C++ compiler**:

| OS | Command |
|---|---|
| Windows | Install [CMake](https://cmake.org/download/) + [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), then `pip install dlib` |
| macOS | `brew install cmake` then `pip install dlib` |
| Linux | `sudo apt install cmake build-essential` then `pip install dlib` |

---

## Project Structure

```
human-detector/
├── code_source.py                      # Main detector (EnhancedDetectorV2)
├── gender_counter.py                   # GenderCounter helper class
├── requirements.txt
├── .gitignore
├── README.md
│
├── shape_predictor_68_face_landmarks.dat   # ⬇ Download separately (99 MB)
│                                           #   http://dlib.net/files/
│
├── age_net.caffemodel                  # ⬇ Auto-downloaded on first run
├── age_deploy.prototxt                 #   when pressing [a] key
├── gender_net.caffemodel
└── gender_deploy.prototxt
```

> **Model files are excluded from Git** (see `.gitignore`).  
> `shape_predictor_68_face_landmarks.dat` must be downloaded manually and placed in the project root.

---

##  Running the Detector

```bash
python code_source.py
```

###  Keyboard Controls

| Key | Action |
|---|---|
| `q` | Quit and show final statistics |
| `s` | Save screenshot |
| `a` | Download & load DNN age/gender models |
| `r` | Reset all counters and tracking |
| `t` | Print statistics to terminal |
| `h` | Help |

---

## 📥 Downloading the Landmark Model

The 68-point landmark model is required for best results.  
Download **`shape_predictor_68_face_landmarks.dat`** from:

```
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

Extract and place `shape_predictor_68_face_landmarks.dat` in the **project root folder**.

---

##  Suggestions & Upgrade Roadmap

###  Short-term improvements

- **Replace heuristic gender/age with a dedicated deep learning model**  
  e.g. [InsightFace](https://github.com/deepinsight/insightface) or [DeepFace](https://github.com/serengil/deepface) — significant accuracy boost.

- **Upgrade emotion detection**  
  Train or use a pre-trained CNN on FER-2013 dataset instead of landmark ratios.

- **Improve ethnicity classification**  
  Current feature-based approach is limited; a fine-tuned ResNet/EfficientNet trained on a diverse dataset is strongly recommended.

- **Add face recognition (re-ID)**  
  Use face embeddings (dlib `face_recognition` or ArcFace) so returning persons get the same ID across sessions.

###  Mid-term upgrades

- **Switch to a YOLO-based face detector** (e.g. YOLOv8-face) for faster, more robust multi-face detection including side profiles and occluded faces.

- **GPU acceleration** — enable CUDA backend for OpenCV DNN (`cv2.dnn.DNN_BACKEND_CUDA`) for 5-10× speed-up.

- **Logging & analytics dashboard** — save per-session CSVs and visualise trends with a simple Streamlit or Dash app.

- **Multi-camera support** — extend `run()` to accept a camera index argument and support multiple simultaneous streams.

###  Long-term / production

- **Edge deployment** — convert models to ONNX or TensorRT for Jetson Nano / Raspberry Pi.

- **Privacy compliance** — add blur/mask toggle for GDPR-compliant deployments.

- **REST API** — wrap detector in FastAPI so results can be consumed by other services.

- **Unit & integration tests** — expand the existing `test_gender_counter()` pattern to cover all detection modules.

---

##  Troubleshooting

| Problem | Fix |
|---|---|
| `dlib` install fails | Install CMake and C++ build tools first (see above) |
| Camera not opening | Check another app isn't using it; try `VideoCapture(1)` |
| Low FPS | Increase `frame_skip` value in `__init__` |
| dlib landmark file missing | Download `.dat` file and place in project root |
| Models not found on `a` key | Check internet connection; SSL errors may require VPN |

---

##  License

This project is released for educational and research purposes.  
Pre-trained model weights retain their original licences.
