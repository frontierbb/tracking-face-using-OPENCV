# Enhanced Human Detector (v3 - Modular Stack)

A real-time facial analysis system built with Python, featuring modular components for face detection, tracking, landmark extraction, attribute prediction (gender, age, ethnicity), emotion recognition, and statistics tracking. Designed for webcam input with optimized performance for CPU-based inference.

## Features

- **Face Detection**: MediaPipe BlazeFace for robust face detection
- **Face Tracking**: Centroid-based tracking with stable IDs across frames
- **Landmark Detection**: 468-point facial landmarks using MediaPipe Face Mesh
- **Attribute Prediction**: Gender, age group, and ethnicity classification via FairFace ONNX model
- **Emotion Recognition**: 8-class emotion detection using HSEmotion ONNX model
- **Statistics Tracking**: Real-time and lifetime gender counts
- **Temporal Smoothing**: Majority-vote smoothing for stable attribute predictions
- **Performance Optimized**: Configurable frame skipping and caching for smooth FPS
- **Modular Design**: Clean separation of concerns for easy maintenance and extension

## Requirements

- **Python**: 3.8 - 3.11 (if using > 3.11, please use venv)
- **Operating System**: Windows, macOS, or Linux
- **Hardware**: Webcam required for live detection
- **Dependencies**: See `requirements.txt`

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/enhanced-human-detector.git
   cd enhanced-human-detector
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download models** (automatic on first run):
   - FairFace ONNX model (~85 MB)
   - HSEmotion ONNX model (~15 MB)
   
   Models are downloaded automatically when you run the application. If download fails, you can manually download from the source URLs in `model_manager.py`.

## Usage

### Running the Detector

```bash
python main.py
```

### Controls

- **q**: Quit the application
- **s**: Save screenshot
- **r**: Reset all state (tracking, statistics)
- **t**: Print statistics to terminal
- **l**: Toggle landmark overlay

### Configuration

Edit the constants in `main.py` to adjust:

- `CAMERA_INDEX`: Webcam device index
- `FRAME_WIDTH/HEIGHT`: Resolution
- `TARGET_FPS`: Target frame rate
- `ANALYSIS_EVERY_N_FRAMES`: Inference frequency (higher = faster but less frequent updates)
- `CACHE_SECONDS`: Cache validity duration

## Project Structure

```
├── main.py                 # Main application entry point
├── face_detector.py        # MediaPipe BlazeFace wrapper
├── landmark_detector.py    # MediaPipe Face Mesh wrapper
├── attribute_predictor.py  # FairFace ONNX inference
├── emotion_detector.py     # HSEmotion ONNX inference
├── face_tracker.py         # Centroid-based face tracking
├── gender_counter.py       # Gender statistics tracking
├── model_manager.py        # Model download and management
├── requirements.txt        # Python dependencies
├── models/                 # Downloaded ONNX models
│   ├── fairface.onnx
│   └── enet_b0_8_best_afew.onnx
└── __pycache__/           # Python bytecode cache
```

## Dependencies

- `opencv-python`: Computer vision operations
- `mediapipe`: Face detection and landmark extraction
- `onnxruntime`: ONNX model inference
- `numpy`: Numerical operations
- `Pillow`: Image processing (if needed)

## Model Sources

- **FairFace**: [yakhyo/fairface-onnx](https://github.com/yakhyo/fairface-onnx)
- **HSEmotion**: [HSE-asavchenko/face-emotion-recognition](https://github.com/HSE-asavchenko/face-emotion-recognition)

## Performance Notes

- Optimized for CPU inference
- Frame skipping reduces computational load while maintaining smooth tracking
- Temporal smoothing prevents attribute flickering
- Models are cached locally after first download

## Troubleshooting

### Common Issues

1. **MediaPipe Import Error**: Ensure MediaPipe is installed: `pip install mediapipe`
2. **ONNX Runtime Error**: Install ONNX Runtime: `pip install onnxruntime`
3. **Model Download Failure**: Check internet connection or manually download models
4. **Camera Not Found**: Verify camera index in `main.py` (usually 0 for built-in webcam)

### Performance Tuning

- Increase `ANALYSIS_EVERY_N_FRAMES` for better FPS (at cost of slower attribute updates)
- Reduce `FRAME_WIDTH/HEIGHT` for lower resolution processing
- Adjust `CACHE_SECONDS` based on desired responsiveness

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for excellent computer vision libraries
- FairFace and HSEmotion researchers for their models
- Open-source community for ONNX and related tools