# Phone Usage Detection System

A computer vision system that detects and analyzes phone usage in videos using three different YOLO-based approaches with increasing accuracy and sophistication.

## 🚀 Features

- **Real-time phone usage detection** in video streams
- **Three different detection approaches** with progressive accuracy improvements
- **Audio preservation** in output videos using FFmpeg
- **Detailed analytics** with frame-by-frame usage statistics
- **Visual feedback** with bounding boxes and usage indicators
- **JSON summary reports** for each processed video

## 📁 Project Structure

```
phone-usage-detector/
├── Detection_code.py          # v1.0 - Basic object detection approach
├── Pose_Code.py              # v2.0 - Pose estimation approach  
├── Segmentation_Code.py      # v3.0 - Segmentation approach (recommended)
├── requirements.txt          # Python dependencies
├── Code_Version_Report.pdf   # Detailed technical documentation
└── Pose_code_Output/         # Output directory for processed videos
    ├── Pose_code_Output/     # Pose model outputs
    └── Segmentation_code_Output/  # Segmentation model outputs
```

## 🔧 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd phone-usage-detector
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install FFmpeg** (for audio processing)
   - Windows: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
   - Linux: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`

## 🎯 Usage

### Quick Start
Run the latest segmentation model (v3.0 - recommended):
```bash
python Segmentation_Code.py
```

### All Versions
```bash
# v1.0 - Detection Model
python Detection_code.py

# v2.0 - Pose Model  
python Pose_Code.py

# v3.0 - Segmentation Model (Best accuracy)
python Segmentation_Code.py
```

**Note**: Update the `INPUT_VIDEO` path in each file before running.

## 📊 Model Versions Comparison

| Version | Model | Approach | Accuracy | Performance | Use Case |
|---------|-------|----------|----------|-------------|----------|
| **v1.0** | yolo11m.pt | Object Detection + Motion | Basic | Fastest | Quick detection |
| **v2.0** | yolo11m.pt + yolo11m-pose.pt | Pose Estimation | Good | Medium | Balanced accuracy |
| **v3.0** | yolo11m-seg.pt | Segmentation Masks | Best | Good | Highest precision |

### v1.0 - Detection Model (`Detection_code.py`)
- **Approach**: Detects phones and people using YOLO object detection
- **Logic**: Checks overlap ratio + motion detection in phone bounding box
- **Strengths**: Fast processing, simple implementation
- **Limitations**: May misclassify phones near people but not in use

### v2.0 - Pose Model (`Pose_Code.py`) 
- **Approach**: Phone detection + human wrist keypoint detection
- **Logic**: Checks if wrist keypoints fall inside phone bounding boxes
- **Strengths**: More accurate than v1.0, reduces false positives
- **Limitations**: Requires two models, fails when hands are occluded

### v3.0 - Segmentation Model (`Segmentation_Code.py`) ⭐
- **Approach**: Pixel-level segmentation masks for person and phone
- **Logic**: Checks pixel-level overlap between phone and person masks
- **Strengths**: Highest accuracy, pixel-perfect detection
- **Limitations**: Computationally intensive

## 📈 Output

Each model generates:
- **Processed video** with visual annotations
- **JSON summary** with detailed analytics:
  ```json
  {
    "total_frames": 1500,
    "frames_with_phone_usage": 450,
    "phone_usage_frame_percent": 30.0,
    "processing_seconds": 45.2,
    "phone_usage_timestamps": [1.2, 3.4, 5.6, ...]
  }
  ```

## 🎨 Visual Indicators

- **Red boxes**: Active phone usage detected
- **Green boxes**: Phone detected but not in use
- **Red circles**: Wrist keypoints (v2.0 only)
- **Usage counter**: Real-time phone activity count

## ⚙️ Configuration

Key parameters you can adjust:

```python
CONF_THRESH = 0.5      # Detection confidence threshold
IOU_THRESH = 0.45      # IoU threshold for NMS
PHONE_INSIDE_RATIO = 0.25  # Phone-person overlap ratio (v1.0)
PADDING = 0.5          # Bounding box padding (v2.0)
```

## 🔧 Technical Requirements

- **Python**: 3.8+
- **OpenCV**: 4.5.0+
- **Ultralytics**: 8.0.0+
- **NumPy**: 1.21.0+
- **FFmpeg**: Latest version
- **GPU**: Recommended for faster processing

## 📝 Dependencies

```
opencv-python>=4.5.0
ultralytics>=8.0.0
numpy>=1.21.0
ffmpeg-python>=0.2.0
```

## 🚨 Important Notes

- Update `INPUT_VIDEO` path in each script before running
- Ensure FFmpeg is installed for audio processing
- YOLO models will be downloaded automatically on first run
- Press 'q' during video playback to stop processing early

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Recommended**: Use `Segmentation_Code.py` (v3.0) for best accuracy in phone usage detection.
