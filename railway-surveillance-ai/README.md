# Railway Surveillance AI 🚄🤖

An end-to-end multi-camera AI surveillance system designed for railway platforms, passenger safety, crowd monitoring, automated cleanliness tracking, criminal identification, and worker attendance monitoring.

## 📁 Directory Structure
```
railway-surveillance-ai/
├── models/
│   ├── yolov8n.pt
│   ├── yolov8n-pose.pt
│   └── custom_cleanliness.pt
├── modules/
│   ├── __init__.py
│   ├── crowd_analyzer.py
│   ├── criminal_detector.py
│   ├── anomaly_detector.py
│   ├── cleanliness_monitor.py
│   ├── worker_monitor.py
│   ├── person_tracker.py
│   └── alert_system.py
├── database/
│   ├── criminals/          # Criminal face images
│   ├── workers/            # Worker face images
│   └── embeddings.pkl      # Saved face embeddings
├── config/
│   ├── zones.json          # Zone configurations
│   ├── cameras.json        # Camera settings
│   └── thresholds.json     # Alert thresholds
├── dashboard/
│   ├── app.py              # Gradio dashboard
│   └── assets/
├── notebooks/
│   ├── 01_setup.ipynb
│   ├── 02_training.ipynb
│   └── 03_demo.ipynb
├── utils/
│   ├── visualization.py
│   └── analytics.py
├── tests/
│   └── test_modules.py
├── main.py                 # Entry point
├── pipeline.py             # Unified pipeline
├── requirements.txt
├── README.md
└── docs/
    ├── architecture.md
    ├── api_docs.md
    └── presentation.pptx
```

## 🚀 Key Modules
1. **Crowd Analyzer**: Real-time crowd count and density estimation.
2. **Criminal Detector**: Facial recognition and vector matching using InsightFace.
3. **Anomaly Detector**: Fall detection, fight detection, and restricted area intrusion alerts.
4. **Cleanliness Monitor**: Automated litter detection and cleanliness scoring using YOLO models.
5. **Worker Monitor**: Face recognition-based attendance and zone occupancy tracking.
6. **Person Tracker**: Multi-object tracking with line crossing analytics.
7. **Alert System**: Multi-channel notification pipeline (Email, SMS, Firebase).

## 🏃 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Launch Dashboard
```bash
python main.py --dashboard
```

### Process Video
```bash
python main.py --video path/to/video.mp4
```

### Run Tests
```bash
python main.py --test
```
