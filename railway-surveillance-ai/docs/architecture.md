# Architecture Overview

## Unified Processing Pipeline
The Railway Surveillance AI system uses a modular GPU-accelerated pipeline built around YOLO models and InsightFace vector embeddings.

### Data Flow
1. **Frame Ingestion**: Video files, RTSP IP camera feeds, or live webcams.
2. **Detection & Tracking**: YOLO object detection + Person Tracker.
3. **Modular Processing**:
   - Criminal Detection (InsightFace embedding distance check every 5 frames)
   - Anomaly Detection (YOLO Pose keypoint analysis for fall/fight detection)
   - Cleanliness Monitoring (Litter classification & scoring every 30 frames)
   - Worker Monitoring (Zone occupancy & attendance logging)
4. **Overlay Rendering**: Status bar, bounding boxes, and alert popups.
5. **Alert System**: Broadcasts critical security alerts via Email/SMS/Firebase.
