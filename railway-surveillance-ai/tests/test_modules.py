import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

"""
Unit Tests for Railway Surveillance AI System
"""
import unittest
import os
import numpy as np
import cv2

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import RailwaySurveillanceSystem, UnifiedPipeline
from modules import (
    CrowdAnalyzer,
    CriminalDetector,
    AnomalyDetector,
    CleanlinessMonitor,
    WorkerMonitor,
    PersonTracker,
    AlertSystem
)


class TestRailwaySurveillance(unittest.TestCase):

    def setUp(self):
        self.system = RailwaySurveillanceSystem()

    def test_system_initialization(self):
        self.assertIsNotNone(self.system)
        self.assertIsInstance(self.system.zones, dict)
        self.assertIsInstance(self.system.criminal_db, dict)
        self.assertIsInstance(self.system.worker_db, dict)

    def test_add_zone(self):
        self.system.add_zone(
            "track_safety_zone",
            [[100, 400], [500, 400], [500, 600], [100, 600]],
            "restricted"
        )
        self.assertIn("track_safety_zone", self.system.zones)
        self.assertEqual(self.system.zones["track_safety_zone"]["type"], "restricted")

    def test_crowd_analyzer(self):
        analyzer = CrowdAnalyzer(self.system)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        count, detections = analyzer.count_people(dummy_frame)
        self.assertIsInstance(count, int)
        level, color = analyzer.get_crowd_level(count)
        self.assertIn(level, ["LOW", "MEDIUM", "HIGH / OVERCROWDING"])

    def test_anomaly_detector(self):
        detector = AnomalyDetector(self.system)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, anomalies = detector.detect_anomalies(dummy_frame)
        self.assertIsNotNone(annotated)
        self.assertIsInstance(anomalies, list)

    def test_cleanliness_monitor(self):
        monitor = CleanlinessMonitor(self.system)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        score = monitor.get_cleanliness_score(dummy_frame)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_worker_monitor(self):
        monitor = WorkerMonitor(self.system)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, present, absent = monitor.check_attendance(dummy_frame)
        self.assertIsNotNone(annotated)
        self.assertIsInstance(present, list)
        self.assertIsInstance(absent, list)

    def test_person_tracker(self):
        tracker = PersonTracker(self.system)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, active_count = tracker.track_persons(dummy_frame)
        self.assertIsNotNone(annotated)
        self.assertIsInstance(active_count, int)

    def test_alert_system(self):
        alert_sys = AlertSystem()
        alert_sys.send_alert({"type": "High Crowd", "details": "Platform 1 is crowded", "severity": "HIGH"})
        self.assertGreater(len(alert_sys.alert_log), 0)
        summary = alert_sys.get_alert_summary()
        self.assertIn("HIGH", summary["by_severity"])

    def test_unified_pipeline(self):
        pipe = UnifiedPipeline()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pipe.process_frame(dummy_frame)
        self.assertIn("frame", results)
        self.assertIn("crowd_count", results)
        self.assertIn("cleanliness_score", results)
        self.assertIn("criminals_found", results)
        self.assertIn("anomalies", results)


if __name__ == "__main__":
    unittest.main()
