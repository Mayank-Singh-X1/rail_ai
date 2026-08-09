"""
FPS Benchmark for Railway Surveillance AI Pipeline
Tests actual CPU performance with synthetic frames to measure throughput.
Run:  python benchmark_fps.py
"""
import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'railway-surveillance-ai'))

print("Loading pipeline (one-time model init)...")
from railway_surveillance_ai.pipeline import UnifiedPipeline
pipeline = UnifiedPipeline()

# Create synthetic test frames (640x480 BGR, like a real webcam)
FRAME_W, FRAME_H = 640, 480
N_FRAMES = 50  # Run 50 frames

print(f"\n=== FPS Benchmark ({N_FRAMES} frames at {FRAME_W}x{FRAME_H}) ===\n")

# Warm-up: first 5 frames let YOLO JIT-compile its kernels
print("Warming up (5 frames)...")
dummy = np.random.randint(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
for _ in range(5):
    pipeline.process_frame(dummy.copy(), enable_criminal=False, enable_worker=False,
                           enable_cleanliness=False)

print("Benchmarking...")
timings = []
for i in range(N_FRAMES):
    # Slightly vary the frame so YOLO doesn't cache
    frame = np.random.randint(60, 200, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    pipeline.process_frame(frame, enable_criminal=False, enable_worker=False,
                           enable_cleanliness=False, enable_anomaly=False)
    dt = time.perf_counter() - t0
    timings.append(dt)
    if (i + 1) % 10 == 0:
        print(f"  Frame {i+1:3d}: {dt*1000:.1f} ms  ({1/dt:.1f} FPS)")

avg_ms = np.mean(timings) * 1000
min_ms = np.min(timings) * 1000
max_ms = np.max(timings) * 1000
avg_fps = 1.0 / np.mean(timings)
p90_fps = 1.0 / np.percentile(timings, 90)

print(f"""
====================================================
  RESULTS
====================================================
  Average frame time:  {avg_ms:.1f} ms
  Min / Max:           {min_ms:.1f} ms / {max_ms:.1f} ms
  Average FPS:         {avg_fps:.1f} FPS
  P90 FPS (stable):    {p90_fps:.1f} FPS
====================================================
""")

if avg_fps >= 30:
    print("✅ TARGET MET: 30+ FPS on CPU!")
elif avg_fps >= 15:
    print(f"⚠️  Partial: {avg_fps:.1f} FPS. Still usable but below 30 FPS target.")
else:
    print(f"❌  {avg_fps:.1f} FPS — further optimizations needed.")
