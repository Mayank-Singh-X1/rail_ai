"""
Direct FPS benchmark — no import path gymnastics.
Run from d:\\SIHackathon:  python bench.py
"""
import sys, os, time
import numpy as np
import cv2

os.chdir(os.path.join(os.path.dirname(__file__), 'railway-surveillance-ai'))
sys.path.insert(0, os.getcwd())

print("Loading pipeline (one-time model init, ~10s) ...")
from pipeline import UnifiedPipeline
pipe = UnifiedPipeline()

FRAME_W, FRAME_H = 640, 480
N_FRAMES = 60

print(f"\n=== FPS Benchmark: {N_FRAMES} frames @ {FRAME_W}x{FRAME_H} ===\n")

dummy = np.ones((FRAME_H, FRAME_W, 3), dtype=np.uint8) * 128

# Warm-up (let ByteTrack initialise)
print("Warming up (5 frames) ...")
for _ in range(5):
    pipe.process_frame(dummy.copy(), enable_criminal=False,
                       enable_worker=False, enable_cleanliness=False,
                       enable_anomaly=False)

print("Benchmarking ...\n")
timings = []
for i in range(N_FRAMES):
    frame = np.random.randint(50, 200, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    pipe.process_frame(frame, enable_criminal=False, enable_worker=False,
                       enable_cleanliness=False, enable_anomaly=False)
    dt = time.perf_counter() - t0
    timings.append(dt)
    if (i + 1) % 10 == 0:
        print(f"  Frame {i+1:3d}: {dt*1000:6.1f} ms  ({1/dt:6.1f} FPS)")

avg_ms  = np.mean(timings)  * 1000
min_fps = 1.0 / np.max(timings)
avg_fps = 1.0 / np.mean(timings)
p90_fps = 1.0 / np.percentile(timings, 90)

print(f"""
====================================================
  RESULTS (Tracking-Only Fast Path)
====================================================
  Average latency:  {avg_ms:.1f} ms / frame
  Average FPS:      {avg_fps:.1f}
  P90 stable FPS:   {p90_fps:.1f}
  Min FPS (worst):  {min_fps:.1f}
====================================================
""")

if avg_fps >= 30:
    print("SUCCESS: 30+ FPS on CPU achieved!")
elif avg_fps >= 15:
    print(f"PARTIAL: {avg_fps:.1f} FPS. Usable for demo, below 30 FPS target.")
else:
    print(f"NEED MORE: {avg_fps:.1f} FPS. Reduce YOLO input size further.")
