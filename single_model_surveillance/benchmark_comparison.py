import os
import sys

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_path)

import time
import numpy as np

from pipeline import SingleModelPipeline
from utils.analytics import AnalyticsDashboard

def run_fps_benchmark(num_frames=30):
    print("\n=======================================================")
    print("  FPS & Latency Benchmark: Concurrent vs Isolated Mode")
    print("=======================================================")
    print(f"Benchmarking across {num_frames} test frames...\n")

    pipeline = SingleModelPipeline()
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 1. Benchmark Concurrent All Mode
    t0 = time.time()
    for _ in range(num_frames):
        pipeline.process_frame(dummy_frame, mode="all")
    t_all = time.time() - t0
    fps_all = num_frames / (t_all + 1e-6)

    # 2. Benchmark Single-Model Mode (Crowd)
    t0 = time.time()
    for _ in range(num_frames):
        pipeline.process_frame(dummy_frame, mode="crowd")
    t_single = time.time() - t0
    fps_single = num_frames / (t_single + 1e-6)

    speedup = fps_single / (fps_all + 1e-6)

    print("📊 BENCHMARK RESULTS:")
    print("-------------------------------------------------------")
    print(f"  1. Concurrent (All 6 Models):  {fps_all:.1f} FPS  ({(1000/fps_all):.1f} ms/frame)")
    print(f"  2. Isolated (Single-Model Mode): {fps_single:.1f} FPS  ({(1000/fps_single):.1f} ms/frame)")
    print(f"  🚀 Performance Gain:             {speedup:.2f}x Speedup (Zero Lag)")
    print("-------------------------------------------------------\n")

    chart_file = AnalyticsDashboard.plot_fps_comparison([fps_all], [fps_single])
    print(f"✅ Benchmark chart saved to: {chart_file}\n")
    return fps_all, fps_single

if __name__ == "__main__":
    run_fps_benchmark()
