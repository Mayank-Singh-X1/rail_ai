import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

"""
Main Entry Point for Isolated Single-Model Railway Surveillance AI System
"""
import argparse
import sys
import os
import cv2
import numpy as np

def run_tests():
    """Test single-model execution across all modes with a synthetic frame."""
    print("\n========================================")
    print("  Testing Isolated Single-Model Pipeline")
    print("========================================")
    
    from pipeline import pipeline
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    modes = ["crowd", "criminal", "anomaly", "cleanliness", "worker", "weapon", "all"]
    
    for m in modes:
        res = pipeline.process_frame(dummy_frame, mode=m)
        fps = res.get("fps", 0)
        print(f"  > Mode '{m.upper()}': OK | Mode FPS: {fps:.1f}")
        
    print("\n✅ ALL SINGLE-MODEL PIPELINE TESTS PASSED SUCCESSFULLY!\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="Isolated Single-Model Railway Surveillance Launcher")
    parser.add_argument("--crowd", action="store_true", help="Launch Dedicated Crowd Analytics Dashboard (Port 7860)")
    parser.add_argument("--criminal", action="store_true", help="Launch Dedicated Criminal Recognition Dashboard (Port 7861)")
    parser.add_argument("--anomaly", action="store_true", help="Launch Dedicated Anomaly & Fall Detection Dashboard (Port 7862)")
    parser.add_argument("--cleanliness", action="store_true", help="Launch Dedicated Cleanliness Monitoring Dashboard (Port 7863)")
    parser.add_argument("--worker", action="store_true", help="Launch Dedicated Staff & RPF Attendance Dashboard (Port 7864)")
    parser.add_argument("--weapon", action="store_true", help="Launch Dedicated Weapon & Threat Scan Dashboard (Port 7865)")
    parser.add_argument("--dashboard", action="store_true", help="Launch Crowd Analytics dashboard")
    parser.add_argument("--mode", type=str, default="crowd", choices=["crowd", "criminal", "anomaly", "cleanliness", "worker", "weapon", "all"], help="Active AI model mode")
    parser.add_argument("--video", type=str, help="Process specified video file in selected mode")
    parser.add_argument("--stream", type=str, nargs="?", const="0", help="Stream live camera (source ID or RTSP URL)")
    parser.add_argument("--output", type=str, default="output_single_model.mp4", help="Output path for processed video")
    parser.add_argument("--test", action="store_true", help="Run test suite across all AI modes")
    
    args = parser.parse_args()

    if args.test:
        run_tests()
        sys.exit(0)

    if args.crowd:
        print("👥 Launching Dedicated Crowd Analytics Dashboard (Port 7860)...")
        from dashboards.app_crowd import create_crowd_dashboard
        demo = create_crowd_dashboard()
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    elif args.criminal:
        print("🚨 Launching Dedicated Criminal Recognition Dashboard (Port 7861)...")
        from dashboards.app_criminal import create_criminal_dashboard
        demo = create_criminal_dashboard()
        demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
    elif args.anomaly:
        print("⚠️ Launching Dedicated Anomaly & Fall Detection Dashboard (Port 7862)...")
        from dashboards.app_anomaly import create_anomaly_dashboard
        demo = create_anomaly_dashboard()
        demo.launch(server_name="127.0.0.1", server_port=7862, share=False)
    elif args.cleanliness:
        print("🧹 Launching Dedicated Cleanliness Monitoring Dashboard (Port 7863)...")
        from dashboards.app_cleanliness import create_cleanliness_dashboard
        demo = create_cleanliness_dashboard()
        demo.launch(server_name="127.0.0.1", server_port=7863, share=False)
    elif args.worker:
        print("👷 Launching Dedicated Staff Attendance Dashboard (Port 7864)...")
        from dashboards.app_worker import create_worker_dashboard
        demo = create_worker_dashboard()
        demo.launch(server_name="127.0.0.1", server_port=7864, share=False)
    elif args.weapon:
        print("🔫 Launching Dedicated Weapon & Threat Scan Dashboard (Port 7865)...")
        from dashboards.app_weapon import create_weapon_dashboard
        demo = create_weapon_dashboard()
        demo.launch(server_name="127.0.0.1", server_port=7865, share=False)
    elif args.video:
        print(f"Processing video: {args.video} in mode '{args.mode}'...")
        from pipeline import process_video
        process_video(args.video, output_path=args.output, mode=args.mode)
    elif args.stream:
        print(f"Streaming live camera source '{args.stream}' in isolated mode '{args.mode}'...")
        from pipeline import stream_live_camera
        gen = stream_live_camera(source=args.stream, mode=args.mode)
        for frame in gen:
            cv2.imshow("Single-Model Railway Surveillance Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        print("Defaulting to launching Dedicated Crowd Analytics Dashboard (Port 7860)...")
        from dashboards.app_crowd import create_crowd_dashboard
        demo = create_crowd_dashboard()
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

if __name__ == "__main__":
    main()
