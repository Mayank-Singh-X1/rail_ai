import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

"""
Main Entry Point for Railway Surveillance AI System
"""
import argparse
import sys
import os
import cv2

def main():
    parser = argparse.ArgumentParser(description="Railway Surveillance AI System Launcher")
    parser.add_argument("--dashboard", action="store_true", help="Launch interactive Gradio dashboard")
    parser.add_argument("--video", type=str, help="Process specified video file")
    parser.add_argument("--stream", type=str, nargs="?", const="0", help="Stream live camera (source ID or RTSP URL)")
    parser.add_argument("--output", type=str, default="output_surveillance.mp4", help="Output path for processed video")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    
    args = parser.parse_args()

    if args.test:
        import pytest
        sys.exit(pytest.main(["tests/test_modules.py"]))

    if args.dashboard:
        print("Launching Gradio Dashboard...")
        from dashboard.app import create_dashboard
        demo = create_dashboard()
        demo.launch(share=False)
    elif args.video:
        print(f"Processing video: {args.video}")
        from pipeline import process_video
        process_video(args.video, output_path=args.output)
    elif args.stream:
        print(f"Streaming live camera source: {args.stream}")
        from pipeline import stream_live_camera
        gen = stream_live_camera(source=args.stream)
        for frame in gen:
            cv2.imshow("Railway Surveillance Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        print("No operation specified. Defaulting to launching Dashboard UI.")
        from dashboard.app import create_dashboard
        demo = create_dashboard()
        demo.launch(share=False)

if __name__ == "__main__":
    main()
