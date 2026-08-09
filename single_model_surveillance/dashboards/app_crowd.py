import sys
import os
import time
import tempfile
import cv2
import numpy as np
import gradio as gr

# Ensure parent directory is in path
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_path)

from pipeline import pipeline


def create_crowd_dashboard():
    def process_webcam(frame):
        if frame is None:
            return None, "⚠️ Waiting for webcam frame..."
        if isinstance(frame, dict):
            frame = frame.get('image', frame.get('composite', None))
        if frame is None:
            return None, "⚠️ Waiting for camera..."

        arr = cv2.cvtColor(np.array(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        res = pipeline.process_frame(arr, mode="crowd")
        out_rgb = cv2.cvtColor(res['frame'], cv2.COLOR_BGR2RGB)
        
        status = f"🟢 **CROWD AI ACTIVE** | Passenger Count: `{res['crowd_count']}` | Density Level: `{res['crowd_level']}` | FPS: `{res['fps']:.1f}`"
        return out_rgb, status

    def process_video_file(video_file):
        if video_file is None:
            return None, "❌ Please upload a video file."
        input_path = video_file if isinstance(video_file, str) else getattr(video_file, "name", str(video_file))
        output_path = os.path.join(tempfile.gettempdir(), "crowd_analytics_output.mp4")

        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        target_w = 1280
        target_h = int(h * (target_w / w)) if w else 720

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

        peak_crowd = 0
        total_frames = 0

        while cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            f_resized = cv2.resize(f, (target_w, target_h))
            res = pipeline.process_frame(f_resized, mode="crowd")
            out.write(res["frame"])
            peak_crowd = max(peak_crowd, res["crowd_count"])
            total_frames += 1

        cap.release()
        out.release()

        report = f"""
### 👥 Platform Crowd Analytics Summary Report

| Metric | Result |
|---|---|
| ⏱️ Total Frames Processed | **{total_frames} frames** |
| 👥 Peak Platform Passengers | **{peak_crowd} passengers** |
| 📍 ByteTrack Object Tracking | **Active** |
| 📍 Density Status | **{"🔴 High Congestion" if peak_crowd > 20 else "🟢 Safe / Normal"}** |
"""
        return output_path, report

    def process_image(img):
        if img is None:
            return None, "❌ Please upload an image."
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        res = pipeline.process_frame(frame, mode="crowd")
        
        # Generate heatmap overlay
        cnt, detections = pipeline.crowd_analyzer.count_people(frame)
        heatmap_overlay, _ = pipeline.crowd_analyzer.generate_density_heatmap(res['frame'], detections)
        heatmap_rgb = cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB)
        
        report = f"""
### 📊 Frame Crowd Metrics

- **Total Detected Passengers:** `{cnt}`
- **Platform Density Level:** `{res['crowd_level']}`
- **Zone Occupancy:** Track Safety Zone (`{cnt}` passengers)
"""
        return heatmap_rgb, report

    with gr.Blocks(title="Crowd Analytics & Density Dashboard — Indian Railways AI") as demo:
        gr.Markdown("""
        # 👥 Indian Railways AI — Crowd Analytics & Passenger Density System
        ### Dedicated High-Performance Passenger Counting, Heatmap Generation & ByteTrack Trajectories
        """)

        with gr.Tabs():
            with gr.Tab("📹 Live Webcam Stream"):
                with gr.Row():
                    with gr.Column():
                        web_in = gr.Image(sources=["webcam"], streaming=True, label="📷 Live Camera Input")
                    with gr.Column():
                        web_out = gr.Image(label="🎯 ByteTrack Passenger Overlay")
                        web_stat = gr.Markdown()
                web_in.stream(process_webcam, inputs=[web_in], outputs=[web_out, web_stat])
                web_in.change(process_webcam, inputs=[web_in], outputs=[web_out, web_stat])

            with gr.Tab("🎥 CCTV Video Processor"):
                with gr.Row():
                    with gr.Column():
                        vid_in = gr.Video(label="Upload Platform CCTV Video")
                        btn_vid = gr.Button("👥 Process Crowd Analytics Video", variant="primary")
                    with gr.Column():
                        vid_out = gr.Video(label="🎬 Processed Video Output")
                        vid_rep = gr.Markdown()
                btn_vid.click(process_video_file, inputs=[vid_in], outputs=[vid_out, vid_rep])

            with gr.Tab("📸 Heatmap & Single Image Inspector"):
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(label="Upload Platform Photo", type="numpy")
                        btn_img = gr.Button("🔍 Generate Heatmap & Count", variant="primary")
                    with gr.Column():
                        img_out = gr.Image(label="🔥 Gaussian Density Heatmap Overlay")
                        img_rep = gr.Markdown()
                btn_img.click(process_image, inputs=[img_in], outputs=[img_out, img_rep])

    return demo


if __name__ == "__main__":
    demo = create_crowd_dashboard()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
