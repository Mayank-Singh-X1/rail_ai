import sys
import os
import time
import tempfile
import cv2
import numpy as np
import gradio as gr

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_path)

from pipeline import pipeline


def create_anomaly_dashboard():
    def process_webcam(frame):
        if frame is None:
            return None, "⚠️ Waiting for webcam frame..."
        if isinstance(frame, dict):
            frame = frame.get('image', frame.get('composite', None))
        if frame is None:
            return None, "⚠️ Waiting for camera..."

        arr = cv2.cvtColor(np.array(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        res = pipeline.process_frame(arr, mode="anomaly")
        out_rgb = cv2.cvtColor(res['frame'], cv2.COLOR_BGR2RGB)

        if res['anomalies']:
            types = ", ".join([a['type'] for a in res['anomalies']])
            status = f"🚨 **SAFETY INCIDENT DETECTED!**\n> **Incident Type:** `{types}` | **Action:** RPF Emergency Team Alerted | FPS: `{res['fps']:.1f}`"
        else:
            status = f"🟢 **ANOMALY & POSE SCANNER ACTIVE** | Safety Breaches: `0` | Status: `NORMAL / SAFE` | FPS: `{res['fps']:.1f}`"

        return out_rgb, status

    def process_video_file(video_file):
        if video_file is None:
            return None, "❌ Please upload a video file."
        input_path = video_file if isinstance(video_file, str) else getattr(video_file, "name", str(video_file))
        output_path = os.path.join(tempfile.gettempdir(), "anomaly_detection_output.mp4")

        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        target_w = 1280
        target_h = int(h * (target_w / w)) if w else 720

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

        all_anomalies = []
        total_frames = 0

        while cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            f_resized = cv2.resize(f, (target_w, target_h))
            res = pipeline.process_frame(f_resized, mode="anomaly")
            out.write(res["frame"])
            for anom in res["anomalies"]:
                all_anomalies.append(anom["type"])
            total_frames += 1

        cap.release()
        out.release()

        report = f"""
### ⚠️ Safety & Anomaly Detection Report

| Security & Safety Metric | Result |
|---|---|
| ⏱️ Total Video Frames Analyzed | **{total_frames} frames** |
| 🆘 Total Safety Incidents Detected | **{len(all_anomalies)}** |
| 📍 Incident Summary | **{', '.join(set(all_anomalies)) if all_anomalies else 'None (All Normal)'}** |
"""
        return output_path, report

    with gr.Blocks(title="Anomaly & Fall Detection — Indian Railways AI") as demo:
        gr.Markdown("""
        # ⚠️ Indian Railways AI — Anomaly, Fall & Track Intrusion Detector
        ### Dedicated YOLO11 Pose Estimation Engine for Real-Time Fall Collapse, Violence & Restricted Zone Safety
        """)

        with gr.Tabs():
            with gr.Tab("📹 Live Pose Stream"):
                with gr.Row():
                    with gr.Column():
                        web_in = gr.Image(sources=["webcam"], streaming=True, label="📷 Live Camera Stream")
                    with gr.Column():
                        web_out = gr.Image(label="🎯 Pose Keypoints & Incident Overlay")
                        web_stat = gr.Markdown()
                web_in.stream(process_webcam, inputs=[web_in], outputs=[web_out, web_stat])
                web_in.change(process_webcam, inputs=[web_in], outputs=[web_out, web_stat])

            with gr.Tab("🎥 CCTV Video Processor"):
                with gr.Row():
                    with gr.Column():
                        vid_in = gr.Video(label="Upload CCTV Video File")
                        btn_vid = gr.Button("⚠️ Run Anomaly & Fall Scan", variant="stop")
                    with gr.Column():
                        vid_out = gr.Video(label="🎬 Processed Video Output")
                        vid_rep = gr.Markdown()
                btn_vid.click(process_video_file, inputs=[vid_in], outputs=[vid_out, vid_rep])

    return demo


if __name__ == "__main__":
    demo = create_anomaly_dashboard()
    demo.launch(server_name="127.0.0.1", server_port=7862, share=False)
