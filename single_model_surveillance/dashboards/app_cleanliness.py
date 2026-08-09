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


def create_cleanliness_dashboard():
    def process_webcam(frame):
        if frame is None:
            return None, "⚠️ Waiting for webcam frame..."
        if isinstance(frame, dict):
            frame = frame.get('image', frame.get('composite', None))
        if frame is None:
            return None, "⚠️ Waiting for camera..."

        arr = cv2.cvtColor(np.array(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        res = pipeline.process_frame(arr, mode="cleanliness")
        out_rgb = cv2.cvtColor(res['frame'], cv2.COLOR_BGR2RGB)

        score = res['cleanliness_score']
        grade = "🟢 Excellent (Grade A)" if score > 80 else ("🟡 Moderate (Grade B)" if score > 50 else "🔴 Dirty (Needs Sanitation Drive)")
        status = f"🧹 **CLEANLINESS MONITOR ACTIVE** | Score: `{score:.1f}%` ({grade}) | FPS: `{res['fps']:.1f}`"

        return out_rgb, status

    def process_video_file(video_file):
        if video_file is None:
            return None, "❌ Please upload a video file."
        input_path = video_file if isinstance(video_file, str) else getattr(video_file, "name", str(video_file))
        output_path = os.path.join(tempfile.gettempdir(), "cleanliness_monitoring_output.mp4")

        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        target_w = 1280
        target_h = int(h * (target_w / w)) if w else 720

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

        scores = []
        total_frames = 0

        while cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            f_resized = cv2.resize(f, (target_w, target_h))
            res = pipeline.process_frame(f_resized, mode="cleanliness")
            out.write(res["frame"])
            scores.append(res["cleanliness_score"])
            total_frames += 1

        cap.release()
        out.release()

        avg_score = np.mean(scores) if scores else 100.0
        grade = "🟢 Grade A (High Sanitation)" if avg_score > 80 else ("🟡 Grade B (Moderate)" if avg_score > 50 else "🔴 Grade C (Dispatch Sanitation Staff)")

        report = f"""
### 🧹 Station Cleanliness Audit Summary Report

| Sanitation Metric | Value |
|---|---|
| ⏱️ Total Frames Inspected | **{total_frames} frames** |
| 🧹 Average Cleanliness Score | **{avg_score:.1f}%** |
| 🏆 Sanitation Grade | **{grade}** |
| 📋 Recommendation | **{"✅ Standards Met" if avg_score > 75 else "🧹 Dispatch Sanitation Staff to Platform"}** |
"""
        return output_path, report

    with gr.Blocks(title="Station Cleanliness Monitor — Indian Railways AI") as demo:
        gr.Markdown("""
        # 🧹 Indian Railways AI — Automated Station Cleanliness & Sanitation Monitor
        ### Dedicated YOLO11 Litter Detection & Automated Station Cleanliness Scoring Engine (0-100%)
        """)

        with gr.Tabs():
            with gr.Tab("📹 Live Sanitation Feed"):
                with gr.Row():
                    with gr.Column():
                        web_in = gr.Image(sources=["webcam"], streaming=True, label="📷 Live Platform Camera")
                    with gr.Column():
                        web_out = gr.Image(label="🎯 Litter Overlay & Sanitation HUD")
                        web_stat = gr.Markdown()
                web_in.stream(process_webcam, inputs=[web_in], outputs=[web_out, web_stat])
                web_in.change(process_webcam, inputs=[web_in], outputs=[web_out, web_stat])

            with gr.Tab("🎥 CCTV Video Sanitation Audit"):
                with gr.Row():
                    with gr.Column():
                        vid_in = gr.Video(label="Upload CCTV Video File")
                        btn_vid = gr.Button("🧹 Run Cleanliness Audit Scan", variant="primary")
                    with gr.Column():
                        vid_out = gr.Video(label="🎬 Processed Video Output")
                        vid_rep = gr.Markdown()
                btn_vid.click(process_video_file, inputs=[vid_in], outputs=[vid_out, vid_rep])

    return demo


if __name__ == "__main__":
    demo = create_cleanliness_dashboard()
    demo.launch(server_name="127.0.0.1", server_port=7863, share=False)
