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


def create_weapon_dashboard():
    def process_webcam(frame):
        if frame is None:
            return None, "⚠️ Waiting for webcam frame..."
        if isinstance(frame, dict):
            frame = frame.get('image', frame.get('composite', None))
        if frame is None:
            return None, "⚠️ Waiting for camera..."

        arr = cv2.cvtColor(np.array(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        res = pipeline.process_frame(arr, mode="weapon")
        out_rgb = cv2.cvtColor(res['frame'], cv2.COLOR_BGR2RGB)

        if res['weapons_found']:
            labels = ", ".join([w['label'] for w in res['weapons_found']])
            confs = ", ".join([f"{w['confidence']:.0%}" for w in res['weapons_found']])
            status = f"🚨 **CRITICAL SECURITY ALERT: WEAPON DETECTED [{labels.upper()}]!**\n> **Threat Type:** `{labels}` | **Confidence:** `{confs}` | **Action:** RPF Emergency Dispatch Alerted | FPS: `{res['fps']:.1f}`"
        else:
            status = f"🟢 **WEAPON SCANNER ACTIVE** | Threat Status: `SECURE / NO WEAPONS` | FPS: `{res['fps']:.1f}`"

        return out_rgb, status

    def process_video_file(video_file):
        if video_file is None:
            return None, "❌ Please upload a video file."
        input_path = video_file if isinstance(video_file, str) else getattr(video_file, "name", str(video_file))
        output_path = os.path.join(tempfile.gettempdir(), "weapon_scan_output.mp4")

        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        target_w = 1280
        target_h = int(h * (target_w / w)) if w else 720

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

        detected_weapons = set()
        total_frames = 0

        while cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            f_resized = cv2.resize(f, (target_w, target_h))
            res = pipeline.process_frame(f_resized, mode="weapon")
            out.write(res["frame"])
            for w_item in res["weapons_found"]:
                detected_weapons.add(w_item["label"])
            total_frames += 1

        cap.release()
        out.release()

        report = f"""
### 🔫 Weapon Threat Detection CCTV Report

| Security Metric | Result |
|---|---|
| ⏱️ Total Video Frames Analyzed | **{total_frames} frames** |
| 🔫 Weapons Detected | **{len(detected_weapons)} ({', '.join(detected_weapons) if detected_weapons else 'None'})** |
| 📍 Threat Level | **{"🚨 CRITICAL SECURITY ALERT" if detected_weapons else "🟢 Safe / Clear"}** |
"""
        return output_path, report

    with gr.Blocks(title="Weapon & Threat Scan — Indian Railways AI") as demo:
        gr.Markdown("""
        # 🔫 Indian Railways AI — Real-Time Weapon & Threat Scanner
        ### Dedicated YOLO11 Weapon Scanning Engine for Concealed/Exposed Knives, Firearms & Improvised Threat Objects
        """)

        with gr.Tabs():
            with gr.Tab("📹 Live Threat Scanner"):
                with gr.Row():
                    with gr.Column():
                        web_in = gr.Image(sources=["webcam"], streaming=True, label="📷 Live Security Camera Stream")
                    with gr.Column():
                        web_out = gr.Image(label="🎯 Real-Time Threat Overlay")
                        web_stat = gr.Markdown()
                web_in.stream(process_webcam, inputs=[web_in], outputs=[web_out, web_stat])
                web_in.change(process_webcam, inputs=[web_in], outputs=[web_out, web_stat])

            with gr.Tab("🎥 CCTV Video Threat Scan"):
                with gr.Row():
                    with gr.Column():
                        vid_in = gr.Video(label="Upload CCTV Video File")
                        btn_vid = gr.Button("🔫 Run Weapon Threat Scan", variant="stop")
                    with gr.Column():
                        vid_out = gr.Video(label="🎬 Processed Video Output")
                        vid_rep = gr.Markdown()
                btn_vid.click(process_video_file, inputs=[vid_in], outputs=[vid_out, vid_rep])

    return demo


if __name__ == "__main__":
    demo = create_weapon_dashboard()
    demo.launch(server_name="127.0.0.1", server_port=7865, share=False)
