import sys
import os
import time
import tempfile
import cv2
import numpy as np
import gradio as gr

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_path)

from pipeline import system, pipeline


def create_criminal_dashboard():
    def process_webcam(frame):
        if frame is None:
            return None, "⚠️ Waiting for webcam frame..."
        if isinstance(frame, dict):
            frame = frame.get('image', frame.get('composite', None))
        if frame is None:
            return None, "⚠️ Waiting for camera..."

        arr = cv2.cvtColor(np.array(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        res = pipeline.process_frame(arr, mode="criminal")
        out_rgb = cv2.cvtColor(res['frame'], cv2.COLOR_BGR2RGB)

        if res['criminals_found']:
            names = ", ".join([m['name'] for m in res['criminals_found']])
            scores = ", ".join([f"{m['score']:.2f}" for m in res['criminals_found']])
            status = f"🚨 **CRITICAL SECURITY ALERT: WANTED SUSPECT IDENTIFIED!**\n> **Suspect Name:** `{names.upper()}` | **Confidence:** `{scores}` | **RPF Dispatched** | FPS: `{res['fps']:.1f}`"
        else:
            status = f"🟢 **CRIMINAL SCANNER ACTIVE** | Enrolled Database Watchlist: `{len(system.criminal_db)}` suspects | Suspects Flagged: `0` | FPS: `{res['fps']:.1f}`"

        return out_rgb, status

    def process_video_file(video_file):
        if video_file is None:
            return None, "❌ Please upload a video file."
        input_path = video_file if isinstance(video_file, str) else getattr(video_file, "name", str(video_file))
        output_path = os.path.join(tempfile.gettempdir(), "criminal_recognition_output.mp4")

        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        target_w = 1280
        target_h = int(h * (target_w / w)) if w else 720

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

        flagged_suspects = set()
        total_frames = 0

        while cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            f_resized = cv2.resize(f, (target_w, target_h))
            res = pipeline.process_frame(f_resized, mode="criminal")
            out.write(res["frame"])
            for match in res["criminals_found"]:
                flagged_suspects.add(match["name"])
            total_frames += 1

        cap.release()
        out.release()

        report = f"""
### 🚨 Suspect Identification CCTV Summary Report

| Metric | Result |
|---|---|
| ⏱️ Total Video Frames Analyzed | **{total_frames} frames** |
| 🚨 Blacklisted Suspects Flagged | **{len(flagged_suspects)} ({', '.join(flagged_suspects) if flagged_suspects else 'None'})** |
| 📍 Status | **{"🚨 SUSPECT IDENTIFIED IN FEED" if flagged_suspects else "✅ Feed Clear — No Suspects"}** |
"""
        return output_path, report

    def register_suspect(image, name, offense):
        if image is None or not name.strip():
            return "❌ Please provide suspect photo and name/case ID.", get_suspect_list()
        
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"criminal_reg_{name.strip()}.jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        success = system.add_criminal_to_db(name.strip(), temp_path)
        if success:
            msg = f"✅ **Suspect '{name.strip()}' successfully enrolled into ArcFace Vector DB!**\n- Offense Notes: {offense or 'General Watchlist'}\n- Total Registered Suspects: {len(system.criminal_db)}"
            return msg, get_suspect_list()
        return "❌ Face not detected in photo. Ensure clear lighting.", get_suspect_list()

    def get_suspect_list():
        if not system.criminal_db:
            return "📋 *No suspects currently enrolled in database.*"
        md = "### 🚨 Active Blacklist Database\n\n| Suspect ID / Case Name | Vector Embedding | Watchlist Status |\n|---|---|---|\n"
        for cname in system.criminal_db.keys():
            md += f"| **{cname}** | 512-dim ArcFace | 🔴 Active Watchlist |\n"
        return md

    with gr.Blocks(title="Criminal Face Recognition — Indian Railways AI") as demo:
        gr.Markdown("""
        # 🚨 Indian Railways AI — Criminal Recognition & Suspect Watchlist System
        ### Dedicated High-Precision 512-Dim ArcFace Vector Face Matching & Security Alert System
        """)

        with gr.Tabs():
            with gr.Tab("📹 Live Webcam Suspect Scanner"):
                with gr.Row():
                    with gr.Column():
                        web_in = gr.Image(sources=["webcam"], streaming=True, label="📷 Live Camera Feed")
                    with gr.Column():
                        web_out = gr.Image(label="🎯 Real-Time Face Recognition Overlay")
                        web_stat = gr.Markdown()
                web_in.stream(process_webcam, inputs=[web_in], outputs=[web_out, web_stat])
                web_in.change(process_webcam, inputs=[web_in], outputs=[web_out, web_stat])

            with gr.Tab("🎥 CCTV Video Processor"):
                with gr.Row():
                    with gr.Column():
                        vid_in = gr.Video(label="Upload CCTV Video File")
                        btn_vid = gr.Button("🚨 Run Criminal Recognition Scan", variant="stop")
                    with gr.Column():
                        vid_out = gr.Video(label="🎬 Processed Video Output")
                        vid_rep = gr.Markdown()
                btn_vid.click(process_video_file, inputs=[vid_in], outputs=[vid_out, vid_rep])

            with gr.Tab("👤 Suspect Enrollment & Watchlist DB"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🚨 Enroll New Wanted Suspect Photo")
                        c_img = gr.Image(label="Suspect Mugshot Photo", sources=["webcam", "upload"], type="numpy")
                        c_name = gr.Textbox(label="Suspect Name / Police Case ID", placeholder="e.g. Suspect_Raju_402")
                        c_offense = gr.Textbox(label="Offense / Crime Description", placeholder="e.g. Wanted for luggage theft on Platform 3")
                        btn_reg = gr.Button("🚨 Enroll Suspect into Watchlist DB", variant="stop")
                        reg_msg = gr.Markdown()
                    with gr.Column():
                        db_view = gr.Markdown(value=get_suspect_list())

                btn_reg.click(register_suspect, inputs=[c_img, c_name, c_offense], outputs=[reg_msg, db_view])

    return demo


if __name__ == "__main__":
    demo = create_criminal_dashboard()
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
