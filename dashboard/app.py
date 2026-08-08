import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

"""
Interactive Gradio Dashboard for Indian Railways AI Surveillance System
Optimized for high-speed continuous streaming, multi-camera uploads, and face databases.
"""
import os
import tempfile
import cv2
import numpy as np
import gradio as gr

try:
    from pipeline import system, pipeline, process_video, stream_live_camera
except ImportError:
    from pipeline import RailwaySurveillanceSystem, UnifiedPipeline
    try:
        from pipeline import process_video, stream_live_camera
    except ImportError:
        pass
    system = RailwaySurveillanceSystem()
    pipeline = UnifiedPipeline()

from utils.analytics import AnalyticsDashboard


def create_dashboard():
    
    # -------------------------------------------------------------
    # Helper 1: Real-time Live Webcam Stream Processing (Continuous & Non-blocking)
    # -------------------------------------------------------------
    def process_webcam_frame(webcam_frame, enable_crowd=True, enable_criminal=True, enable_anomaly=True, enable_cleanliness=True):
        """Processes real-time webcam frame from browser in Colab/Local continuously without freeze."""
        if webcam_frame is None:
            return None
        
        # Handle dict or raw numpy array from Gradio
        if isinstance(webcam_frame, dict):
            webcam_frame = webcam_frame.get('image', webcam_frame.get('composite', None))
        
        if webcam_frame is None:
            return None
            
        try:
            arr = np.array(webcam_frame, dtype=np.uint8)
            if len(arr.shape) == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            
            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            results = pipeline.process_frame(
                frame,
                enable_crowd=bool(enable_crowd),
                enable_criminal=bool(enable_criminal),
                enable_anomaly=bool(enable_anomaly),
                enable_cleanliness=bool(enable_cleanliness)
            )
            return cv2.cvtColor(results['frame'], cv2.COLOR_BGR2RGB)
        except Exception as e:
            return webcam_frame

    # -------------------------------------------------------------
    # Helper 2: Video File Processing
    # -------------------------------------------------------------
    def process_uploaded_video(video_file, frame_stride, progress=gr.Progress()):
        if video_file is None:
            return None, "❌ Please upload a video file (.mp4, .avi, .mov)."
        
        progress(0.1, desc="Loading video file...")
        input_path = video_file if isinstance(video_file, str) else getattr(video_file, "name", str(video_file))
        output_path = os.path.join(tempfile.gettempdir(), "annotated_surveillance_output.mp4")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return None, "❌ Error reading video file."
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
        
        target_w = 1280
        target_h = int(orig_h * (target_w / orig_w)) if orig_w else 720
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, max(1, fps // int(frame_stride)), (target_w, target_h))
        
        frame_idx = 0
        processed_count = 0
        max_crowd = 0
        total_cleanliness = []
        all_suspects = set()
        all_anomalies = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % int(frame_stride) == 0:
                frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                res = pipeline.process_frame(frame_resized)
                out.write(res["frame"])
                
                max_crowd = max(max_crowd, res["crowd_count"])
                total_cleanliness.append(res["cleanliness_score"])
                for match in res["criminals_found"]:
                    all_suspects.add(match["name"])
                for anom in res["anomalies"]:
                    all_anomalies.append(anom["type"])
                
                processed_count += 1
                progress(min(0.95, frame_idx / total_frames), desc=f"Analyzing frame {frame_idx}/{total_frames}...")
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        avg_clean = np.mean(total_cleanliness) if total_cleanliness else 100.0
        clean_grade = "🟢 Excellent (Grade A)" if avg_clean > 80 else ("🟡 Moderate (Grade B)" if avg_clean > 50 else "🔴 Needs Attention (Grade C)")
        
        summary_report = f"""
### 📊 Video Surveillance Summary Report

| Security & Operations Metric | Result |
|---|---|
| ⏱️ Total Frames Processed | **{processed_count} frames** |
| 👥 Peak Platform Crowd | **{max_crowd} passengers** |
| 🧹 Average Cleanliness Score | **{avg_clean:.1f}% — {clean_grade}** |
| 🚨 Suspects / Criminals Flagged | **{len(all_suspects)} ({', '.join(all_suspects) if all_suspects else 'None'})** |
| ⚠️ Total Anomalies / Safety Incidents | **{len(all_anomalies)}** |
| 📍 Status | ✅ **Processing Complete** |
"""
        return output_path, summary_report

    # -------------------------------------------------------------
    # Helper 3: Single Image Inspection
    # -------------------------------------------------------------
    def process_image(image, enable_crowd, enable_criminal, enable_anomaly, enable_cleanliness):
        if image is None:
            return None, "❌ Please capture or upload an image.", "No data"
        
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = pipeline.process_frame(
            frame,
            enable_crowd=enable_crowd,
            enable_criminal=enable_criminal,
            enable_anomaly=enable_anomaly,
            enable_cleanliness=enable_cleanliness
        )
        output_frame = cv2.cvtColor(results['frame'], cv2.COLOR_BGR2RGB)
        
        clean_badge = "🟢 Clean" if results['cleanliness_score'] > 75 else ("🟡 Moderate" if results['cleanliness_score'] > 45 else "🔴 Dirty")
        
        analytics = f"""
### 📊 Platform Frame Metrics

| Metric | Result | Status |
|---|---|---|
| 👥 **Passenger Count** | **{results['crowd_count']}** | Level: `{results['crowd_level']}` |
| 🧹 **Cleanliness Score** | **{results['cleanliness_score']:.1f}%** | {clean_badge} |
| 🚨 **Suspects Identified** | **{len(results['criminals_found'])}** | {'⚠️ MATCH FOUND' if results['criminals_found'] else '✅ Clear'} |
| ⚠️ **Pose Anomalies** | **{len(results['anomalies'])}** | {'⚠️ Fall/Fight Detected' if results['anomalies'] else '✅ Normal'} |
| 📍 **Tracked Objects** | **{results['tracked_persons']}** | ByteTrack Active |
"""
        alerts_text = "### 🔔 Active Security Alerts\n\n"
        if results['alerts']:
            for alert in results['alerts'][-5:]:
                alerts_text += f"- **{alert.get('type', 'Alert')}** ({alert.get('severity', 'LOW')}) at {alert.get('timestamp', 'N/A')}\n"
        else:
            alerts_text += "✅ No active safety alerts for this frame.\n"
            
        return output_frame, analytics, alerts_text

    # -------------------------------------------------------------
    # Helper 4: Criminal / Suspect Registration
    # -------------------------------------------------------------
    def register_criminal(image, name, offense_notes):
        if image is None or not name.strip():
            return "❌ Please provide both a photo and a valid Suspect Name/ID.", get_criminal_list()
        
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"criminal_{name.strip()}.jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        success = system.add_criminal_to_db(name.strip(), temp_path)
        if success:
            msg = f"✅ **Suspect '{name.strip()}' successfully registered in Vector DB!**\n- Offense Notes: {offense_notes or 'General Watchlist'}\n- Total Registered Suspects: {len(system.criminal_db)}"
            return msg, get_criminal_list()
        return f"❌ **Face not detected** in photo. Please ensure face is clearly visible and well-lit.", get_criminal_list()

    def get_criminal_list():
        if not system.criminal_db:
            return "📋 *No suspects currently registered in active memory database.*"
        md = "### 🚨 Active Blacklist Database\n\n| ID / Name | Vector Dimensions | Status |\n|---|---|---|\n"
        for cname in system.criminal_db.keys():
            md += f"| **{cname}** | 512-dim ArcFace | 🔴 Active Watchlist |\n"
        return md

    # -------------------------------------------------------------
    # Helper 5: Railway Staff / Worker Registration & Attendance
    # -------------------------------------------------------------
    def register_worker(image, name, department):
        if image is None or not name.strip():
            return "❌ Please provide both a photo and a valid Official Name/Badge ID.", get_worker_list()
        
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"worker_{name.strip()}.jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        success = system.add_worker_to_db(name.strip(), temp_path)
        if success:
            msg = f"✅ **Staff Member '{name.strip()}' successfully registered!**\n- Department: {department or 'General Operations'}\n- Total Registered Officials: {len(system.worker_db)}"
            return msg, get_worker_list()
        return f"❌ **Face not detected** in photo. Please ensure face is clearly visible.", get_worker_list()

    def get_worker_list():
        if not system.worker_db:
            return "📋 *No staff members currently registered in active memory database.*"
        md = "### 👷 Active Staff Roster\n\n| Official Name / Badge ID | Vector Embedding | Attendance Status |\n|---|---|---|\n"
        for wname in system.worker_db.keys():
            md += f"| **{wname}** | 512-dim InsightFace | 🟢 Enrolled on Duty |\n"
        return md

    # -------------------------------------------------------------
    # BUILD GRADIO DASHBOARD UI
    # -------------------------------------------------------------
    with gr.Blocks(title="🚂 Indian Railways AI Surveillance System", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🚂 Indian Railways AI Surveillance & Safety Platform
        ### Smart India Hackathon (SIH) — Autonomous Multi-Model Video Analytics
        *Real-time passenger safety, crowd management, criminal identification, automated cleanliness tracking, and staff attendance.*
        """)
        
        with gr.Tabs():
            
            # =========================================================
            # TAB 1: LIVE BROWSER WEBCAM (CONTINUOUS STREAMING)
            # =========================================================
            with gr.Tab("📹 Live Browser Webcam (Real-Time AI)"):
                gr.Markdown("### ⚡ Stream directly from your Laptop/Mobile Webcam to the GPU in real-time")
                with gr.Row():
                    with gr.Column(scale=1):
                        webcam_input = gr.Image(
                            sources=["webcam"], 
                            streaming=True,
                            stream_every=0.08,
                            label="📷 Live Client Webcam Feed"
                        )
                        with gr.Accordion("⚙️ Active Module Filters", open=True):
                            live_crowd_chk = gr.Checkbox(True, label="👥 Crowd Counting & Density")
                            live_crim_chk = gr.Checkbox(True, label="🚨 Criminal / Suspect Recognition")
                            live_anom_chk = gr.Checkbox(True, label="⚠️ Anomaly & Fall Detection")
                            live_clean_chk = gr.Checkbox(True, label="🧹 Cleanliness Scoring")
                    
                    with gr.Column(scale=1):
                        webcam_output = gr.Image(
                            label="🎯 Annotated AI Output with Tracking & Safety HUD"
                        )
                
                # Real-time continuous non-blocking frame stream
                webcam_input.stream(
                    fn=process_webcam_frame,
                    inputs=[webcam_input, live_crowd_chk, live_crim_chk, live_anom_chk, live_clean_chk],
                    outputs=[webcam_output],
                    concurrency_limit=None,
                    show_progress="hidden"
                )

            # =========================================================
            # TAB 2: VIDEO FILE UPLOAD & PROCESSING
            # =========================================================
            with gr.Tab("🎥 Upload Video & CCTV Processing"):
                gr.Markdown("### 📂 Upload Platform CCTV Video (.mp4 / .avi / .mov) for Full Analysis")
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Surveillance Video")
                        frame_stride_slider = gr.Slider(
                            minimum=1, 
                            maximum=10, 
                            value=2, 
                            step=1, 
                            label="⚡ Processing Frame Stride (1 = Every Frame, 2 = 2x Speed)"
                        )
                        process_video_btn = gr.Button("🎬 Run Full AI Video Analysis", variant="primary")
                    
                    with gr.Column(scale=1):
                        video_output = gr.Video(label="🎬 Processed Annotated Surveillance Video")
                        video_summary_output = gr.Markdown(label="Video Analytics Report")
                
                process_video_btn.click(
                    process_uploaded_video,
                    inputs=[video_input, frame_stride_slider],
                    outputs=[video_output, video_summary_output]
                )

            # =========================================================
            # TAB 3: SINGLE IMAGE & ZONE INSPECTOR
            # =========================================================
            with gr.Tab("📸 Single Frame & Zone Inspector"):
                gr.Markdown("### 🔍 Inspect Single Image / Screenshot for Crowd & Safety Breaches")
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="Upload or Snap Platform Photo", 
                            sources=["webcam", "upload", "clipboard"], 
                            type="numpy"
                        )
                        with gr.Row():
                            crowd_check = gr.Checkbox(True, label="👥 Crowd")
                            criminal_check = gr.Checkbox(True, label="🚨 Suspects")
                            anomaly_check = gr.Checkbox(True, label="⚠️ Anomalies")
                            clean_check = gr.Checkbox(True, label="🧹 Cleanliness")
                        analyze_btn = gr.Button("🔍 Run AI Inspection", variant="primary")
                    
                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Annotated Result")
                        analytics_output = gr.Markdown()
                        alerts_output = gr.Markdown()
                
                analyze_btn.click(
                    process_image,
                    inputs=[input_image, crowd_check, criminal_check, anomaly_check, clean_check],
                    outputs=[output_image, analytics_output, alerts_output]
                )

            # =========================================================
            # TAB 4: SUSPECT & STAFF FACE DATABASE REGISTRATION
            # =========================================================
            with gr.Tab("👤 Face Database & Staff Attendance"):
                gr.Markdown("### 🗄️ Register Suspects for Blacklist Watch and Railway Staff for Attendance")
                
                with gr.Row():
                    # Register Criminal
                    with gr.Column():
                        gr.Markdown("#### 🚨 1. Register Blacklisted Suspect")
                        criminal_image = gr.Image(label="Suspect Photo (Webcam / Upload)", sources=["webcam", "upload"], type="numpy")
                        criminal_name = gr.Textbox(label="Suspect Name / Police Case ID", placeholder="e.g. Suspect_Raju_402")
                        criminal_notes = gr.Textbox(label="Offense / Watchlist Reason", placeholder="e.g. Wanted for luggage theft on Platform 3")
                        register_criminal_btn = gr.Button("🚨 Register Suspect into Watchlist", variant="stop")
                        criminal_status = gr.Markdown()
                        criminal_db_view = gr.Markdown(value=get_criminal_list())
                    
                    # Register Worker
                    with gr.Column():
                        gr.Markdown("#### 👷 2. Register Railway Staff Member")
                        worker_image = gr.Image(label="Staff Photo (Webcam / Upload)", sources=["webcam", "upload"], type="numpy")
                        worker_name = gr.Textbox(label="Official Name / Badge ID", placeholder="e.g. RPF_Inspector_Kumar")
                        worker_dept = gr.Dropdown(
                            choices=["Railway Protection Force (RPF)", "Station Management", "Sanitation & Cleaning Crew", "Ticket Checking Staff", "Track Maintenance"],
                            value="Railway Protection Force (RPF)",
                            label="Department / Role"
                        )
                        register_worker_btn = gr.Button("👷 Register Official into Staff Roster", variant="primary")
                        worker_status = gr.Markdown()
                        worker_db_view = gr.Markdown(value=get_worker_list())
                
                register_criminal_btn.click(
                    register_criminal, 
                    inputs=[criminal_image, criminal_name, criminal_notes], 
                    outputs=[criminal_status, criminal_db_view]
                )
                register_worker_btn.click(
                    register_worker, 
                    inputs=[worker_image, worker_name, worker_dept], 
                    outputs=[worker_status, worker_db_view]
                )
                
    return demo


if __name__ == "__main__":
    demo = create_dashboard()
    demo.queue(default_concurrency_limit=20)
    demo.launch(share=False)
