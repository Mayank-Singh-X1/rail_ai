import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

"""
Interactive Gradio Dashboard for Railway Surveillance AI System
"""
import os
import tempfile
import cv2
import numpy as np
import gradio as gr

from pipeline import system, pipeline, process_video, stream_live_camera
from utils.analytics import AnalyticsDashboard


# ============================================
# CELL 11: Interactive Dashboard (Gradio with Live Stream)
# ============================================
import os
import tempfile
import cv2
import numpy as np
import gradio as gr

def create_dashboard():
    
    def process_image(image, enable_crowd, enable_criminal, enable_anomaly, enable_cleanliness):
        if image is None:
            return None, "❌ Please upload an image.", "No data"
        
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = pipeline.process_frame(
            frame,
            enable_crowd=enable_crowd,
            enable_criminal=enable_criminal,
            enable_anomaly=enable_anomaly,
            enable_cleanliness=enable_cleanliness
        )
        output_frame = cv2.cvtColor(results['frame'], cv2.COLOR_BGR2RGB)
        
        analytics = f"""
## 📊 Analysis Results

| Metric | Value |
|--------|-------|
| 👥 People Count | {results['crowd_count']} |
| 📊 Crowd Level | {results['crowd_level']} |
| 🧹 Cleanliness Score | {results['cleanliness_score']:.1f}% |
| 🚨 Criminals Detected | {len(results['criminals_found'])} |
| ⚠️ Anomalies | {len(results['anomalies'])} |
| 📍 Tracked Persons | {results['tracked_persons']} |
"""
        alerts_text = "## 🔔 Alerts\n\n"
        if results['alerts']:
            for alert in results['alerts'][-5:]:
                alerts_text += f"- **{alert.get('type', 'Alert')}** at {alert.get('timestamp', 'N/A')}\n"
        else:
            alerts_text += "✅ No active alerts\n"
            
        return output_frame, analytics, alerts_text

    def register_criminal(image, name):
        if image is None or not name.strip():
            return "❌ Please provide both an image and a valid name."
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"criminal_{name}.jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        success = system.add_criminal_to_db(name.strip(), temp_path)
        if success:
            return f"✅ Suspect '{name}' registered! Active DB size: {len(system.criminal_db)}"
        return f"❌ Failed to register '{name}'. No face detected."

    def register_worker(image, name):
        if image is None or not name.strip():
            return "❌ Please provide both an image and a valid name."
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"worker_{name}.jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        success = system.add_worker_to_db(name.strip(), temp_path)
        if success:
            return f"✅ Worker '{name}' registered! Active DB size: {len(system.worker_db)}"
        return f"❌ Failed to register '{name}'. No face detected."

    # ---- BUILD GRADIO INTERFACE ----
    with gr.Blocks(title="🚂 Indian Railways AI Surveillance System", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🚂 Indian Railways AI Surveillance System
        ### Smart India Hackathon
        *AI-powered CCTV analytics for crowd management, crime prevention, and operations monitoring*
        """)
        
        with gr.Tabs():
            
            # ---- TAB 1: REAL-TIME LIVE STREAMING ----
            with gr.Tab("🔴 Real-Time Live Feed"):
                with gr.Row():
                    with gr.Column():
                        stream_source_input = gr.Textbox(
                            label="Live CCTV Source", 
                            value="test_video.mp4", 
                            placeholder="Type 0 for Webcam OR video path OR rtsp://... stream URL"
                        )
                        start_stream_btn = gr.Button("▶️ Start Live Stream", variant="primary")
                    
                    with gr.Column():
                        live_stream_output = gr.Image(label="Live Annotated Surveillance Feed", streaming=True)
                
                # Link generator function directly to streaming output
                start_stream_btn.click(
                    stream_live_camera,
                    inputs=[stream_source_input],
                    outputs=[live_stream_output]
                )

            # ---- TAB 2: Image Analysis ----
            with gr.Tab("📸 Image Analysis"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Upload Station Image", type="numpy")
                        with gr.Row():
                            crowd_check = gr.Checkbox(True, label="👥 Crowd Detection")
                            criminal_check = gr.Checkbox(True, label="🚨 Criminal Detection")
                            anomaly_check = gr.Checkbox(True, label="⚠️ Anomaly Detection")
                            clean_check = gr.Checkbox(True, label="🧹 Cleanliness")
                        analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                    
                    with gr.Column():
                        output_image = gr.Image(label="Analyzed Output")
                        analytics_output = gr.Markdown()
                        alerts_output = gr.Markdown()
                
                analyze_btn.click(
                    process_image,
                    inputs=[input_image, crowd_check, criminal_check, anomaly_check, clean_check],
                    outputs=[output_image, analytics_output, alerts_output]
                )
            
            # ---- TAB 3: Face Registration ----
            with gr.Tab("👤 Face Registration"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🚨 Register Suspect")
                        criminal_image = gr.Image(label="Suspect Photo", type="numpy")
                        criminal_name = gr.Textbox(label="Suspect Name / ID")
                        register_criminal_btn = gr.Button("Register Suspect", variant="stop")
                        criminal_status = gr.Textbox(label="Status")
                    
                    with gr.Column():
                        gr.Markdown("### 👷 Register Railway Staff")
                        worker_image = gr.Image(label="Worker Photo", type="numpy")
                        worker_name = gr.Textbox(label="Worker Name / ID")
                        register_worker_btn = gr.Button("Register Worker", variant="primary")
                        worker_status = gr.Textbox(label="Status")
                
                register_criminal_btn.click(register_criminal, inputs=[criminal_image, criminal_name], outputs=[criminal_status])
                register_worker_btn.click(register_worker, inputs=[worker_image, worker_name], outputs=[worker_status])
                
    return demo

# Launch dashboard
demo = create_dashboard()
demo.launch(share=True, debug=True)


if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(share=False)
