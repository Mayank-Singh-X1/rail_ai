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


def create_worker_dashboard():
    def process_webcam(frame):
        if frame is None:
            return None, "⚠️ Waiting for webcam frame..."
        if isinstance(frame, dict):
            frame = frame.get('image', frame.get('composite', None))
        if frame is None:
            return None, "⚠️ Waiting for camera..."

        arr = cv2.cvtColor(np.array(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        res = pipeline.process_frame(arr, mode="worker")
        out_rgb = cv2.cvtColor(res['frame'], cv2.COLOR_BGR2RGB)

        present = res.get('workers_present', [])
        status = f"🟢 **STAFF ATTENDANCE ACTIVE** | Staff On Duty: `{len(present)}` ({', '.join(present) if present else 'None'}) | FPS: `{res['fps']:.1f}`"

        return out_rgb, status

    def register_staff(image, name, dept):
        if image is None or not name.strip():
            return "❌ Please provide photo and staff name/badge ID.", get_staff_roster()

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"worker_reg_{name.strip()}.jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        success = system.add_worker_to_db(name.strip(), temp_path)
        if success:
            msg = f"✅ **Staff Member '{name.strip()}' registered in Roster!**\n- Department: {dept or 'General Operations'}\n- Enrolled Officials: {len(system.worker_db)}"
            return msg, get_staff_roster()
        return "❌ Face not detected in photo.", get_staff_roster()

    def get_staff_roster():
        if not system.worker_db:
            return "📋 *No staff members currently registered in roster.*"
        md = "### 👷 Enrolled Railway Staff Roster\n\n| Official Name / Badge ID | Vector Embedding | Duty Status |\n|---|---|---|\n"
        for wname in system.worker_db.keys():
            md += f"| **{wname}** | 512-dim ArcFace | 🟢 Registered |\n"
        return md

    with gr.Blocks(title="Railway Staff & RPF Attendance — Indian Railways AI") as demo:
        gr.Markdown("""
        # 👷 Indian Railways AI — Staff & RPF Attendance System
        ### Dedicated Vector Face Recognition for Railway Officers, RPF Inspectors & Station Staff Duty Tracking
        """)

        with gr.Tabs():
            with gr.Tab("📹 Live Attendance Feed"):
                with gr.Row():
                    with gr.Column():
                        web_in = gr.Image(sources=["webcam"], streaming=True, label="📷 Live Gate / Station Camera")
                    with gr.Column():
                        web_out = gr.Image(label="🎯 Recognized Staff Overlay")
                        web_stat = gr.Markdown()
                web_in.stream(process_webcam, inputs=[web_in], outputs=[web_out, web_stat])
                web_in.change(process_webcam, inputs=[web_in], outputs=[web_out, web_stat])

            with gr.Tab("👷 Register Staff Member"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 👷 Enroll Railway Official into Roster")
                        w_img = gr.Image(label="Staff Official Photo", sources=["webcam", "upload"], type="numpy")
                        w_name = gr.Textbox(label="Official Name / Badge ID", placeholder="e.g. RPF_Inspector_Kumar")
                        w_dept = gr.Dropdown(
                            choices=["Railway Protection Force (RPF)", "Station Management", "Sanitation & Cleaning Crew", "Ticket Checking Staff"],
                            value="Railway Protection Force (RPF)",
                            label="Department / Role"
                        )
                        btn_reg = gr.Button("👷 Register Official into Roster", variant="primary")
                        reg_msg = gr.Markdown()
                    with gr.Column():
                        roster_view = gr.Markdown(value=get_staff_roster())

                btn_reg.click(register_staff, inputs=[w_img, w_name, w_dept], outputs=[reg_msg, roster_view])

    return demo


if __name__ == "__main__":
    demo = create_worker_dashboard()
    demo.launch(server_name="127.0.0.1", server_port=7864, share=False)
