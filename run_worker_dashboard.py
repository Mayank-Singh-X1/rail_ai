import os
import sys

dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'single_model_surveillance')
sys.path.insert(0, dir_path)

from dashboards.app_worker import create_worker_dashboard

if __name__ == "__main__":
    print("👷 Launching Dedicated Staff & RPF Attendance Dashboard (Port 7864)...")
    demo = create_worker_dashboard()
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7864, share=False)
