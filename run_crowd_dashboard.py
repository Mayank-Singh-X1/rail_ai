import os
import sys

dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'single_model_surveillance')
sys.path.insert(0, dir_path)

from dashboards.app_crowd import create_crowd_dashboard

if __name__ == "__main__":
    print("👥 Launching Dedicated Crowd Analytics & Density Dashboard (Port 7860)...")
    demo = create_crowd_dashboard()
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
