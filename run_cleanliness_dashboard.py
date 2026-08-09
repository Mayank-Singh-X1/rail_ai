import os
import sys

dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'single_model_surveillance')
sys.path.insert(0, dir_path)

from dashboards.app_cleanliness import create_cleanliness_dashboard

if __name__ == "__main__":
    print("🧹 Launching Dedicated Cleanliness Monitoring Dashboard (Port 7863)...")
    demo = create_cleanliness_dashboard()
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7863, share=False)
