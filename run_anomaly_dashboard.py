import os
import sys

dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'single_model_surveillance')
sys.path.insert(0, dir_path)

from dashboards.app_anomaly import create_anomaly_dashboard

if __name__ == "__main__":
    print("⚠️ Launching Dedicated Anomaly & Fall Detection Dashboard (Port 7862)...")
    demo = create_anomaly_dashboard()
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7862, share=False)
