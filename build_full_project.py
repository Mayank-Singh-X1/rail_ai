import json
import os
import pickle
import shutil

ROOT = "railway-surveillance-ai"

# Ensure directories exist
directories = [
    f"{ROOT}/models",
    f"{ROOT}/modules",
    f"{ROOT}/database/criminals",
    f"{ROOT}/database/workers",
    f"{ROOT}/config",
    f"{ROOT}/dashboard/assets",
    f"{ROOT}/notebooks",
    f"{ROOT}/utils",
    f"{ROOT}/tests",
    f"{ROOT}/docs"
]

for d in directories:
    os.makedirs(d, exist_ok=True)

# Load notebook
nb = json.load(open('test.ipynb', encoding='utf-8'))

def cell_text(idx):
    return ''.join(nb['cells'][idx]['source'])

# -------------------------------------------------------------
# 1. config files
# -------------------------------------------------------------
zones_json = {
    "track_safety_zone": {
        "polygon": [[100, 400], [500, 400], [500, 600], [100, 600]],
        "type": "restricted"
    },
    "platform_edge_zone": {
        "polygon": [[50, 200], [600, 200], [600, 300], [50, 300]],
        "type": "caution"
    }
}
with open(f"{ROOT}/config/zones.json", "w", encoding="utf-8") as f:
    json.dump(zones_json, f, indent=2)

cameras_json = {
    "cam_01": {
        "name": "Platform 1 Main Stream",
        "source": "0",
        "fps": 30,
        "resolution": [1280, 720]
    },
    "cam_02": {
        "name": "Track Crossing Camera",
        "source": "assets/sample_feed.mp4",
        "fps": 25,
        "resolution": [1280, 720]
    }
}
with open(f"{ROOT}/config/cameras.json", "w", encoding="utf-8") as f:
    json.dump(cameras_json, f, indent=2)

thresholds_json = {
    "crowd": {
        "low_max": 5,
        "medium_max": 15
    },
    "criminal_similarity": 0.45,
    "worker_similarity": 0.45,
    "anomaly": {
        "fall_aspect_ratio": 1.3,
        "unattended_bag_seconds": 15
    },
    "cleanliness": {
        "warning_score": 70,
        "critical_score": 40
    }
}
with open(f"{ROOT}/config/thresholds.json", "w", encoding="utf-8") as f:
    json.dump(thresholds_json, f, indent=2)

# Database placeholders
with open(f"{ROOT}/database/embeddings.pkl", "wb") as f:
    pickle.dump({"criminals": {}, "workers": {}}, f)

with open(f"{ROOT}/database/criminals/.gitkeep", "w") as f:
    f.write("")

with open(f"{ROOT}/database/workers/.gitkeep", "w") as f:
    f.write("")

# Placeholders for models
for model_name in ["yolov8n.pt", "yolov8n-pose.pt", "custom_cleanliness.pt"]:
    model_path = f"{ROOT}/models/{model_name}"
    if not os.path.exists(model_path):
        with open(model_path, "w") as f:
            f.write("# Model placeholder weights file\n")

print("Created config and database placeholders.")
