# 🚂 Indian Railways AI Surveillance — Standalone Dedicated Dashboards

Each AI model in this project has been **decoupled into its own standalone dashboard application** with its own direct launcher command and dedicated port (Ports 7860–7865). This eliminates multi-model CPU/GPU thrashing completely and delivers **ultra-high FPS (50–110+ FPS)**.

---

## 🚀 Direct Launcher Commands

Run any of the 6 dedicated AI dashboards directly:

| AI Module | Direct Python Launcher Command | Dedicated Port | Focus & Purpose |
|---|---|---|---|
| 👥 **Crowd Analytics & ByteTrack** | `python run_crowd_dashboard.py` | `http://localhost:7860` | Passenger counting, ByteTrack, density heatmaps |
| 🚨 **Criminal Recognition** | `python run_criminal_dashboard.py` | `http://localhost:7861` | 512-dim ArcFace face matching, wanted alert banners |
| ⚠️ **Anomaly & Fall Detection** | `python run_anomaly_dashboard.py` | `http://localhost:7862` | YOLO Pose fall detection, fight velocity, intrusion |
| 🧹 **Cleanliness Scoring** | `python run_cleanliness_dashboard.py` | `http://localhost:7863` | 0-100% cleanliness score & litter item detection |
| 👷 **Staff Attendance** | `python run_worker_dashboard.py` | `http://localhost:7864` | RPF & railway staff face duty attendance |
| 🔫 **Weapon & Threat Scan** | `python run_weapon_dashboard.py` | `http://localhost:7865` | Knives, guns, scissors, bats threat scanner |

---

## ⚡ Alternative CLI Launcher Flags (`main.py`)

You can also launch any dashboard from `single_model_surveillance/`:

```bash
# Launch Crowd Analytics Dashboard (Port 7860)
python single_model_surveillance/main.py --crowd

# Launch Criminal Recognition Dashboard (Port 7861)
python single_model_surveillance/main.py --criminal

# Launch Anomaly Pose Detection Dashboard (Port 7862)
python single_model_surveillance/main.py --anomaly

# Launch Cleanliness Scoring Dashboard (Port 7863)
python single_model_surveillance/main.py --cleanliness

# Launch Staff Attendance Dashboard (Port 7864)
python single_model_surveillance/main.py --worker

# Launch Weapon Threat Scanner Dashboard (Port 7865)
python single_model_surveillance/main.py --weapon
```

---

## 📹 Direct Live Camera Stream Commands

Stream live webcam feed directly with a specific fine-tuned AI module:

```bash
# Criminal Recognition Live Camera
python single_model_surveillance/main.py --stream 0 --mode criminal

# Weapon Scan Live Camera
python single_model_surveillance/main.py --stream 0 --mode weapon

# Crowd Density Live Camera
python single_model_surveillance/main.py --stream 0 --mode crowd
```

---

## 🧪 Verification & Benchmarking

Run the pipeline test suite:
```bash
python single_model_surveillance/main.py --test
```

Generate side-by-side benchmark comparison chart:
```bash
python single_model_surveillance/benchmark_comparison.py
```
