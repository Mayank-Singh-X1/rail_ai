"""
Criminal Face Detection Test
==============================
This script lets you:
  1. Register YOUR OWN face as a "test criminal" 
  2. Then verify the system recognises you on the next webcam frame
  3. Remove yourself after testing

Run from d:\\SIHackathon:
    python test_criminal_detection.py

USAGE:
  - Press S to save a face snapshot as the enrolled criminal
  - Press D to run detection against enrolled faces
  - Press C to clear all enrolled faces
  - Press Q to quit
"""
import sys, os, time, cv2, numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'railway-surveillance-ai'))
sys.path.insert(0, os.getcwd())

print("\n========================================")
print("  Criminal Detection Live Test")
print("========================================")
print("Loading pipeline (first run may take ~10s)...\n")

from pipeline import system, pipeline

print("\n--- CONTROLS ---")
print("  S  →  Snapshot and ENROLL current face as 'TestCriminal_1'")
print("  D  →  Run DETECTION on current webcam frame and show matches")
print("  C  →  Clear all enrolled criminals from memory")
print("  Q  →  Quit")
print("----------------\n")

cap = None
for cam_idx in range(3):
    # Try DSHOW first (more compatible on Windows), then default
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        c = cv2.VideoCapture(cam_idx, backend)
        if c.isOpened():
            ret, test_frame = c.read()
            if ret and test_frame is not None:
                cap = c
                print(f"Camera {cam_idx} opened (backend={backend})")
                break
            c.release()
    if cap is not None:
        break

if cap is None:
    print("ERROR: No working webcam found.")
    print("  - Make sure no other app (Teams, browser, etc.) is using the camera")
    print("  - Try unplugging and re-plugging the webcam")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

SAVE_DIR = os.path.join("database", "criminals")
os.makedirs(SAVE_DIR, exist_ok=True)

enrolled_count = 0
last_result_text = "Press S to enroll your face, then D to detect."
last_faces_drawn = None

def enroll_face(frame, name):
    img_path = os.path.join(SAVE_DIR, f"{name}.jpg")
    cv2.imwrite(img_path, frame)
    success = system.add_criminal_to_db(name, img_path)
    if success:
        return f"ENROLLED: '{name}' added to DB  (total: {len(system.criminal_db)})"
    else:
        return f"FAIL: No clear face detected — move closer to camera and try again."

def run_detection(frame):
    if not system.criminal_db:
        return frame, "DB is empty — press S first to enroll a face."
    
    t0 = time.perf_counter()
    # Reuse the pipeline's detector instance (keeps cooldown state correctly)
    from modules.criminal_detector import CriminalDetector
    ann, matches = pipeline.criminal_detector.detect_criminals(frame)
    elapsed = (time.perf_counter() - t0) * 1000

    if matches:
        names = [m['name'] for m in matches]
        scores = [f"{m['score']:.3f}" for m in matches]
        txt = f"MATCH FOUND: {names}  score={scores}  ({elapsed:.0f}ms)"
    else:
        # Show all faces found (even if no criminal match)
        faces = system.face_app.get(frame)
        txt = f"No criminal match (threshold=0.45) — {len(faces)} face(s) detected  ({elapsed:.0f}ms)"
        col = (0, 200, 80)
        for f in faces:
            b = f.bbox.astype(int)
            cv2.rectangle(ann, (b[0], b[1]), (b[2], b[3]), col, 2)
            cv2.putText(ann, f"Age:{int(f.age)} {'M' if f.gender==1 else 'F'}",
                        (b[0], max(14, b[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return ann, txt


print("Webcam opened. Window should appear now...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    display = frame.copy() if last_faces_drawn is None else last_faces_drawn.copy()

    # Overlay enrolled count
    db_txt = f"DB: {len(system.criminal_db)} enrolled criminal(s)"
    cv2.putText(display, db_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

    # Overlay last result
    lines = last_result_text.split("  ")
    for i, line in enumerate(lines[:3]):
        cv2.putText(display, line, (10, 58 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1, cv2.LINE_AA)

    # Controls hint at bottom
    cv2.putText(display, "S=Enroll  D=Detect  C=Clear  Q=Quit",
                (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow("Criminal Detection Test  — Railway Surveillance AI", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        break

    elif key == ord('s'):
        enrolled_count += 1
        name = f"TestCriminal_{enrolled_count}"
        last_result_text = enroll_face(frame.copy(), name)
        last_faces_drawn = None
        print(f"  > {last_result_text}")

    elif key == ord('d'):
        ann, txt = run_detection(frame.copy())
        last_faces_drawn = ann
        last_result_text = txt
        print(f"  > {txt}")

    elif key == ord('c'):
        system.criminal_db.clear()
        enrolled_count = 0
        last_faces_drawn = None
        last_result_text = "Cleared all enrolled criminals from memory."
        print("  > Cleared.")

cap.release()
cv2.destroyAllWindows()
print("\nTest complete. Bye!")
