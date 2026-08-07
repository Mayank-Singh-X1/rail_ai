# ============================================
# CELL 4: Criminal Detection Module (Vector Accelerated)
# ============================================
import os
import time
import cv2
import numpy as np


class CriminalDetector:

  def __init__(self, system, similarity_threshold=0.45):
    self.system = system
    self.similarity_threshold = similarity_threshold
    self.alert_cooldown = {}
    self.COOLDOWN_SECONDS = 30

  def compute_similarity(self, embedding1, embedding2):
    """Fast dot product similarity for normalized embeddings."""
    return np.dot(embedding1, embedding2)

  def detect_criminals(self, frame):
    """Detect faces and match embeddings against database in parallel."""
    faces = self.system.face_app.get(frame)
    matches = []
    annotated_frame = frame.copy()

    if not faces or not self.system.criminal_db:
      return annotated_frame, matches

    # Extract names and normalized embedding matrix from DB for batch dot product
    db_names = list(self.system.criminal_db.keys())
    db_matrix = np.array(
        list(self.system.criminal_db.values()), dtype=np.float32
    )

    for face in faces:
      bbox = face.bbox.astype(int)

      # Extract & normalize live face embedding
      live_embedding = face.embedding / np.linalg.norm(face.embedding)

      # Vectorized matrix-vector multiplication (Compute similarity across DB at once)
      similarities = np.dot(db_matrix, live_embedding)
      best_idx = np.argmax(similarities)
      best_score = similarities[best_idx]
      best_match = db_names[best_idx]

      if best_score > self.similarity_threshold:
        current_time = time.time()

        # Handle alert cooldown to prevent notification spam
        if (
            best_match not in self.alert_cooldown
            or (current_time - self.alert_cooldown[best_match])
            > self.COOLDOWN_SECONDS
        ):

          gender_str = "Male" if face.gender == 1 else "Female"
          matches.append({
              "name": best_match,
              "score": float(best_score),
              "bbox": bbox,
              "age": int(face.age),
              "gender": gender_str,
          })

          self.alert_cooldown[best_match] = current_time

          # Push alert to system queue
          self.system.alerts.append({
              "type": "🚨 CRIMINAL DETECTED",
              "name": best_match,
              "confidence": f"{best_score:.2f}",
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "location": "Platform Camera 1",
          })

        # Render RED alert box for identified suspect
        cv2.rectangle(
            annotated_frame,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 0, 255),
            3,
        )
        label = f"⚠️ SUSPECT: {best_match} ({best_score:.2f})"
        cv2.putText(
            annotated_frame,
            label,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
      else:
        # Render GREEN bounding box for normal passengers
        cv2.rectangle(
            annotated_frame,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),
            2,
        )
        info = f"Age:{int(face.age)} {'M' if face.gender==1 else 'F'}"
        cv2.putText(
            annotated_frame,
            info,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    return annotated_frame, matches

  def bulk_register_criminals(self, image_folder):
    """Batch process criminal mugshots from a target folder into DB."""
    if not os.path.exists(image_folder):
      print(f"⚠️ Image directory '{image_folder}' does not exist.")
      return

    for filename in os.listdir(image_folder):
      if filename.endswith((".jpg", ".png", ".jpeg")):
        name = os.path.splitext(filename)[0]
        filepath = os.path.join(image_folder, filename)
        self.system.add_criminal_to_db(name, filepath)
