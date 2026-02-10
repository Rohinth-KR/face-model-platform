import os
import cv2
import joblib
import numpy as np
from face_detector import detect_and_crop_face
from embedding_generator import generate_embeddings

MODEL_PATH = "ml/models/person_1_face_model.pkl"
TEST_DIR = "ml/data/test_images"
TEMP_DIR = "ml/data/temp_test_faces"

os.makedirs(TEMP_DIR, exist_ok=True)

# Load saved mean embedding (this is NOT an SVM)
mean_embedding = joblib.load(MODEL_PATH)

# Threshold (ArcFace cosine similarity)
THRESHOLD = 0.30   # <-- IMPORTANT (your scores show this is correct)

for img_name in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, img_name)
    face_path = os.path.join(TEMP_DIR, img_name)

    # 🔴 CLEAR TEMP DIR (CRITICAL FIX)
    for f in os.listdir(TEMP_DIR):
        os.remove(os.path.join(TEMP_DIR, f))

    try:
        # Detect & crop face
        detect_and_crop_face(img_path, face_path)

        # Generate embedding for THIS image only
        emb = generate_embeddings(TEMP_DIR)[0]

        # Cosine similarity
        similarity = np.dot(mean_embedding, emb)

        if similarity >= THRESHOLD:
            print(f"{img_name}: ✅ FACE MATCH (sim={similarity:.3f})")
        else:
            print(f"{img_name}: ❌ NO MATCH (sim={similarity:.3f})")

    except Exception as e:
        print(f"{img_name}: ERROR - {e}")
