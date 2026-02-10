import os
from face_detector import detect_and_crop_face
from embedding_generator import generate_embeddings
from train_classifier import train_and_save_model

RAW_DIR = "ml/data/raw_images/person_1"
FACE_DIR = "ml/data/processed_faces/person_1"
MODEL_DIR = "ml/models"

MODEL_PATH = os.path.join(MODEL_DIR, "person_1_face_model.pkl")

os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

for img_name in os.listdir(RAW_DIR):
    detect_and_crop_face(
        os.path.join(RAW_DIR, img_name),
        os.path.join(FACE_DIR, img_name)
    )

embeddings = generate_embeddings(FACE_DIR)
print("Embeddings shape:", embeddings.shape)

train_and_save_model(embeddings, MODEL_PATH)
print("✅ Model saved at:", MODEL_PATH)
