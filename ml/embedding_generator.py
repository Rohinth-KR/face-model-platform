import os
import cv2
import numpy as np
from insightface.model_zoo import get_model

# Absolute path to ArcFace ONNX model
MODEL_PATH = os.path.expanduser(
    "~/.insightface/models/buffalo_l/w600k_r50.onnx"
)

# Load ArcFace model
model = get_model(MODEL_PATH)
model.prepare(ctx_id=0)


def generate_embeddings(face_dir):
    """
    Generates L2-normalized ArcFace embeddings
    Returns shape: (N, 512)
    """
    embeddings = []

    for img_name in os.listdir(face_dir):
        img_path = os.path.join(face_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract embedding
        embedding = model.get_feat(img)          # (1, 512)
        embedding = embedding.squeeze()           # (512,)
        embedding = embedding / np.linalg.norm(embedding)  # L2 normalize

        embeddings.append(embedding)

    return np.array(embeddings)  # (N, 512)
