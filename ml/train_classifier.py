import numpy as np
import joblib

def train_and_save_model(embeddings, output_path):
    """
    embeddings: (N, 512)
    Saves mean face embedding
    """
    embeddings = embeddings.squeeze()  # safety

    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

    joblib.dump(mean_embedding, output_path)
    return output_path
