import cv2
from retinaface import RetinaFace


def detect_and_crop_face(image_path, output_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Detect faces
    detections = RetinaFace.detect_faces(image_path)

    if detections is None:
        raise ValueError("No face detected")

    # Take first detected face
    first_key = list(detections.keys())[0]
    x1, y1, x2, y2 = detections[first_key]["facial_area"]

    # Crop face
    face = image[y1:y2, x1:x2]

    # Resize to ArcFace input size
    face = cv2.resize(face, (112, 112))

    # Save cropped face
    cv2.imwrite(output_path, face)

    return output_path
