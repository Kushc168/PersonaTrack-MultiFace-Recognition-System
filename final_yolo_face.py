import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pickle
import logging
from collections import defaultdict, deque
from ultralytics import YOLO  # YOLO for face & person detection

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMILARITY_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3  # Minimum IoU to consider same person
HISTORY_LENGTH = 26  # Smooth ID assignment

# Load YOLO models
yolo_person_detector = YOLO("E:/yolov8n.pt")  # Person detection
yolo_face_detector = YOLO("E:/yolov8n-face.pt")  # YOLO Face Model

# Load FaceNet model for embedding generation
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

# Load known faces data
DATA_FILE = "known_faces.pkl"
known_embeddings = []
known_ids = []
id_counter = 1
stable_person_map = {}  # Persistent mapping of persons to their assigned IDs
face_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))



if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        stored_data = pickle.load(f)
        known_embeddings = stored_data["embeddings"]
        known_ids = stored_data["ids"]
        id_counter = max(known_ids) + 1 if known_ids else 1
    logging.info("Loaded existing known faces.")

cap = cv2.VideoCapture(0)  # Open webcam

def compute_iou(box1, box2):
    """ Compute Intersection Over Union (IoU) between two bounding boxes. """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union = area1 + area2 - intersection

    return intersection / union if union else 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logging.warning("Failed to capture frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(rgb_frame)

    # Detect people using YOLO
    person_results = yolo_person_detector.track(frame, persist=True , verbose=False)
    person_boxes = [box.xyxy[0].cpu().numpy() for box in person_results[0].boxes if int(box.cls) == 0]  # Class 0 = Person

    # Detect faces using YOLO
    face_results = yolo_face_detector.track(frame, persist=True, verbose=False)
    face_boxes = [box.xyxy[0].cpu().numpy() for box in face_results[0].boxes]  # Assuming face model outputs faces

    face_embeddings = {}

    # Process each detected face
    for box in face_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        face = pil_frame.crop((x1, y1, x2, y2)).resize((160, 160))

        # Convert face to tensor and generate embedding
        face_tensor = torch.tensor(np.array(face)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
        face_embeddings[tuple(box)] = embedding  # Store embedding for later matching

    # Match face to existing known embeddings
    for face_box, embedding in face_embeddings.items():
        recognized_id = None
        if known_embeddings:
            similarities = cosine_similarity([embedding], known_embeddings)
            max_sim = np.max(similarities)
            best_match_index = np.argmax(similarities)

            if max_sim > SIMILARITY_THRESHOLD:
                recognized_id = known_ids[best_match_index]
            else:
                recognized_id = id_counter
                known_embeddings.append(embedding)
                known_ids.append(id_counter)
                id_counter += 1
        else:
            recognized_id = id_counter
            known_embeddings.append(embedding)
            known_ids.append(id_counter)
            id_counter += 1

        # Find the closest person bounding box
        best_match_person_box = None
        highest_iou = 0.0
        for p_box in person_boxes:
            iou = compute_iou(face_box, p_box)
            if iou > highest_iou:
                highest_iou = iou
                best_match_person_box = tuple(p_box)

        # Assign recognized ID to the full person detection
        if best_match_person_box:
            # Keep the same ID if the person was previously assigned
            if best_match_person_box in stable_person_map:
                stable_person_map[best_match_person_box] = stable_person_map[best_match_person_box]
            else:
                stable_person_map[best_match_person_box] = recognized_id

    # Draw results
    for p_box in person_boxes:
        px1, py1, px2, py2 = map(int, p_box)
        assigned_id = stable_person_map.get(tuple(p_box), "Unknown")
        color = (0, 255, 0) if assigned_id != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        cv2.putText(frame, f"Person ID-{assigned_id}", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Real-Time Face & Person Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save updated face data
with open(DATA_FILE, "wb") as f:
    pickle.dump({"embeddings": known_embeddings, "ids": known_ids}, f)
    logging.info("Saved updated known faces.")

cap.release()
cv2.destroyAllWindows()


#--- USing traditional IOU ---#
# import os
# import cv2
# import numpy as np
# import torch
# from facenet_pytorch import InceptionResnetV1
# from sklearn.metrics.pairwise import cosine_similarity
# from PIL import Image
# import pickle
# import logging
# from collections import defaultdict, deque
# from ultralytics import YOLO  # YOLO for face & person detection
#
# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SIMILARITY_THRESHOLD = 0.5
# HISTORY_LENGTH = 12  # Smooth ID assignment
#
# # Load YOLO models
# yolo_person_detector = YOLO("E:/yolov8n.pt")  # Person detection
# yolo_face_detector = YOLO("E:/yolov8n-face.pt")  # YOLO Face Model
#
# # Load FaceNet model for embedding generation
# resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
#
# # Load known faces data
# DATA_FILE = "known_faces.pkl"
# known_embeddings = []
# known_ids = []
# id_counter = 1
# person_id_map = {}  # Maps person bounding box to an ID
# face_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
#
# if os.path.exists(DATA_FILE):
#     with open(DATA_FILE, "rb") as f:
#         stored_data = pickle.load(f)
#         known_embeddings = stored_data["embeddings"]
#         known_ids = stored_data["ids"]
#         id_counter = max(known_ids) + 1 if known_ids else 1
#     logging.info("Loaded existing known faces.")
#
# cap = cv2.VideoCapture(0)  # Open webcam
#
# def compute_iou(box1, box2):
#     """ Compute Intersection Over Union (IoU) between two bounding boxes. """
#     x1, y1, x2, y2 = box1
#     x1_p, y1_p, x2_p, y2_p = box2
#
#     xi1 = max(x1, x1_p)
#     yi1 = max(y1, y1_p)
#     xi2 = min(x2, x2_p)
#     yi2 = min(y2, y2_p)
#
#     intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
#     area1 = (x2 - x1) * (y2 - y1)
#     area2 = (x2_p - x1_p) * (y2_p - y1_p)
#     union = area1 + area2 - intersection
#
#     return intersection / union if union else 0
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         logging.warning("Failed to capture frame.")
#         break
#
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_frame = Image.fromarray(rgb_frame)
#
#     # Detect people using YOLO
#     person_results = yolo_person_detector(frame)
#     person_boxes = [box.xyxy[0].cpu().numpy() for box in person_results[0].boxes if int(box.cls) == 0]  # Class 0 = Person
#
#     # Detect faces using YOLO
#     face_results = yolo_face_detector(frame)
#     face_boxes = [box.xyxy[0].cpu().numpy() for box in face_results[0].boxes]  # Assuming face model outputs faces
#
#     face_embeddings = {}
#
#     # Process each detected face
#     for box in face_boxes:
#         x1, y1, x2, y2 = [int(coord) for coord in box]
#         face = pil_frame.crop((x1, y1, x2, y2)).resize((160, 160))
#
#         # Convert face to tensor and generate embedding
#         face_tensor = torch.tensor(np.array(face)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
#         embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
#         face_embeddings[tuple(box)] = embedding  # Store embedding for later matching
#
#     # Match face to existing known embeddings
#     for face_box, embedding in face_embeddings.items():
#         recognized_id = None
#         if known_embeddings:
#             similarities = cosine_similarity([embedding], known_embeddings)
#             max_sim = np.max(similarities)
#             best_match_index = np.argmax(similarities)
#
#             if max_sim > SIMILARITY_THRESHOLD:
#                 recognized_id = known_ids[best_match_index]
#                 logging.info(f"Recognized existing face: ID-{recognized_id}")
#             else:
#                 recognized_id = id_counter
#                 known_embeddings.append(embedding)
#                 known_ids.append(id_counter)
#                 id_counter += 1
#                 logging.info(f"New face detected, assigning ID-{recognized_id}")
#         else:
#             recognized_id = id_counter
#             known_embeddings.append(embedding)
#             known_ids.append(id_counter)
#             id_counter += 1
#             logging.info(f"New face detected, assigning ID-{recognized_id}")
#
#         # Find the closest person bounding box
#         best_match_person_box = None
#         highest_iou = 0.0
#         for p_box in person_boxes:
#             iou = compute_iou(face_box, p_box)
#             if iou > highest_iou:
#                 highest_iou = iou
#                 best_match_person_box = tuple(p_box)
#
#         # Assign recognized ID to the full person detection
#         if best_match_person_box:
#             person_id_map[best_match_person_box] = recognized_id
#
#     # Draw results
#     for p_box in person_boxes:
#         px1, py1, px2, py2 = map(int, p_box)
#         assigned_id = person_id_map.get(tuple(p_box), "Unknown")
#         color = (0, 255, 0) if assigned_id != "Unknown" else (0, 0, 255)
#         cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
#         cv2.putText(frame, f"Person ID-{assigned_id}", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#
#     cv2.imshow("Real-Time Face & Person Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Save updated face data
# with open(DATA_FILE, "wb") as f:
#     pickle.dump({"embeddings": known_embeddings, "ids": known_ids}, f)
#     logging.info("Saved updated known faces.")
#
# cap.release()
# cv2.destroyAllWindows()
