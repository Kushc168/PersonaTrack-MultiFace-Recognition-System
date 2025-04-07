import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pickle
import logging
import face_alignment
from collections import defaultdict, deque
from ultralytics import YOLO  # Import YOLO for person detection


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMILARITY_THRESHOLD = 0.5  # Adjust based on testing
HISTORY_LENGTH = 12  # Temporal smoothing


# Initialize models
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLO("E:/yolov8n.pt")  # Load YOLO model for person detection


# Load known faces data
DATA_FILE = "known_faces.pkl"
known_embeddings = []
known_ids = []
id_counter = 1  # Assign new IDs dynamically
face_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))


if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        stored_data = pickle.load(f)
        known_embeddings = stored_data["embeddings"]
        known_ids = stored_data["ids"]
        id_counter = max(known_ids) + 1 if known_ids else 1
    logging.info("Loaded existing known faces.")

cap = cv2.VideoCapture(0)  # Open webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logging.warning("Failed to capture frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(rgb_frame)

    # Detect people using YOLO
    results = yolo_model(frame)
    person_boxes = [box.xyxy[0].cpu().numpy() for box in results[0].boxes if int(box.cls) == 0]  # Class 0 = Person

    # Detect faces
    boxes, _ = mtcnn.detect(pil_frame)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            face = pil_frame.crop((x1, y1, x2, y2)).resize((160, 160))

            # Convert face to tensor and generate embedding
            face_tensor = mtcnn(face)
            if face_tensor is None:
                continue

            # Face alignment
            landmarks = fa.get_landmarks(np.array(face))
            if landmarks is not None:
                face = face.rotate(-landmarks[0][0][0])  # Align using the first landmark

            embedding = resnet(face_tensor.unsqueeze(0).to(DEVICE)).detach().cpu().numpy().flatten()

            # Compare with known embeddings
            if known_embeddings:
                similarities = cosine_similarity([embedding], known_embeddings)
                max_sim = np.max(similarities)
                best_match_index = np.argmax(similarities)

                if max_sim > SIMILARITY_THRESHOLD:
                    recognized_id = known_ids[best_match_index]
                    face_history[recognized_id].append(recognized_id)
                    most_frequent_id = max(set(face_history[recognized_id]), key=face_history[recognized_id].count)
                    recognized_id = most_frequent_id
                    color = (0, 255, 0)  # Green for known faces
                    logging.info(f"Recognized existing face: ID-{recognized_id}")
                else:
                    recognized_id = f"New-{id_counter}"
                    known_embeddings.append(embedding)
                    known_ids.append(id_counter)
                    id_counter += 1
                    color = (0, 0, 255)  # Red for new faces
                    logging.info(f"New face detected, assigning ID-{recognized_id}")
            else:
                recognized_id = f"New-{id_counter}"
                known_embeddings.append(embedding)
                known_ids.append(id_counter)
                id_counter += 1
                color = (0, 0, 255)  # Red for new faces
                logging.info(f"New face detected, assigning ID-{recognized_id}")

            # Find the closest person bounding box
            best_match_box = None
            for p_box in person_boxes:
                px1, py1, px2, py2 = map(int, p_box)
                if x1 >= px1 and y1 >= py1 and x2 <= px2 and y2 <= py2:
                    best_match_box = (px1, py1, px2, py2)
                    break

            if best_match_box:
                cv2.rectangle(frame, (best_match_box[0], best_match_box[1]), (best_match_box[2], best_match_box[3]), color, 2)
                cv2.putText(frame, f"Person ID-{recognized_id}", (best_match_box[0], best_match_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show frame
    cv2.imshow("Real-Time Face & Person Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save updated face data
with open(DATA_FILE, "wb") as f:
    pickle.dump({"embeddings": known_embeddings, "ids": known_ids}, f)
    logging.info("Saved updated known faces.")

cap.release()
cv2.destroyAllWindows()



#|||||| Prominent Good Accurate Code |||||
# import os
# import cv2
# import numpy as np
# import torch
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from sklearn.metrics.pairwise import cosine_similarity
# from PIL import Image
# import pickle
# import logging
#
# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SIMILARITY_THRESHOLD = 0.75  # Adjust based on testing
#
# # Initialize models
# mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
# resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
#
# # Load known faces data
# DATA_FILE = "known_faces.pkl"
# known_embeddings = []
# known_ids = []
# id_counter = 1  # Assign new IDs dynamically
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
#     # Detect faces
#     boxes, _ = mtcnn.detect(pil_frame)
#     if boxes is not None:
#         for box in boxes:
#             x1, y1, x2, y2 = [int(coord) for coord in box]
#             face = pil_frame.crop((x1, y1, x2, y2)).resize((160, 160))
#
#             # Convert face to tensor and generate embedding
#             face_tensor = mtcnn(face)
#             if face_tensor is None:
#                 continue
#
#             embedding = resnet(face_tensor.unsqueeze(0).to(DEVICE)).detach().cpu().numpy().flatten()
#
#             # Compare with known embeddings
#             if known_embeddings:
#                 similarities = cosine_similarity([embedding], known_embeddings)
#                 max_sim = np.max(similarities)
#                 best_match_index = np.argmax(similarities)
#
#                 if max_sim > SIMILARITY_THRESHOLD:
#                     recognized_id = known_ids[best_match_index]
#                     color = (0, 255, 0)  # Green for known faces
#                     display_text = f"Hey ID-{recognized_id}"
#                     logging.info(f"Recognized existing face: ID-{recognized_id}")  # Print in console
#                 else:
#                     recognized_id = f"New-{id_counter}"
#                     known_embeddings.append(embedding)
#                     known_ids.append(id_counter)
#                     id_counter += 1
#                     color = (0, 0, 255)  # Red for new faces
#                     display_text = f"New Face-{recognized_id}"
#                     logging.info(f"New face detected, assigning ID-{recognized_id}")  # Print in console
#
#             else:
#                 recognized_id = f"New-{id_counter}"
#                 known_embeddings.append(embedding)
#                 known_ids.append(id_counter)
#                 id_counter += 1
#                 color = (0, 0, 255)  # Red for new faces
#                 display_text = f"New Face-{recognized_id}"
#                 logging.info(f"New face detected, assigning ID-{recognized_id}")  # Print in console
#
#             # Draw bounding box and label
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#
#     # Show frame
#     cv2.imshow("Real-Time Face Recognition", frame)
#
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


# import os
# import cv2
# import numpy as np
# import torch
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics.pairwise import cosine_similarity
# from tkinter import Tk, simpledialog, messagebox
# from PIL import Image
# import pickle
# import logging
#
# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#
# # =======================
# # Configuration Settings
# # =======================
# KNOWN_FACES_PATH = 'known_faces.pkl'
# SCALER_PATH = 'scaler.pkl'
# CLASSIFIER_PATH = 'svm_classifier.pkl'
# LABEL_DICT_PATH = 'label_dict.pkl'
# SIMILARITY_THRESHOLD = 0.5
#
# # Initialize device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logging.info(f"Using device: {device}")
#
# # =======================
# # Initialize Models
# # =======================
# mtcnn = MTCNN(image_size=160, margin=20, device=device)
# resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#
# # =======================
# # Utility Functions
# # =======================
#
# def list_available_cameras(max_cameras=5):
#     """Lists available camera indices."""
#     available_cameras = []
#     for index in range(max_cameras):
#         cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
#         if cap.isOpened():
#             ret, _ = cap.read()
#             if ret:
#                 available_cameras.append(index)
#             cap.release()
#     return available_cameras
#
# def select_camera():
#     """Prompts the user to select a camera."""
#     Tk().withdraw()  # Hide the root window
#     available_cameras = list_available_cameras()
#
#     if not available_cameras:
#         messagebox.showerror("No Cameras Found", "No webcams were detected on this system.")
#         logging.error("No webcams found.")
#         return None
#
#     camera_selection = simpledialog.askinteger(
#         "Select Camera",
#         f"Available Cameras: {available_cameras}\nEnter camera index to use:"
#     )
#
#     if camera_selection not in available_cameras:
#         messagebox.showerror("Invalid Selection", f"Camera index {camera_selection} is not available.")
#         logging.error(f"Invalid camera index selected: {camera_selection}")
#         return None
#
#     return camera_selection
#
# def load_file(file_path, description):
#     """Load a file and handle errors."""
#     if not os.path.exists(file_path):
#         logging.error(f"{description} file not found: {file_path}")
#         return None
#     with open(file_path, 'rb') as f:
#         return pickle.load(f)
#
# def save_file(data, file_path, description):
#     """Save a file and handle errors."""
#     try:
#         with open(file_path, 'wb') as f:
#             pickle.dump(data, f)
#         logging.info(f"{description} saved successfully at {file_path}")
#     except Exception as e:
#         logging.error(f"Error saving {description}: {e}")
#
# # =======================
# # Core Functions
# # =======================
#
# def generate_embeddings(data_path):
#     """Generate embeddings from face images."""
#     logging.info("Generating embeddings...")
#     embeddings, names = [], []
#     for person in os.listdir(data_path):
#         person_path = os.path.join(data_path, person)
#         if not os.path.isdir(person_path):
#             continue
#         for image_name in os.listdir(person_path):
#             image_path = os.path.join(person_path, image_name)
#             try:
#                 image = Image.open(image_path).convert('RGB')
#                 face = mtcnn(image)
#                 if face is not None:
#                     embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
#                     embeddings.append(embedding.flatten())
#                     names.append(person)
#             except Exception as e:
#                 logging.warning(f"Error processing {image_path}: {e}")
#     save_file({'embeddings': embeddings, 'names': names}, KNOWN_FACES_PATH, "Known faces")
#     return np.array(embeddings), names
#
# def train_model(data_path):
#     """Train an SVM classifier on generated embeddings."""
#     embeddings, names = generate_embeddings(data_path)
#     if embeddings.size == 0:
#         logging.error("No embeddings generated. Ensure images are correctly formatted.")
#         return
#
#     scaler = StandardScaler()
#     embeddings_scaled = scaler.fit_transform(embeddings)
#     save_file(scaler, SCALER_PATH, "Scaler")
#
#     label_dict = {name: idx for idx, name in enumerate(set(names))}
#     labels = np.array([label_dict[name] for name in names])
#
#     classifier = SVC(kernel='linear', probability=True)
#     classifier.fit(embeddings_scaled, labels)
#     save_file(classifier, CLASSIFIER_PATH, "SVM classifier")
#     save_file(label_dict, LABEL_DICT_PATH, "Label dictionary")
#
# def recognize_faces():
#     """Perform real-time face recognition using webcam."""
#     scaler = load_file(SCALER_PATH, "Scaler")
#     classifier = load_file(CLASSIFIER_PATH, "SVM classifier")
#     label_dict = load_file(LABEL_DICT_PATH, "Label dictionary")
#     if not scaler or not classifier or not label_dict:
#         logging.error("Failed to load required files. Ensure training is complete.")
#         return
#
#     label_dict_inv = {v: k for k, v in label_dict.items()}
#     camera_index = select_camera()
#     if camera_index is None:
#         logging.info("No camera selected. Exiting...")
#         return
#
#     cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
#     if not cap.isOpened():
#         logging.error(f"Could not access the webcam at index {camera_index}.")
#         return
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logging.warning("Failed to capture frame.")
#             break
#
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_frame = Image.fromarray(rgb_frame)
#         boxes, _ = mtcnn.detect(pil_frame)
#
#         if boxes is not None:
#             for box in boxes:
#                 x1, y1, x2, y2 = [int(coord) for coord in box]
#                 face = pil_frame.crop((x1, y1, x2, y2)).resize((160, 160))
#                 face_tensor = mtcnn.face_detector.transform(face).to(device)
#                 embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()
#                 embedding_scaled = scaler.transform(embedding)
#                 prediction = classifier.predict_proba(embedding_scaled)[0]
#                 max_prob = np.max(prediction)
#                 predicted_label = np.argmax(prediction)
#
#                 name = label_dict_inv[predicted_label] if max_prob >= SIMILARITY_THRESHOLD else "Unknown"
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{name} ({max_prob:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#         cv2.imshow("Face Recognition", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# # =======================
# # Main Execution
# # =======================
# if __name__ == "__main__":
#     action = input("Enter 'train' to train the model or 'recognize' to start recognition: ").strip().lower()
#     if action == 'train':
#         data_path = input("Enter the path to the dataset: ").strip()
#         train_model(data_path)
#     elif action == 'recognize':
#         recognize_faces()
#     else:
#         logging.error("Invalid action. Use 'train' or 'recognize'.")