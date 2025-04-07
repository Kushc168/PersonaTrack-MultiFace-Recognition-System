import cv2
import numpy as np
from deepface import DeepFace
from bytetracker import BYTETracker
from db_operations import save_face_to_db, get_all_embeddings, cleanup_invalid_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import tensorflow as tf

# Load YOLO model (pre-trained for face detection)
yolo_model = YOLO("E:/yolov8n-face.pt")  # Replace with your YOLO face model path

# Initialize tracker (ByteTrack)
tracker = BYTETracker(track_thresh=0.6, match_thresh=0.5, track_buffer=50)

# Function to detect faces using YOLO
def detect_faces_yolo(frame):
    results = yolo_model(frame)  # Perform inference with YOLO
    detections = []

    for result in results[0].boxes:  # Access boxes from the first result
        x_min, y_min, x_max, y_max = map(int, result.xyxy[0])  # Bounding box coordinates
        confidence = result.conf[0].item()  # Confidence score
        class_label = 0  # Assuming a single class for faces
        detections.append([x_min, y_min, x_max, y_max, confidence, class_label])

    return np.array(detections)

# Function to preprocess face for DeepFace
def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32') / 255.0
    face = (face - 0.5) / 0.5  # Standard normalization
    return face

# Function to generate embeddings using DeepFace (ArcFace)
def get_embedding(face):
    try:
        preprocessed_face = preprocess_face(face)
        result = DeepFace.represent(preprocessed_face, model_name="ArcFace", enforce_detection=False)
        embedding = np.array(result[0]["embedding"])
        print(f"Generated embedding: {embedding.shape}")
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Function to match face against database
def match_face(embedding, database, threshold=0.5):
    print(f"Matching embedding against database with {len(database)} entries.")
    for unique_id, saved_embedding in database:
        if embedding.shape != saved_embedding.shape:
            print(f"Skipping mismatched embedding for ID {unique_id}")
            continue
        similarity = cosine_similarity([embedding], [saved_embedding])[0][0]
        print(f"ID {unique_id}: Similarity = {similarity}")
        if similarity > threshold:
            return unique_id, similarity
    return None, None

# Real-time face recognition
def recognize_faces():
    database = get_all_embeddings()
    print(f"Loaded {len(database)} embeddings from database.")
    cap = cv2.VideoCapture(0)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_faces_yolo(frame)

        if detections.shape[0] > 0:
            detections_tensor = tf.convert_to_tensor(detections, dtype=tf.float32)
            tracks = tracker.update(detections_tensor, frame_id)
            frame_id += 1
        else:
            tracks = []

        for track in tracks:
            x_min, y_min, x_max, y_max, track_id = map(int, track[:5])
            face = frame[y_min:y_max, x_min:x_max]

            if face.size == 0:
                continue

            embedding = get_embedding(face)
            if embedding is None:
                continue

            unique_id, similarity = match_face(embedding, database)
            if unique_id:
                label = f"ID: {unique_id}, Sim: {similarity:.2f}"
            else:
                unique_id = len(database) + 1
                save_face_to_db(embedding, unique_id)
                label = f"New ID: {unique_id}"
                database.append((unique_id, embedding))

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start real-time recognition
if __name__ == "__main__":
    cleanup_invalid_embeddings()
    recognize_faces()


# import cv2
# import numpy as np
# from mtcnn import MTCNN
# from tensorflow.keras.models import load_model
# from deepface import DeepFace
# from bytetracker import BYTETracker
# from db_operations import save_face_to_db, get_all_embeddings, cleanup_invalid_embeddings
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Initialize face detector (MTCNN)
# detector = MTCNN()
#
# # Initialize tracker (ByteTrack)
# tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)
#
# # Function to preprocess face for DeepFace
# def preprocess_face(face):
#     face = cv2.resize(face, (160, 160))
#     face = face.astype('float32') / 255.0
#     mean, std = face.mean(), face.std()
#     face = (face - mean) / std
#     return face
#
# # Function to generate embeddings using DeepFace (ArcFace)
# def get_embedding(face):
#     preprocessed_face = preprocess_face(face)
#     result = DeepFace.represent(preprocessed_face, model_name="ArcFace", enforce_detection=False)
#     return np.array(result[0]["embedding"])
#
# # Function to match face against database
# def match_face(embedding, database, threshold=0.6):
#     for unique_id, saved_embedding in database:
#         similarity = cosine_similarity([embedding], [saved_embedding])[0][0]
#         if similarity > threshold:
#             return unique_id, similarity
#     return None, None
#
# # Real-time face recognition
# def recognize_faces():
#     database = get_all_embeddings()  # Load embeddings from MongoDB
#     cap = cv2.VideoCapture(0)
#     frame_id = 0  # Initialize frame ID for ByteTrack
#
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Detect faces
#         results = detector.detect_faces(frame)
#         detections = []
#
#         for result in results:
#             x, y, width, height = result['box']
#             confidence = result['confidence']
#
#             # Ensure valid box dimensions
#             x, y, width, height = max(0, x), max(0, y), max(0, width), max(0, height)
#             detections.append([x, y, x + width, y + height, confidence])
#
#         # Convert detections to NumPy array
#         # detections = np.array(detections)
#         detections = np.array([det + [0] for det in detections]) if len(detections) > 0 else np.empty((0, 6))
#
#         # Track faces using ByteTrack
#         tracks = tracker.update(detections, frame_id)
#         frame_id += 1
#
#         # Process each track
#         for track in tracks:
#             x_min, y_min, x_max, y_max, track_id = map(int, track[:5])
#             face = frame[y_min:y_max, x_min:x_max]
#
#             # Skip invalid face regions
#             if face.size == 0 or y_min < 0 or x_min < 0:
#                 continue
#
#             # Generate embedding for the face
#             embedding = get_embedding(face)
#
#             # Match face with database
#             unique_id, similarity = match_face(embedding, database)
#
#             if unique_id:
#                 label = f"ID: {unique_id}, Sim: {similarity:.2f}"
#             else:
#                 # Assign new ID and save face to database
#                 unique_id = len(database) + 1
#                 save_face_to_db(embedding, unique_id)
#                 label = f"New ID: {unique_id}"
#                 database.append((unique_id, embedding))  # Update local database
#
#             # Draw bounding box and label
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#         # Display the frame
#         cv2.imshow('Face Recognition', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# # Start real-time recognition
# if __name__ == "__main__":
#     cleanup_invalid_embeddings()  # Ensure database is clean
#     recognize_faces()

# import cv2
# import numpy as np
# from mtcnn import MTCNN
# from tensorflow.keras.models import load_model
# from deepface import DeepFace
# from bytetracker import BYTETracker
# from db_operations import save_face_to_db, get_all_embeddings, cleanup_invalid_embeddings
# from sklearn.metrics.pairwise import cosine_similarity
# import tensorflow as tf
#
# # Initialize face detector (MTCNN)
# detector = MTCNN()
#
# # Initialize tracker (ByteTrack)
# tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)
#
# # Function to preprocess face for DeepFace
# def preprocess_face(face):
#     face = cv2.resize(face, (160, 160))
#     face = face.astype('float32') / 255.0
#     mean, std = face.mean(), face.std()
#     face = (face - mean) / std
#     return face
#
# # Function to generate embeddings using DeepFace (ArcFace)
# def get_embedding(face):
#     preprocessed_face = preprocess_face(face)
#     result = DeepFace.represent(preprocessed_face, model_name="ArcFace", enforce_detection=False)
#     return np.array(result[0]["embedding"])
#
# # Function to match face against database
# def match_face(embedding, database, threshold=0.5):
#     for unique_id, saved_embedding in database:
#         print(f"Embedding shape: {embedding.shape}, Saved embedding shape: {saved_embedding.shape}")
#         if embedding.shape != saved_embedding.shape:
#             print(f"Skipping ID {unique_id} due to mismatched embedding size.")
#             continue  # Skip mismatched embeddings
#
#         similarity = cosine_similarity([embedding], [saved_embedding])[0][0]
#         if similarity > threshold:
#             return unique_id, similarity
#     return None, None
#
# # Real-time face recognition
# def recognize_faces():
#     database = get_all_embeddings()  # Load embeddings from MongoDB
#     cap = cv2.VideoCapture("D:/smoke2.mp4")
#     frame_id = 0  # Initialize frame ID for ByteTrack
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Detect faces
#         results = detector.detect_faces(frame)
#         detections = []
#
#         for result in results:
#             x, y, width, height = result['box']
#             confidence = result['confidence']
#
#             class_label = 0
#             detections.append([x, y, x + width, y + height, confidence, class_label])
#
#         # Convert detections to NumPy array
#         detections = np.array(detections)
#         detections = tf.convert_to_tensor(detections, dtype=tf.float32)
#
#         # Check if detections are valid before updating the tracker
#         if detections.shape[0] > 0:
#             tracks = tracker.update(detections, frame_id)
#             frame_id += 1
#         else:
#             tracks = []  # No detections, no tracking
#
#         # Process each track
#         for track in tracks:
#             x_min, y_min, x_max, y_max, track_id = map(int, track[:5])
#             face = frame[y_min:y_max, x_min:x_max]
#
#             # Skip if the face is not valid
#             if face.size == 0:
#                 continue
#
#             # Generate embedding for the face
#             embedding = get_embedding(face)
#
#             # Match face with database
#             unique_id, similarity = match_face(embedding, database)
#             if unique_id:
#                 label = f"ID: {unique_id}, Sim: {similarity:.2f}"
#             else:
#                 # Assign new ID and save face to database
#                 unique_id = len(database) + 1
#                 save_face_to_db(embedding, unique_id)
#                 label = f"New ID: {unique_id}"
#                 database.append((unique_id, embedding))  # Update local database
#
#             # Draw bounding box and label
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#         # Display the frame
#         cv2.imshow('Face Recognition', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# # Start real-time recognition
# if __name__ == "__main__":
#     cleanup_invalid_embeddings()  # Ensure database is clean
#     recognize_faces()

# import cv2
# import numpy as np
# from mtcnn import MTCNN
# from tensorflow.keras.models import load_model
# from deepface import DeepFace
# from bytetracker import BYTETracker
# from db_operations import save_face_to_db, get_all_embeddings
# from sklearn.metrics.pairwise import cosine_similarity
# import tensorflow as tf
#
#
# # Initialize face detector (MTCNN)
# detector = MTCNN()
#
# # Initialize tracker (ByteTrack)
# tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)
#
# # Function to preprocess face for DeepFace
# def preprocess_face(face):
#     face = cv2.resize(face, (160, 160))
#     face = face.astype('float32') / 255.0
#     mean, std = face.mean(), face.std()
#     face = (face - mean) / std
#     return face
#
# # Function to generate embeddings using DeepFace (ArcFace)
# def get_embedding(face):
#     preprocessed_face = preprocess_face(face)
#     result = DeepFace.represent(preprocessed_face, model_name="ArcFace", enforce_detection=False)
#     return np.array(result[0]["embedding"])
#
#
# # Function to match face against database
# def match_face(embedding, database, threshold=0.5):
#     for unique_id, saved_embedding in database:
#         print(f"Embedding shape: {embedding.shape}, Saved embedding shape: {saved_embedding.shape}")
#
#         similarity = cosine_similarity([embedding], [saved_embedding])[0][0]
#         if similarity > threshold:
#             return unique_id, similarity
#     return None, None
#
# # Real-time face recognition
# def recognize_faces():
#     database = get_all_embeddings()  # Load embeddings from MongoDB
#     cap = cv2.VideoCapture(0)
#     frame_id = 0  # Initialize frame ID for ByteTrack
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Detect faces
#         results = detector.detect_faces(frame)
#         detections = []
#
#         for result in results:
#             x, y, width, height = result['box']
#             confidence = result['confidence']
#
#             class_label = 0
#             detections.append([x, y, x + width, y + height, confidence,class_label])
#
#         # Convert detections to NumPy array
#         detections = np.array(detections)
#
#         detections = tf.convert_to_tensor(detections, dtype=tf.float32)
#
#
#
#         # # Update tracker
#         # tracks = tracker.update(detections, frame_id)
#         # frame_id += 1
#
#         # Check if detections are valid before updating the tracker
#         if detections.shape[0] > 0:
#             tracks = tracker.update(detections, frame_id)
#             frame_id += 1
#         else:
#             tracks = []  # No detections, no tracking
#
#         # Process each track
#         for track in tracks:
#             x_min, y_min, x_max, y_max, track_id = map(int, track[:5])
#             face = frame[y_min:y_max, x_min:x_max]
#
#             # Skip if the face is not valid
#             if face.size == 0:
#                 continue
#
#             # Generate embedding for the face
#             embedding = get_embedding(face)
#
#             # Match face with database
#             unique_id, similarity = match_face(embedding, database)
#             if unique_id:
#                 label = f"ID: {unique_id}, Sim: {similarity:.2f}"
#             else:
#                 # Assign new ID and save face to database
#                 unique_id = len(database) + 1
#                 save_face_to_db(embedding, unique_id)
#                 label = f"New ID: {unique_id}"
#                 database.append((unique_id, embedding))  # Update local database
#
#             # Draw bounding box and label
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#         # Display the frame
#         cv2.imshow('Face Recognition', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# # Start real-time recognition
# if __name__ == "__main__":
#     recognize_faces()
