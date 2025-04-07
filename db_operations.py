from pymongo import MongoClient
from datetime import datetime
import numpy as np
import os

# MongoDB connection URI (replace with your actual URI)
MONGO_URI = os.getenv("MONGO_URI", "Actual Url"/")
client = MongoClient(MONGO_URI)
db = client["dt_face"]
collection = db["recognition"]

# Function to save face embedding to MongoDB (512-dimensions enforced)
def save_face_to_db(embedding, unique_id):
    try:
        # Ensure the embedding is 512-dimensional
        if len(embedding) != 512:
            raise ValueError(f"Embedding size mismatch: expected 512, got {len(embedding)}")

        # Convert numpy array to list for MongoDB storage
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

        # Prepare the document
        document = {
            "embedding": embedding_list,
            "unique_id": unique_id,
            "created_at": datetime.now()
        }

        # Insert the document into MongoDB
        collection.insert_one(document)
        print(f"Face with ID {unique_id} saved to the database.")
    except Exception as e:
        print(f"Error saving face to database: {e}")

# Function to retrieve all embeddings from MongoDB
def get_all_embeddings():
    try:
        # Fetch all documents from the collection
        faces = collection.find()

        embeddings = []
        for face in faces:
            embedding = np.array(face["embedding"])  # Retrieve embedding as a numpy array
            print(f"Retrieved embedding for ID {face['unique_id']}: {embedding.shape}")  # Debug
            embeddings.append((face["unique_id"], embedding))
        return embeddings
    except Exception as e:
        print(f"Error retrieving embeddings from database: {e}")
        return []

# Utility function to clear the database
def clear_database():
    try:
        collection.delete_many({})  # Delete all documents
        print("Database cleared successfully.")
    except Exception as e:
        print(f"Error clearing database: {e}")

# Utility function to count the number of stored embeddings
def count_faces_in_db():
    try:
        count = collection.count_documents({})
        print(f"Total faces in database: {count}")
        return count
    except Exception as e:
        print(f"Error counting documents in database: {e}")
        return 0

# Utility function to validate all embeddings in the database
def validate_embeddings():
    try:
        faces = collection.find()
        for face in faces:
            embedding = face["embedding"]
            if len(embedding) != 512:
                print(f"Invalid embedding size for unique_id {face['unique_id']}: {len(embedding)}")
        print("Validation complete.")
    except Exception as e:
        print(f"Error validating embeddings: {e}")

# Cleanup function to remove invalid embeddings
def cleanup_invalid_embeddings():
    try:
        # Delete all documents where the embedding size is not 512
        result = collection.delete_many({"$expr": {"$ne": [{"$size": "$embedding"}, 512]}})
        print(f"Cleaned up {result.deleted_count} invalid embeddings from the database.")
    except Exception as e:
        print(f"Error during cleanup: {e}")



