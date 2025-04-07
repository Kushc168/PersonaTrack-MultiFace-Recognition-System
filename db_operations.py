from pymongo import MongoClient
from datetime import datetime
import numpy as np
import os

# MongoDB connection URI (replace with your actual URI)
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://maheshwarikush20:Kush1176@cluster0.nlafj.mongodb.net/")
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

# from pymongo import MongoClient
# from datetime import datetime
# import numpy as np
# import os
#
# # MongoDB connection URI (replace with your actual URI)
# MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://maheshwarikush20:Kush1176@cluster0.nlafj.mongodb.net/")
# client = MongoClient(MONGO_URI)
# db = client["dt_face"]
# collection = db["recognition"]
#
# # Function to save face embedding to MongoDB (temporary reduced embedding size)
# def save_face_to_db(embedding, unique_id):
#     try:
#         # Reduce the embedding size for faster insertion (e.g., keep the first 5 values)
#         reduced_embedding = embedding[:512]  # Store only the first 5 values
#
#         # Convert numpy array to list for MongoDB storage
#         reduced_embedding_list = reduced_embedding.tolist() if isinstance(reduced_embedding, np.ndarray) else reduced_embedding
#
#         document = {
#             "embedding": reduced_embedding_list,  # Store the reduced embedding
#             "unique_id": unique_id,
#             "created_at": datetime.now()
#         }
#
#         # Insert the document into MongoDB
#         collection.insert_one(document)
#         print(f"Face with ID {unique_id} saved to the database.")
#     except Exception as e:
#         print(f"Error saving face to database: {e}")
#
# # Function to retrieve all embeddings from MongoDB
# def get_all_embeddings():
#     try:
#         # Fetch all documents from the collection
#         faces = collection.find()
#
#         # Return a list of tuples: (unique_id, embedding as numpy array)
#         embeddings = []
#         for face in faces:
#             embedding = np.array(face["embedding"])  # Retrieve reduced embedding (e.g., 5 values)
#             embeddings.append((face["unique_id"], embedding))
#         return embeddings
#     except Exception as e:
#         print(f"Error retrieving embeddings from database: {e}")
#         return []
#
# # Utility function to clear the database (optional)
# def clear_database():
#     try:
#         collection.delete_many({})
#         print("Database cleared successfully.")
#     except Exception as e:
#         print(f"Error clearing database: {e}")
#
# # Utility function to count the number of stored embeddings
# def count_faces_in_db():
#     try:
#         count = collection.count_documents({})
#         print(f"Total faces in database: {count}")
#         return count
#     except Exception as e:
#         print(f"Error counting documents in database: {e}")
#         return 0

# from pymongo import MongoClient
# from datetime import datetime
# import numpy as np
# import os  # For environment variables
#
# # Fetch MongoDB URI from environment variables for security (replace with your actual value if hardcoding)
# MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://maheshwarikush20:Kush1176@cluster0.nlafj.mongodb.net/")
# client = MongoClient(MONGO_URI)
# db = client["facial_recognition"]
# collection = db["face"]
#
#
# # Function to save face embedding to MongoDB
# def save_face_to_db(embedding, unique_id):
#     try:
#         # Ensure the embedding is a numpy array and has the correct shape
#         embedding = np.array(embedding)
#         if embedding.shape[0] != 512:
#             print(f"Warning: Embedding has unexpected shape {embedding.shape}.")
#
#         # Convert numpy array to list for MongoDB storage
#         embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
#
#         document = {
#             "embedding": embedding_list,  # Convert numpy array to list
#             "unique_id": unique_id,
#             "created_at": datetime.now()
#         }
#
#         # Insert the document into MongoDB
#         collection.insert_one(document)
#         print(f"Face with ID {unique_id} saved to the database.")
#     except Exception as e:
#         print(f"Error saving face to database: {e}")
#
#
# # Function to retrieve all embeddings from MongoDB
# def get_all_embeddings():
#     try:
#         # Fetch all documents from the collection
#         faces = collection.find()
#
#         # Return a list of tuples: (unique_id, embedding as numpy array)
#         embeddings = []
#         for face in faces:
#             embedding = np.array(face["embedding"])
#             if embedding.shape[0] != 512:
#                 print(f"Warning: Retrieved embedding has unexpected shape {embedding.shape}.")
#             embeddings.append((face["unique_id"], embedding))
#         return embeddings
#     except Exception as e:
#         print(f"Error retrieving embeddings from database: {e}")
#         return []
#
#
# # Utility function to clear the database (optional)
# def clear_database():
#     try:
#         collection.delete_many({})
#         print("Database cleared successfully.")
#     except Exception as e:
#         print(f"Error clearing database: {e}")
#
#
# # Utility function to count the number of stored embeddings
# def count_faces_in_db():
#     try:
#         count = collection.count_documents({})
#         print(f"Total faces in database: {count}")
#         return count
#     except Exception as e:
#         print(f"Error counting documents in database: {e}")
#         return 0

# from pymongo import MongoClient
# from datetime import datetime
# import numpy as np
#
# # Connect to MongoDB
# MONGO_URI = "mongodb+srv://maheshwarikush20:Kush1176@cluster0.nlafj.mongodb.net/"  # Replace <password> with your actual password
# client = MongoClient(MONGO_URI)
# db = client["face_database"]
# collection = db["faces"]
#
# # Function to save face embedding to MongoDB
# def save_face_to_db(embedding, unique_id):
#     try:
#         # Convert numpy array to a list for MongoDB storage
#         document = {
#             "embedding": embedding.tolist(),  # Convert numpy array to list
#             "unique_id": unique_id,
#             "created_at": datetime.now()
#         }
#         # Insert the document into MongoDB
#         collection.insert_one(document)
#         print(f"Face with ID {unique_id} saved to the database.")
#         count_faces_in_db()
#     except Exception as e:
#         print(f"Error saving face to database: {e}")
#
# # Function to retrieve all embeddings from MongoDB
# def get_all_embeddings():
#     try:
#         # Fetch all documents from the collection
#         faces = collection.find()
#         # Return a list of tuples: (unique_id, embedding as numpy array)
#         return [(face["unique_id"], np.array(face["embedding"])) for face in faces]
#     except Exception as e:
#         print(f"Error retrieving embeddings from database: {e}")
#         return []
#
# # Utility function to clear the database (optional)
# def clear_database():
#     try:
#         collection.delete_many({})
#         print("Database cleared successfully.")
#     except Exception as e:
#         print(f"Error clearing database: {e}")
#
# # Utility function to count the number of stored embeddings
# def count_faces_in_db():
#     try:
#         count = collection.count_documents({})
#         print(f"Total faces in database: {count}")
#         return count
#     except Exception as e:
#         print(f"Error counting documents in database: {e}")
#         return 0

