import os
import uuid
from langchain_community.vectorstores import FAISS

class FaissService:
    def __init__(self, index_path, embedding_model):
        self.index_path = index_path
        self.embedding_model = embedding_model
        
        # Load existing index or create an empty one
        if os.path.exists(self.index_path) and os.path.exists(os.path.join(self.index_path, "index.faiss")):
            self.vectorstore = FAISS.load_local(
                self.index_path, 
                self.embedding_model, 
                allow_dangerous_deserialization=True
            )
        else:
            # Need at least one fake vector to initialize an empty FAISS index in Langchain
            print("[FAISS] No existing DB found. Creating a new one...")
            self.vectorstore = FAISS.from_texts(
                texts=["initial_placeholder"], 
                embedding=self.embedding_model,
                ids=["placeholder_id"]
            )
            # Remove the placeholder immediately so it's perfectly clean
            self.vectorstore.delete(["placeholder_id"])

    def add_documents(self, documents, metadatas, ids):
        """Adds new documents to the FAISS index."""
        if documents:
            self.vectorstore.add_texts(texts=documents, metadatas=metadatas, ids=ids)

    def delete_documents(self, ids):
        """Deletes vectors from the FAISS index by their unique hash ID."""
        if ids:
            try:
                self.vectorstore.delete(ids)
            except Exception as e:
                # Sometimes a delete can fail if the ID was somehow wiped but still in sync_state
                print(f"[FAISS] Warning during deletion: {e}")

    def save(self):
        """Saves the FAISS index back to disk."""
        self.vectorstore.save_local(self.index_path)
