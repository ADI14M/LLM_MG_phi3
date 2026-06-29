import time
import sys
import os

# Add parent directory to path so config can be imported if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
from services.database_watcher import DatabaseWatcher
from services.document_builder import DocumentBuilder
from services.embedding_service import EmbeddingService
from services.faiss_service import FaissService

class BuilderService:
    def __init__(self, poll_interval=10):
        self.poll_interval = poll_interval
        print("[Builder] Initializing Builder Service...")
        
        self.watcher = DatabaseWatcher(DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT)
        self.embedding_model = EmbeddingService.get_embeddings_model()
        self.faiss_service = FaissService(index_path="./faiss_db", embedding_model=self.embedding_model)
        
        print("[Builder] Ready. Monitoring PostgreSQL for changes...")

    def run(self):
        while True:
            try:
                new_records, deleted_hashes = self.watcher.fetch_changes(DocumentBuilder.build_document_and_metadata)
                
                changes_made = False

                if deleted_hashes:
                    print(f"[Builder] Detected {len(deleted_hashes)} deleted/updated records. Removing from FAISS...")
                    self.faiss_service.delete_documents(deleted_hashes)
                    self.watcher.remove_deleted_hashes(deleted_hashes)
                    changes_made = True

                if new_records:
                    print(f"[Builder] Detected {len(new_records)} new/updated records. Generating embeddings...")
                    docs = [r[0] for r in new_records]
                    metas = [r[1] for r in new_records]
                    ids = [r[2] for r in new_records]
                    
                    self.faiss_service.add_documents(docs, metas, ids)
                    self.watcher.save_new_hashes(ids)
                    changes_made = True
                    print(f"[Builder] Successfully embedded {len(new_records)} new chunks.")

                if changes_made:
                    print("[Builder] Saving synchronized FAISS index to disk...")
                    self.faiss_service.save()
                    print("[Builder] Synchronization Complete! UI will automatically hot-reload.")

            except Exception as e:
                print(f"[Builder] Critical Error during sync cycle: {e}")
                
            time.sleep(self.poll_interval)

if __name__ == "__main__":
    service = BuilderService(poll_interval=5)
    service.run()
