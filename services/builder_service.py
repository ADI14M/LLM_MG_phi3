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
                print("[Builder] Polling database...", flush=True)
                print("[Builder] Checking for changes...", flush=True)
                
                new_records, deleted_hashes = self.watcher.fetch_changes(DocumentBuilder.build_document_and_metadata)
                
                print(f"[Builder] SQL returned {len(new_records)} new/updated records and {len(deleted_hashes)} deleted records.", flush=True)
                
                if not new_records and not deleted_hashes:
                    print("[Builder] No new or modified records found.", flush=True)
                    time.sleep(self.poll_interval)
                    continue
                
                changes_made = False

                if deleted_hashes:
                    print(f"[Builder] Detected {len(deleted_hashes)} deleted/updated records. Removing from FAISS...", flush=True)
                    self.faiss_service.delete_documents(deleted_hashes)
                    self.watcher.remove_deleted_hashes(deleted_hashes)
                    changes_made = True

                if new_records:
                    print(f"[Builder] New records detected: {len(new_records)}", flush=True)
                    print("[Builder] Generating clinical documents...", flush=True)
                    docs = [r[0] for r in new_records]
                    metas = [r[1] for r in new_records]
                    ids = [r[2] for r in new_records]
                    
                    print("[Builder] Creating chunks...", flush=True)
                    print("[Builder] Generating embeddings...", flush=True)
                    self.faiss_service.add_documents(docs, metas, ids)
                    
                    print("[Builder] Updating sync state...", flush=True)
                    self.watcher.save_new_hashes(ids)
                    changes_made = True

                if changes_made:
                    print("[Builder] Updating FAISS...", flush=True)
                    self.faiss_service.save()
                    print("[Builder] Synchronization Complete.", flush=True)

            except Exception as e:
                import traceback
                print(f"[Builder] Critical Error during sync cycle: {e}", flush=True)
                traceback.print_exc()
                
            time.sleep(self.poll_interval)

if __name__ == "__main__":
    service = BuilderService(poll_interval=5)
    service.run()
