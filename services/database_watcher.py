import sqlite3
import psycopg2
import os

class DatabaseWatcher:
    def __init__(self, db_host, db_name, db_user, db_password, db_port, sync_db_path="sync_state.db"):
        self.pg_config = {
            "host": db_host, "database": db_name, "user": db_user, 
            "password": db_password, "port": db_port, "connect_timeout": 10
        }
        self.sync_db_path = sync_db_path
        self._init_sync_db()

    def _init_sync_db(self):
        """Initializes the SQLite database used to track sync state."""
        conn = sqlite3.connect(self.sync_db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sync_hashes (
                key_hash TEXT PRIMARY KEY
            )
        """)
        conn.commit()
        conn.close()

    def _get_known_hashes(self):
        conn = sqlite3.connect(self.sync_db_path)
        cur = conn.cursor()
        cur.execute("SELECT key_hash FROM sync_hashes")
        known = {row[0] for row in cur.fetchall()}
        conn.close()
        return known

    def save_new_hashes(self, new_hashes):
        if not new_hashes:
            return
        conn = sqlite3.connect(self.sync_db_path)
        cur = conn.cursor()
        cur.executemany("INSERT OR IGNORE INTO sync_hashes (key_hash) VALUES (?)", [(h,) for h in new_hashes])
        conn.commit()
        conn.close()

    def remove_deleted_hashes(self, deleted_hashes):
        if not deleted_hashes:
            return
        conn = sqlite3.connect(self.sync_db_path)
        cur = conn.cursor()
        cur.executemany("DELETE FROM sync_hashes WHERE key_hash = ?", [(h,) for h in deleted_hashes])
        conn.commit()
        conn.close()

    def fetch_changes(self, build_document_func):
        """
        Connects to PostgreSQL, fetches the current state, and identifies diffs.
        Returns:
            new_records: list of tuples (document_text, metadata, key_hash)
            deleted_hashes: list of key_hashes that are no longer in PostgreSQL
        """
        known_hashes = self._get_known_hashes()
        current_hashes = set()
        
        new_records = []
        
        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()

            query = """
            SELECT 
                p.patient_id, p.full_name, p.gender, s.study_date, s.priority,
                i.image_type, a.findings_summary, a.confidence_score
            FROM oads.patients p
            JOIN oads.studies s ON p.patient_id = s.patient_id
            JOIN oads.images i ON s.study_id = i.study_id
            JOIN oads.analysis a ON i.image_id = a.image_id
            ORDER BY s.study_date DESC;
            """
            
            cursor.execute(query)
            
            while True:
                rows = cursor.fetchmany(1000)
                if not rows:
                    break
                    
                for row in rows:
                    text, metadata, key_hash = build_document_func(row)
                    current_hashes.add(key_hash)
                    
                    if key_hash not in known_hashes:
                        new_records.append((text, metadata, key_hash))
                        
            # Any hash that we knew about but isn't in current_hashes was deleted
            deleted_hashes = list(known_hashes - current_hashes)
            
            return new_records, deleted_hashes

        except Exception as e:
            print(f"[Watcher] Error reading from PostgreSQL: {e}")
            return [], []
        finally:
            if cursor: cursor.close()
            if conn: conn.close()
