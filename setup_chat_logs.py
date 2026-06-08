import psycopg2
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

def setup_table():
    try:
        print(f"Connecting to {DB_NAME} at {DB_HOST}:{DB_PORT} as {DB_USER}...")
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cur = conn.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS oads.chat_logs (
            id SERIAL PRIMARY KEY,
            session_mode VARCHAR(50) NOT NULL,
            patient_id INTEGER,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        print("Executing CREATE TABLE query...")
        cur.execute(create_table_query)
        conn.commit()
        
        print("Table oads.chat_logs successfully created/verified.")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error setting up database: {e}")

if __name__ == "__main__":
    setup_table()
