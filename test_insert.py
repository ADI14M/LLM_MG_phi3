import psycopg2
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

try:
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
    cur = conn.cursor()

    cur.execute("SELECT MAX(patient_id) FROM oads.patients;")
    max_id = cur.fetchone()[0] or 5000
    new_id = max_id + 1

    cur.execute(f"INSERT INTO oads.patients (patient_id, full_name, gender) VALUES ({new_id}, 'Test Patient Automation', 'Male');")

    cur.execute("SELECT MAX(study_id) FROM oads.studies;")
    max_sid = cur.fetchone()[0] or 1000
    new_sid = max_sid + 1

    cur.execute(f"INSERT INTO oads.studies (study_id, patient_id, study_date, priority) VALUES ({new_sid}, {new_id}, '2026-06-29', 'High');")
    
    conn.commit()
    print(f"Successfully inserted mock patient_id {new_id} and study_id {new_sid}.")

except Exception as e:
    print(f"Error: {e}")
finally:
    if cur: cur.close()
    if conn: conn.close()
