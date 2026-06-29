from database.db_pool import DatabasePool

class PatientRepository:
    
    @staticmethod
    def get_patient_name_by_id(patient_id):
        conn = DatabasePool.get_connection()
        name = None
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT full_name FROM oads.patients WHERE patient_id = %s", (patient_id,))
                row = cur.fetchone()
                if row:
                    name = row[0]
                cur.close()
            except Exception as e:
                print(f"Error fetching patient by id: {e}")
            finally:
                DatabasePool.release_connection(conn)
        return name

    @staticmethod
    def get_patient_name_by_phrase(phrase):
        conn = DatabasePool.get_connection()
        name = None
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT full_name FROM oads.patients WHERE full_name ILIKE %s ORDER BY patient_id LIMIT 1", (f"%{phrase}%",))
                row = cur.fetchone()
                if row:
                    name = row[0]
                cur.close()
            except Exception as e:
                print(f"Error fetching patient by phrase: {e}")
            finally:
                DatabasePool.release_connection(conn)
        return name

    @staticmethod
    def get_exact_patient(name):
        conn = DatabasePool.get_connection()
        patient = None
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT patient_id, full_name FROM oads.patients WHERE LOWER(full_name) = LOWER(%s)", (name,))
                row = cur.fetchone()
                if row:
                    patient = (row[0], row[1])
                cur.close()
            except Exception as e:
                print(f"Error checking exact patient identity: {e}")
            finally:
                DatabasePool.release_connection(conn)
        return patient

    @staticmethod
    def get_all_patient_names():
        conn = DatabasePool.get_connection()
        names = []
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT full_name FROM oads.patients ORDER BY full_name")
                names = [row[0] for row in cur.fetchall()]
                cur.close()
            except Exception as e:
                print(f"Error fetching all patients: {e}")
            finally:
                DatabasePool.release_connection(conn)
        return names

    @staticmethod
    def get_patient_info_by_name_search(search_name):
        conn = DatabasePool.get_connection()
        info = None
        if conn:
            try:
                cur = conn.cursor()
                query = """
                    SELECT p.patient_id, p.full_name, p.gender, 
                           MAX(s.study_date) as last_visit, COUNT(s.study_id) as total_studies
                    FROM oads.patients p
                    LEFT JOIN oads.studies s ON p.patient_id = s.patient_id
                    WHERE p.full_name ILIKE %s
                    GROUP BY p.patient_id, p.full_name, p.gender
                    LIMIT 1
                """
                cur.execute(query, (f"%{search_name}%",))
                info = cur.fetchone()
                cur.close()
            except Exception as e:
                print(f"Error fetching patient info: {e}")
            finally:
                DatabasePool.release_connection(conn)
        return info

    @staticmethod
    def get_patient_clinical_records(patient_id):
        conn = DatabasePool.get_connection()
        records = []
        if conn:
            try:
                cur = conn.cursor()
                sql_query = """
                    SELECT
                        p.patient_id, p.full_name, p.gender,
                        s.study_date, s.priority,
                        COALESCE(i.image_type, 'Unknown') AS image_type,
                        COALESCE(a.findings_summary, 'No clinical findings recorded') AS findings_summary,
                        a.confidence_score
                    FROM oads.patients p
                    JOIN oads.studies s ON p.patient_id = s.patient_id
                    JOIN oads.images i ON s.study_id = i.study_id
                    LEFT JOIN oads.analysis a ON i.image_id = a.image_id
                    WHERE p.patient_id = %s
                    ORDER BY s.study_date DESC
                """
                cur.execute(sql_query, (patient_id,))
                records = cur.fetchall()
                cur.close()
            except Exception as e:
                print(f"Error fetching clinical records: {e}")
            finally:
                DatabasePool.release_connection(conn)
        return records

    @staticmethod
    def get_patient_gender(patient_id):
        conn = DatabasePool.get_connection()
        gender = "Unknown"
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT gender FROM oads.patients WHERE patient_id = %s", (patient_id,))
                row = cur.fetchone()
                if row and row[0]:
                    gender = row[0]
                cur.close()
            except Exception as e:
                print(f"Error fetching gender: {e}")
            finally:
                DatabasePool.release_connection(conn)
        return gender

    @staticmethod
    def get_patients_by_modality(modality):
        conn = DatabasePool.get_connection()
        records = []
        if conn:
            try:
                cur = conn.cursor()
                sql_query = """
                    SELECT p.patient_id, p.full_name, string_agg(DISTINCT i.image_type, ', ') as scan_types, string_agg(DISTINCT a.findings_summary, '; ') as findings
                    FROM oads.patients p
                    JOIN oads.studies s ON p.patient_id = s.patient_id
                    JOIN oads.images i ON s.study_id = i.study_id
                    LEFT JOIN oads.analysis a ON i.image_id = a.image_id
                    WHERE i.image_type ILIKE %s
                    GROUP BY p.patient_id, p.full_name
                    ORDER BY p.patient_id;
                """
                sql_modality = "xray" if modality in ["xray", "x-ray"] else modality
                cur.execute(sql_query, (sql_modality,))
                records = cur.fetchall()
                cur.close()
            except Exception as e:
                print(f"Error fetching records by modality: {e}")
            finally:
                DatabasePool.release_connection(conn)
        return records

    @staticmethod
    def get_db_stats():
        """Returns stats used strictly for the Developer Debug panel."""
        stats = {
            "total_patients": "Unknown",
            "total_labevents": "Unknown",
            "imaging_summary": ""
        }
        
        # Connect to ehr_db
        try:
            import psycopg2
            from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
            conn_ehr = psycopg2.connect(host=DB_HOST, database='ehr_db', user=DB_USER, password=DB_PASSWORD, port=DB_PORT, connect_timeout=5)
            cur_ehr = conn_ehr.cursor()
            cur_ehr.execute("SELECT COUNT(*) FROM ehr.patients")
            stats["total_patients"] = cur_ehr.fetchone()[0]
            cur_ehr.execute("SELECT COUNT(*) FROM ehr.labevents")
            stats["total_labevents"] = cur_ehr.fetchone()[0]
            cur_ehr.close()
            conn_ehr.close()
        except Exception:
            pass

        # Connect to oads_db via pool
        conn = DatabasePool.get_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT DISTINCT p.patient_id, p.full_name, i.image_type
                    FROM oads.patients p
                    JOIN oads.studies s ON p.patient_id = s.patient_id
                    JOIN oads.images i ON s.study_id = i.study_id
                    ORDER BY p.patient_id
                """)
                rows = cur.fetchall()
                cur.close()
                
                patient_map = {}
                for pid, name, img_type in rows:
                    if name not in patient_map:
                        patient_map[name] = []
                    if img_type:
                        patient_map[name].append(img_type.upper())
                
                summary_lines = ["[ PATIENT IMAGING MAPPING ]"]
                for name, modalities in patient_map.items():
                    summary_lines.append(f"- {name}: {', '.join(modalities)}")
                stats["imaging_summary"] = "\n".join(summary_lines)
            except Exception:
                pass
            finally:
                DatabasePool.release_connection(conn)
                
        return stats
