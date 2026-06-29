import hashlib
from datetime import datetime

class DocumentBuilder:
    @staticmethod
    def build_document_and_metadata(row):
        """
        Converts a raw PostgreSQL row into a formatted clinical string, metadata dict, and MD5 ID.
        Expected row format (Tuple):
        (patient_id, full_name, gender, study_date, priority, image_type, findings_summary, confidence_score)
        """
        patient_id = row[0]
        full_name = row[1]
        gender = row[2] or "Unknown"
        study_date = row[3]
        priority = row[4] or "Normal"
        image_type = row[5]
        findings = row[6] or "No findings recorded"
        confidence = row[7] if row[7] is not None else 0.0

        if isinstance(study_date, datetime):
            study_date = study_date.strftime("%Y-%m-%d")

        confidence_str = f"{confidence:.2f}"
        
        # Unique Hash ID for Incremental Sync (Must match extract_data.py structure for consistency)
        key_str = f"{patient_id}|{study_date}|{image_type}|{findings}|{confidence_str}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()

        # The actual text fed to the LLM/Embedding model
        text = f"""Patient ID: {patient_id}
Patient Name: {full_name}
Gender: {gender}
Study Date: {study_date}
Study Priority: {priority}
Image Type: {image_type}
Findings: {findings}
AI Confidence Score: {confidence:.2f}
"""

        metadata = {
            "patient_id": int(patient_id),
            "gender": gender,
            "priority": priority,
            "study_date": study_date,
            "image_type": image_type,
            "source": "oads_db"
        }

        return text, metadata, key_hash
