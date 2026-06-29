import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import time
import re
import psycopg2

def extract_and_verify_patient(prompt):
    """
    Parses the prompt to detect any patient name or ID.
    If a patient name or ID is present, performs dynamic lookup & exact database verification.
    Returns:
        is_patient_query (bool): True if the query targets a patient.
        patient_id (int or None): The verified patient ID.
        patient_name (str or None): The verified patient name.
        lookup_status (str): 'verified' or 'not_found' or 'not_patient_query'.
        detected_name (str or None): The exact name string that was resolved.
    """
    import re
    from database.patient_repository import PatientRepository

    query_lower = prompt.lower()
    
    # 1. Check for ID (e.g. "Patient 3" or just "3")
    match_id = re.search(r'\bpatient\s+(?:id\s+)?(\d+)\b', query_lower)
    if not match_id:
        match_id = re.search(r'\b(?:patient\s+)?(?:id\s*)?#?(\d+)\b', query_lower)
        
    patient_id_num = None
    if match_id:
        patient_id_num = int(match_id.group(1))

    # 2. Stopword filtering to extract name candidate words
    words = re.findall(r'\b[a-zA-Z]+\b', prompt)
    stopwords = {
        # Query intent words
        "patient", "patients", "summarize", "history", "insights", "report",
        "findings", "results", "detail", "details", "visit", "visits", "multiple",
        "retrieve", "dates", "recorded", "identify", "detect", "generate",
        "structured", "draft", "formal", "impression", "impressions", "recommendations",
        "recommendation", "suggest", "steps", "provide", "list", "show", "give", "get",
        "me", "us",
        # Common prepositions / articles / verbs
        "of", "the", "a", "an", "is", "are", "what", "recent", "who", "has",
        "having", "which", "had", "for", "with", "on", "based", "their", "all",
        "there", "any", "does", "have", "made", "next", "do", "in", "and", "or",
        # Clinical / imaging terms (modalities, adjectives)
        "scan", "scans", "scanning", "abnormal", "imaging", "priority", "levels",
        "studies", "study", "result", "flagged", "abnormality", "minor", "issue",
        "issues", "clinical", "ct", "mri", "xray", "mr", "ultrasound", "x",
        "radiology", "radiological"
    }
    candidate_words = [w for w in words if w.lower() not in stopwords]
    candidate_words.sort(key=len, reverse=True)

    # If neither candidate name words nor ID are found, it's not a patient-specific query
    if not candidate_words and not patient_id_num:
        return False, None, None, "not_patient_query", None

    candidate_name = None

    try:
        # A. Resolve using patient ID if explicitly specified
        if patient_id_num:
            candidate_name = PatientRepository.get_patient_name_by_id(patient_id_num)

        # B. If not resolved yet, resolve using name phrases dynamically (e.g. "Priya Reddy")
        if not candidate_name and len(candidate_words) >= 2:
            for i in range(len(candidate_words) - 1):
                phrase = f"{candidate_words[i]} {candidate_words[i+1]}"
                candidate_name = PatientRepository.get_patient_name_by_phrase(phrase)
                if candidate_name:
                    break

        # C. If not resolved yet, resolve using single names (e.g. "rahul")
        if not candidate_name:
            for word in candidate_words:
                if len(word) < 3:
                    continue
                candidate_name = PatientRepository.get_patient_name_by_phrase(word)
                if candidate_name:
                    break

        if not candidate_name:
            # If a patient name was mentioned but could not be mapped to any record, return not_found
            return True, None, None, "not_found", " ".join(candidate_words) if candidate_words else f"Patient ID {patient_id_num}"

        # D. Execute exact verification query
        patient = PatientRepository.get_exact_patient(candidate_name)

        if patient:
            return True, patient[0], patient[1], "verified", candidate_name
        else:
            return True, None, None, "not_found", candidate_name

    except Exception as e:
        print(f"Error checking patient identity: {e}")
        return True, None, None, "not_found", " ".join(candidate_words) if candidate_words else None

def parse_chunk_text(text, metadata=None):
    pid = None
    study_date = ""
    image_type = ""
    findings = ""
    priority = ""
    confidence = ""
    
    # Parse lines
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Patient ID:"):
            try:
                pid = int(line.split(":", 1)[1].strip())
            except:
                pass
        elif line.startswith("Study Date:"):
            study_date = line.split(":", 1)[1].strip()
        elif line.startswith("Image Type:"):
            image_type = line.split(":", 1)[1].strip()
        elif line.startswith("Findings:"):
            findings = line.split(":", 1)[1].strip()
        elif line.startswith("Study Priority:"):
            priority = line.split(":", 1)[1].strip()
        elif line.startswith("AI Confidence Score:"):
            confidence = line.split(":", 1)[1].strip()
            
    # Fallback to metadata
    if pid is None and metadata and "patient_id" in metadata:
        pid = int(metadata["patient_id"])
    if not study_date and metadata and "study_date" in metadata:
        study_date = str(metadata["study_date"]).strip()
    if not image_type and metadata and "image_type" in metadata:
        image_type = str(metadata["image_type"]).strip()
    if not priority and metadata and "priority" in metadata:
        priority = str(metadata["priority"]).strip()
        
    return pid, study_date, image_type, findings, priority, confidence

def aggregate_chunks_into_summary(patient_name, patient_id, gender, unique_parsed_chunks):
    summary_parts = [
        f"Patient ID: {patient_id}",
        f"Patient Name: {patient_name}",
        f"Gender: {gender}",
        "Imaging Studies & Clinical History:"
    ]
    for chunk in unique_parsed_chunks:
        pid, study_date, image_type, findings, priority, confidence = chunk
        confidence_str = f" (Confidence: {confidence})" if confidence else ""
        priority_str = f" ({priority} priority)" if priority else ""
        summary_parts.append(f"- Date: {study_date} | Type: {image_type.upper()}{priority_str} | Findings: {findings}{confidence_str}")
        
    return "\n".join(summary_parts)

def clean_llm_output(text):
    """
    Cleans the LLM output to prevent exposing prompt headers and structures.
    Only the final clinical answer / summary is shown to the user.
    """
    marker = "final clinical answer:"
    lower_text = text.lower()
    if marker in lower_text:
        idx = lower_text.rfind(marker)
        return text[idx + len(marker):].strip()
        
    # Fallback: strip standard prompt sections from output
    headers = [
        "[ DATABASE STATISTICS ]",
        "[ PATIENT IMAGING MAPPING ]",
        "[ PATIENT LAB EVENT CONTEXT ]",
        "[ STRICT RULES ]"
    ]
    lines = text.split("\n")
    cleaned_lines = []
    skip_mode = False
    for line in lines:
        stripped = line.strip()
        if any(h in stripped for h in headers):
            skip_mode = True
            continue
        if stripped.startswith("Question:") or stripped.startswith("Final Clinical Answer:"):
            skip_mode = False
            continue
        if skip_mode:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()




# ====================== Page Config ======================
st.set_page_config(
    page_title="Visukhi Medical Assistant",
    page_icon="🩺",
    layout="centered"
)



st.html(
    """
    <script>
    const doc = window.parent.document;
    const oldBtn = doc.getElementById('chatgpt-scroll-btn');
    if (oldBtn) oldBtn.remove();

    let btn = doc.createElement('div');
    btn.id = 'chatgpt-scroll-btn';
    btn.innerHTML = '&#8595;'; 
    btn.style.cssText = "position:fixed; bottom:100px; left:50%; transform:translateX(-50%); width:36px; height:36px; border-radius:50%; background-color:#ffffff; color:#333333; text-align:center; line-height:34px; font-size:20px; cursor:pointer; z-index:999999; box-shadow:0px 2px 8px rgba(0,0,0,0.15); border:1px solid #e5e5e5; display:none;";
    doc.body.appendChild(btn);

    btn.addEventListener('click', () => {
        const msgs = doc.querySelectorAll('[data-testid=\"stChatMessage\"]');
        if (msgs.length > 0) {
            msgs[msgs.length - 1].scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
    });

    setInterval(() => {
        const msgs = doc.querySelectorAll('[data-testid=\"stChatMessage\"]');
        if(msgs.length > 0) {
            const lastMsg = msgs[msgs.length - 1];
            const rect = lastMsg.getBoundingClientRect();
            if (rect.bottom > window.parent.innerHeight + 50) {
                btn.style.display = 'block';
            } else {
                btn.style.display = 'none';
            }
        }
    }, 300);
    </script>
    """
)

st.markdown(
    """
    <style>
    /* Make the title sticky */
    div[data-testid="stHeadingWithActionElements"] {
        position: sticky;
        top: 2.875rem;
        z-index: 999;
        background-color: #c6cbd3;
        padding-top: 1rem;
        border-bottom: 1px solid #a0a6b1;
    }
    /* Stop chat text from overlapping behind the sticky input bar */
    div.block-container {
        padding-bottom: 150px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Visukhi Medical Chatbot")

if "chat_mode" not in st.session_state:
    st.session_state["chat_mode"] = "Database Mode"
if "last_selected_patient" not in st.session_state:
    st.session_state["last_selected_patient"] = ""
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ====================== Patient Search Sidebar ======================
@st.cache_data(ttl=3600)
def get_all_patient_names():
    from database.patient_repository import PatientRepository
    return PatientRepository.get_all_patient_names()



with st.sidebar:
    st.header("🔍 Patient Search")
    
    patient_names = get_all_patient_names()
    options = [""] + patient_names
    
    search_name = st.selectbox(
        "Search and Select a Patient:",
        options=options,
        index=0,
        help="Start typing a name to see suggestions"
    )
    
    # State change detection (Rules 1, 4, 5)
    if search_name != st.session_state["last_selected_patient"]:
        st.session_state["last_selected_patient"] = search_name
        if search_name:
            st.session_state["chat_mode"] = "Patient Mode"
            # History will be loaded automatically further down
        else:
            st.session_state["selected_patient_id"] = None
            st.session_state["selected_patient_name"] = None

    if not search_name:
        st.session_state["selected_patient_id"] = None
        st.session_state["selected_patient_name"] = None

    if search_name:
        with st.spinner(f"Searching for {search_name}..."):
            from database.patient_repository import PatientRepository
            patient_info = PatientRepository.get_patient_info_by_name_search(search_name)
            
            if patient_info:
                st.session_state["selected_patient_id"] = patient_info[0]
                st.session_state["selected_patient_name"] = patient_info[1]
                st.subheader(f"Patient: {patient_info[1]}")
                st.write(f"**ID:** {patient_info[0]} | **Gender:** {patient_info[2].capitalize() if patient_info[2] else 'Unknown'}")
                
                # Fetch recent studies summary for sidebar
                records = PatientRepository.get_patient_clinical_records(patient_info[0])
                if records:
                    st.markdown("### Recent Studies & Findings")
                    studies_text = ""
                    for r in records[:10]: # Limit sidebar to 10
                        if r[3]: # study_date
                            conf_str = f"(Conf: {r[7]:.2f})" if r[7] is not None else ""
                            findings = r[6] if r[6] else "No findings recorded"
                            img_type = r[5].upper() if r[5] else "Unknown"
                                date_str = r[3].strftime('%Y-%m-%d')
                                st.markdown(f"""
**Date:** {date_str}
- **Type:** {img_type} ({r[4]} priority)
- **Findings:** {findings} {conf_str}
                                """)
                                st.divider()
                                studies_text += f"- Date: {date_str}, Type: {img_type}, Priority: {r[4]}, Findings: {findings}\n"
                        
                        st.markdown("---")
                        if st.button("📄 Generate Radiology Report (PDF)"):
                            with st.spinner("Generating AI Radiology Report..."):
                                # Collect study rows for PDF: (study_date, priority, image_type, findings, confidence)
                                pdf_study_rows = []
                                for r in rows:
                                    if r[3]:  # has study_date
                                        pdf_study_rows.append((
                                            r[3],                                     # study_date
                                            r[4],                                     # priority
                                            r[5] if r[5] else "Unknown",              # image_type
                                            r[6] if r[6] else "No findings recorded", # findings
                                            r[7],                                     # confidence
                                        ))

                                # Generate LLM impression & recommendations
                                impression_text = ""
                                recommendations_text = ""
                                try:
                                    # Always use clinical model for radiology report generation
                                    report_llm = ChatOllama(model="medgemma1.5:4b", temperature=0.1)

                                    impression_prompt = f"""Based on these radiology findings for patient {patient_info[1]}, write a brief clinical impression in 2-3 sentences. Only output the impression, nothing else.

Findings:
{studies_text}"""
                                    impression_text = report_llm.invoke(impression_prompt).content.strip()

                                    recs_prompt = f"""Based on these radiology findings for patient {patient_info[1]}, write 2-3 specific clinical recommendations. Only output the recommendations, nothing else.

Findings:
{studies_text}"""
                                    recommendations_text = report_llm.invoke(recs_prompt).content.strip()
                                except Exception:
                                    pass  # fallback to auto-generated text in the PDF

                                from generate_report import generate_pdf_report

                                pdf_bytes = generate_pdf_report(
                                    patient_id=patient_info[0],
                                    patient_name=patient_info[1],
                                    gender=patient_info[2] or "Unknown",
                                    study_rows=pdf_study_rows,
                                    ai_impression=impression_text,
                                    ai_recommendations=recommendations_text,
                                )
                                st.session_state[f"pdf_report_{search_name}"] = pdf_bytes
                                st.session_state[f"pdf_report_name_{search_name}"] = patient_info[1]
                                st.success("✅ PDF report generated successfully!")

                        if f"pdf_report_{search_name}" in st.session_state:
                            pdf_data = st.session_state[f"pdf_report_{search_name}"]
                            report_patient_name = st.session_state.get(f"pdf_report_name_{search_name}", "Patient")
                            safe_name = report_patient_name.replace(" ", "_")

                            st.download_button(
                                label="⬇️ Download Radiology Report (PDF)",
                                data=pdf_data,
                                file_name=f"Radiology_Report_{safe_name}.pdf",
                                mime="application/pdf",
                            )
                    else:
                        st.info("No studies found for this patient.")
                else:
                    st.warning("Patient not found in database.")
            except Exception as e:
                st.error(f"Error fetching patient data: {e}")


# ====================== Settings ======================
TEMPERATURE = 0.1

def get_llm(question):
    """Routes the question to the appropriate model based on clinical intent."""
    q_lower = question.lower()
    clinical_keywords = [
        "summarize", "radiology", "ct", "mri", "abnormalities", "clinical", 
        "findings", "diagnosis", "recommendations", "scan", "xray", "ultrasound",
        "impression", "report", "patient", "history", "health", "disease", "treatment"
    ]
    if any(k in q_lower for k in clinical_keywords):
        return "medgemma1.5:4b"
    return "phi3"


TOP_K = 10
FETCH_K = 40
MAX_CONTEXT_CHARS = 3000


import os

@st.cache_resource(show_spinner=False)
def load_vectorstore(mod_time):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    try:
        return FAISS.load_local("./faiss_db", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load FAISS DB. Error: {e}")
        return None

faiss_path = "./faiss_db/index.faiss"
mod_time = os.path.getmtime(faiss_path) if os.path.exists(faiss_path) else 0

with st.spinner("Loading vector database..."):
    st.session_state.vectorstore = load_vectorstore(mod_time)

mode_index = 0 if st.session_state.get("chat_mode") == "Patient Mode" else 1
new_mode = st.radio(
    "Chat Mode",
    ["Patient Mode", "Database Mode"],
    horizontal=True,
    index=mode_index
)
st.session_state["chat_mode"] = new_mode

selected_patient_id = st.session_state.get("selected_patient_id")
selected_patient_name = st.session_state.get("selected_patient_name")
selected_patient_name = st.session_state.get("selected_patient_name")

# Detect mode/patient changes and reload history
current_mode = st.session_state["chat_mode"]
last_loaded_mode = st.session_state.get("last_loaded_mode")
last_loaded_patient = st.session_state.get("last_loaded_patient")

if ("messages" not in st.session_state 
    or last_loaded_mode != current_mode 
    or last_loaded_patient != selected_patient_id):
    
    st.session_state.messages = []
    st.session_state["last_loaded_mode"] = current_mode
    st.session_state["last_loaded_patient"] = selected_patient_id

if st.session_state["chat_mode"] == "Patient Mode":
    if selected_patient_id:
        st.info(f"**Mode:** Patient Mode\n\n**Patient:** {selected_patient_name}\n\n**ID:** {selected_patient_id}", icon="🩺")
    else:
        st.warning("Please select a patient first.")
else:
    st.info("**Mode:** Database Mode\n\n**Searching:** Entire Database", icon="🌍")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Ask a medical question..."):

    # Safety filter
    if any(w in prompt.lower() for w in ["suicide", "kill", "overdose"]):
        st.warning("⚠️ Cannot handle this query.")
        st.stop()

    # Show original user message first so the UI feels responsive
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ====================== AUTOCORRECT TYPOS ======================
    original_prompt = prompt
    with st.spinner("Checking for typos..."):
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.prompts import ChatPromptTemplate
            
            correction_llm = ChatOllama(model="phi3", temperature=0.0)
            correction_prompt = ChatPromptTemplate.from_template(
                "You are an advanced medical spelling correction system. Your ONLY job is to fix typos and grammatical errors in the following user question. "
                "You must preserve all medical terminology. If there are no typos, output the exact original text. "
                "DO NOT answer the question. DO NOT add conversational filler. "
                "ONLY output the corrected question.\n\nUser Question: {question}"
            )
            corrected_prompt = correction_llm.invoke(correction_prompt.format(question=prompt)).content.strip()
            
            # Clean up potential quotes added by the LLM
            if corrected_prompt.startswith('"') and corrected_prompt.endswith('"'):
                corrected_prompt = corrected_prompt[1:-1]
                
            # Validation: Ensure it didn't hallucinate a massive response
            if corrected_prompt and len(corrected_prompt) < len(prompt) * 2.5:
                if corrected_prompt.lower() != prompt.lower():
                    st.info(f"💡 *Auto-corrected:* {corrected_prompt}")
                    prompt = corrected_prompt
        except Exception as e:
            print(f"Autocorrect error: {e}")

    # Assistant
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        if st.session_state.get("chat_mode") == "Patient Mode" and not st.session_state.get("selected_patient_id"):
            placeholder.error("Please select a patient first from the sidebar.")
            st.stop()

        start_time = time.time()

        # Initialize debug logging variables
        debug_sql_query = "N/A"
        debug_sql_records = 0
        debug_faiss_chunks = 0

        # ====================== PATIENT DETECTION & VERIFICATION ======================
        if st.session_state.get("chat_mode") == "Patient Mode" and st.session_state.get("selected_patient_id"):
            verified_patient_id = st.session_state["selected_patient_id"]
            verified_patient_name = st.session_state["selected_patient_name"]
            detected_name = verified_patient_name
            lookup_status = "verified"
            is_patient_query = True
        else:
            is_patient_query, verified_patient_id, verified_patient_name, lookup_status, detected_name = extract_and_verify_patient(prompt)

        # Safety Check / Validation check: If patient is requested but does not exist in DB
        # Requirement 4: If no patient exists, Return: "Patient not found."
        if is_patient_query and lookup_status == "not_found":
            full_response = "Patient not found."
            placeholder.markdown(full_response)
            
            # Print minimal logs for validation failure
            print("\n=== RAG PIPELINE DEBUG LOGS ===")
            print(f"Detected patient name: {detected_name}")
            print(f"SQL query executed: SELECT patient_id, full_name FROM oads.patients WHERE LOWER(full_name)=LOWER(%s)")
            print(f"Retrieved patient_id: None")
            print(f"Retrieved chunk count: 0")
            print(f"Unique patient_ids found: []")
            print("===============================\n")
            
            # Render in Streamlit UI
            with st.expander("Developer Debug Logs", expanded=False):
                st.markdown(f"**Detected Patient Name:** `{detected_name}`")
                st.markdown(f"**Verified Patient ID:** `None`")
                st.markdown(f"**SQL Query Executed:** `SELECT patient_id, full_name FROM oads.patients WHERE LOWER(full_name)=LOWER(%s)`")
                st.markdown(f"**Status:** `Validation Failed - Patient Not Found`")
        else:
            # ====================== RETRIEVAL & COMBINATION ======================
            context = ""
            combined_chunks = []
            
            # Keep track of unique pids for logs
            retrieved_pids = set()
            
            # If a patient is found, restrict all retrieval to that patient_id (SQL first, FAISS second)
            if verified_patient_id is not None:
                # 1. SQL Filtering First
                patient_gender = PatientRepository.get_patient_gender(verified_patient_id)
                sql_rows = PatientRepository.get_patient_clinical_records(verified_patient_id)

                debug_sql_records = len(sql_rows)
                debug_sql_query = "SELECT ... FROM oads.patients ... WHERE patient_id = %s"

                # ── STEP 4/5 guard: if no studies/images exist, report cleanly ──
                if not sql_rows:
                    full_response = f"No clinical records found for patient {verified_patient_name} (ID: {verified_patient_id})."
                    placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.stop()

                # Check if ALL findings are the 'no findings' placeholder
                real_findings = [r[6] for r in sql_rows if r[6] and r[6] != 'No clinical findings recorded']
                if not real_findings:
                    full_response = f"No clinical findings available for patient {verified_patient_name}."
                    placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.stop()

                # Format SQL rows as chunks
                for r in sql_rows:
                    pid, fname, gender, study_date, priority, img_type, findings, confidence = r
                    if hasattr(study_date, "strftime"):
                        study_date = study_date.strftime("%Y-%m-%d")
                    confidence_str = f"{confidence:.2f}" if confidence is not None else "N/A"
                    findings_str = findings if findings else "No clinical findings recorded"

                    chunk_text = (
                        f"Patient ID: {pid}\nPatient Name: {fname}\nGender: {gender}\n"
                        f"Study Date: {study_date}\nStudy Priority: {priority}\n"
                        f"Image Type: {img_type}\nFindings: {findings_str}\n"
                        f"AI Confidence Score: {confidence_str}\n"
                    )
                    meta = {
                        "patient_id": pid,
                        "study_date": study_date,
                        "image_type": img_type,
                        "findings": findings_str,
                        "priority": priority,
                        "confidence": confidence_str
                    }
                    combined_chunks.append((chunk_text, meta))
                    retrieved_pids.add(pid)
                    
                # 2. FAISS Retrieval Second
                try:
                    search_kwargs = {"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": 0.5, "filter": {"patient_id": verified_patient_id}}
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs=search_kwargs
                    )
                    docs = retriever.invoke(prompt)
                    debug_faiss_chunks = len(docs)
                    for doc in docs:
                        combined_chunks.append((doc.page_content, doc.metadata))
                        if "patient_id" in doc.metadata:
                            retrieved_pids.add(doc.metadata["patient_id"])
                except Exception as e:
                    debug_faiss_chunks = 0
            
            else:
                # General query or aggregate modality query
                # Detect if aggregate modality query
                query_lower = prompt.lower()
                matched_modality = None
                for m in ["ct", "mri", "xray", "x-ray", "ultrasound"]:
                    if m in query_lower:
                        matched_modality = m
                        break
                        
                is_list_query = any(w in query_lower for w in ["patients with", "who has", "patients having", "list of patients", "which patients", "who had"])
                
                if is_list_query and matched_modality:
                    # Hybrid Path: Query database directly for all patients with this scan type
                    rows = PatientRepository.get_patients_by_modality(matched_modality)
                    debug_sql_query = "SELECT ... FROM oads.patients ... WHERE i.image_type ILIKE %s"
                    
                    debug_sql_records = len(rows)
                    debug_faiss_chunks = 0
                    
                    import math
                    total_count = len(rows)
                    page_size = 10
                    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
                    page = 1  # Default to page 1
                    
                    context_parts = [
                        f"Total Patients with {matched_modality.upper()} scans found in database: {total_count}.",
                        f"Showing Page {page} of {total_pages}:"
                    ]
                    
                    start_idx = (page - 1) * page_size
                    end_idx = start_idx + page_size
                    page_rows = rows[start_idx:end_idx]
                    
                    for idx, r in enumerate(page_rows):
                        pid, name, scan_types, findings = r
                        findings_str = findings if findings else "No findings recorded"
                        context_parts.append(f"{start_idx + idx + 1}. Patient ID: {pid} | Name: {name} | Modalities: {scan_types.upper()} | Findings: {findings_str}")
                        retrieved_pids.add(pid)
                        
                    context = "\n".join(context_parts)
                else:
                    # Standard Path: MMR Vector Search for general questions
                    search_kwargs = {"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": 0.5}
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs=search_kwargs
                    )
                    docs = retriever.invoke(prompt)
                    
                    debug_faiss_chunks = len(docs)
                    debug_sql_query = "N/A (Standard vector search executed)"
                    debug_sql_records = 0
                    
                    for doc in docs:
                        combined_chunks.append((doc.page_content, doc.metadata))
                        if "patient_id" in doc.metadata:
                            retrieved_pids.add(doc.metadata["patient_id"])

            # ====================== VERIFICATION & DEDUPLICATION ======================
            verified_chunks = []
            seen_keys = set()
            discarded_count = 0
            mismatched_pids = set()
            
            # Before sending context to the LLM: Verify and deduplicate chunks
            for text, meta in combined_chunks:
                pid, study_date, image_type, findings, priority, confidence = parse_chunk_text(text, meta)
                
                # 5. Verify that all retrieved chunks belong to the same verified_patient_id (if patient query)
                if verified_patient_id is not None and pid != verified_patient_id:
                    discarded_count += 1
                    if pid is not None:
                        mismatched_pids.add(pid)
                    continue
                
                # 6. Deduplicate by patient_id + study_date + image_type + findings
                key = (pid, str(study_date).strip().lower(), str(image_type).strip().lower(), str(findings).strip().lower())
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                
                verified_chunks.append((pid, study_date, image_type, findings, priority, confidence))
                
            # ====================== AGGREGATION & CONTEXT BUILDING ======================
            if verified_patient_id is not None:
                # 7. If multiple chunks belong to the same patient, aggregate them into a single summary
                if verified_chunks:
                    context = aggregate_chunks_into_summary(verified_patient_name, verified_patient_id, patient_gender, verified_chunks)
                else:
                    context = "Not found in database"
            elif not (is_list_query and matched_modality):
                context_parts = []
                current_len = 0
                for chunk in verified_chunks:
                    pid, study_date, image_type, findings, priority, confidence = chunk
                    confidence_str = f"Score: {confidence}" if confidence else ""
                    chunk_str = f"Patient ID: {pid}\nStudy Date: {study_date}\nImage Type: {image_type}\nFindings: {findings}\n{confidence_str}"
                    
                    if current_len + len(chunk_str) > MAX_CONTEXT_CHARS:
                        break
                    
                    context_parts.append(chunk_str)
                    current_len += len(chunk_str)
                    
                context = "\n\n".join(context_parts)
                
            # ====================== PROMPT ======================
            patient_context_block = ""
            if st.session_state.get("chat_mode") == "Patient Mode" and st.session_state.get("selected_patient_id"):
                p_id = st.session_state["selected_patient_id"]
                p_name = st.session_state["selected_patient_name"]
                patient_context_block = f"\n[ CURRENT PATIENT ]\nPatient Name: {p_name}\nPatient ID: {p_id}\n\nRULE:\nAnswer ONLY using this patient's records. Do not mix records from other patients.\n"
            elif st.session_state.get("chat_mode") == "Database Mode":
                patient_context_block = "\n[ GLOBAL DATABASE MODE ]\nYou may search across all patients to answer the user's query.\n"

            prompt_template = ChatPromptTemplate.from_template("""
You are a direct, highly-precise clinical assistant.
{patient_context_block}

[ PATIENT LAB EVENT CONTEXT ]
{context}

[ STRICT RULES - NO HALLUCINATIONS ]
1. Answer using ONLY natural language. NEVER output raw SQL queries or database code.
2. Focus strictly on clinical summarization. DO NOT hallucinate, guess, or bluff. If you do not know, say you do not know.
3. Provide cohesive patient insights based exactly on the provided Context. Do NOT repeat the same sentences multiple times.
4. Output ONLY the final analytical answer. DO NOT explain your reasoning. YOU MUST NOT output any internal thoughts, <unused94>thought tags, or step-by-step reasoning lists. Jump immediately to the final answer.
5. If the exact answer cannot be confidently deduced from the Context, you MUST output exactly: "Not found in database". Do not attempt to guess or infer.

Question: {question}

Final Clinical Answer:
""")

            final_prompt = prompt_template.format(
                patient_context_block=patient_context_block,
                context=context,
                question=prompt
            )

            # ====================== LLM ======================
            selected_model = get_llm(prompt)
            final_cleaned_response = ""
            
            try:
                llm = ChatOllama(
                    model=selected_model,
                    temperature=TEMPERATURE,
                    num_ctx=4096,  # Increased to prevent generation cutoff
                    repeat_penalty=1.2 # Hard constraint against repetitive looping
                )

                stream = llm.stream(final_prompt)

                for chunk in stream:
                    if hasattr(chunk, "content"):
                        full_response += chunk.content
                    else:
                        full_response += str(chunk)

                    display_response = clean_llm_output(full_response)
                    if display_response:
                        placeholder.markdown(display_response + "▌")
                    else:
                        placeholder.markdown("*Thinking...*")

                final_cleaned_response = clean_llm_output(full_response)
                placeholder.markdown(final_cleaned_response)
            
            except Exception as e:
                final_cleaned_response = f"An error occurred while communicating with the AI model: {e}"
                placeholder.markdown(final_cleaned_response)

            # ====================== TIME ======================
            end_time = time.time()
            st.caption(f"⏱️ {(end_time - start_time):.2f} sec | 🧠 Answered by: {selected_model}")

            # ====================== DEBUG LOGGING ======================
            # Print to terminal console
            debug_detected_name = detected_name if detected_name else "None"
            debug_verified_id = verified_patient_id if verified_patient_id else "None"
            
            # Retrieved chunk count is sum of sql rows and faiss chunks retrieved
            retrieved_chunk_count = debug_sql_records + debug_faiss_chunks
            
            print("\n=== RAG PIPELINE DEBUG LOGS ===")
            print(f"Detected patient name: {debug_detected_name}")
            print(f"SQL query executed: {debug_sql_query}")
            print(f"Retrieved patient_id: {debug_verified_id}")
            print(f"Retrieved chunk count: {retrieved_chunk_count}")
            print(f"Unique patient_ids found: {list(retrieved_pids)}")
            print("===============================\n")

            # Render in Streamlit UI
            with st.expander("Developer Debug Logs", expanded=False):
                is_patient_filter = (st.session_state.get("chat_mode") == "Patient Mode" and st.session_state.get("selected_patient_id") is not None)
                st.markdown(f"**Selected Model:** `{selected_model}`")
                st.markdown(f"**Current Chat Mode:** `{st.session_state.get('chat_mode', 'Database Mode')}`")
                st.markdown(f"**Selected Patient ID:** `{st.session_state.get('selected_patient_id', 'None')}`")
                st.markdown(f"**Selected Patient Name:** `{st.session_state.get('selected_patient_name', 'None')}`")
                st.markdown(f"**Patient Filter Applied:** `{is_patient_filter}`")
                st.markdown(f"**Retrieved Documents Count:** `{retrieved_chunk_count}`")
                st.markdown(f"**Retrieved Patient IDs:** `{list(retrieved_pids)}`")
                st.markdown(f"**Detected Patient Name:** `{debug_detected_name}`")
                st.markdown(f"**Verified Patient ID:** `{debug_verified_id}`")
                st.markdown(f"**SQL Query Executed:**\n```sql\n{debug_sql_query}\n```")
                st.markdown(f"**Retrieved patient_id:** `{debug_verified_id}`")
                st.markdown(f"**Retrieved chunk count:** `{retrieved_chunk_count}`")
                st.markdown(f"**Unique patient_ids found:** `{list(retrieved_pids)}`")
                if verified_patient_id is not None:
                    st.markdown(f"**FAISS Chunks Discarded:** `{discarded_count}`")
                    if mismatched_pids:
                        st.markdown(f"**Mismatched Patient IDs Discarded:** `{list(mismatched_pids)}`")
                
                st.divider()
                st.markdown("### 📊 Database Statistics")
                st.markdown(f"- **Total Registered Patients:** {stats['total_patients']}")
                st.markdown(f"- **Total EHR Lab Events:** {stats['total_labevents']}")
                st.markdown(f"**Patient Imaging Mapping:**\n{stats.get('imaging_summary', '')}")
                
                st.markdown("### 📄 Patient Lab Event Context")
                st.text_area("Context Chunks Sent to LLM", context, height=150)
                
                st.markdown("### ⚙️ LLM Prompt Instructions")
                st.text_area("Final Prompt Generation Input", final_prompt, height=250)

    final_assistant_content = final_cleaned_response
    st.session_state.messages.append({"role": "assistant", "content": final_assistant_content})