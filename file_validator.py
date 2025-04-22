import os
import torch
import fitz  # PyMuPDF
import docx
from sentence_transformers import util

from model_loader import model
from config import SIMILARITY_THRESHOLD, MIN_WORD_COUNT

# ✅ Allowed file types
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}

# ✅ Extract plain text from resume file
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text

        elif ext == '.docx':
            doc = docx.Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])

        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    except Exception as e:
        print(f"❌ Error extracting text from {file_path}: {e}")
    
    return ""

# ✅ Check if file type is allowed
def is_allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

# ✅ Keyword match helper
def contains_resume_keywords(text):
    keywords = [
        "education", "project", "internship", "experience", "skills",
        "certification", "technical", "achievements", "responsibility",
        "university", "college", "CGPA", "challenge", "hackathon",
        "teamwork", "leadership", "communication", "problem-solving"
    ]
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)

# ✅ Cache reference resume vector
_reference_vector = None
def get_reference_resume_vector(folder='reference_resumes'):
    global _reference_vector
    if _reference_vector is not None:
        return _reference_vector

    vectors = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            path = os.path.join(folder, filename)
            text = extract_text(path)
            if text.strip():
                embedding = model.encode(text, convert_to_tensor=True)
                vectors.append(embedding)

    if not vectors:
        raise ValueError("No valid reference resumes found for similarity comparison.")

    _reference_vector = torch.mean(torch.stack(vectors), dim=0)
    return _reference_vector

# ✅ Validate if uploaded file is a resume
def is_resume(file_path):
    try:
        text = extract_text(file_path)
        if len(text.split()) < MIN_WORD_COUNT:
            return False

        embedding = model.encode(text, convert_to_tensor=True)
        sim_score = util.pytorch_cos_sim(embedding, get_reference_resume_vector()).item()
        keyword_boost = contains_resume_keywords(text)

        if sim_score >= SIMILARITY_THRESHOLD:
            return True
        elif sim_score >= (SIMILARITY_THRESHOLD - 0.10) and keyword_boost >= 6:
            return True
        else:
            return False

    except Exception as e:
        print(f"❌ Resume validation failed for {file_path}: {e}")
        return False
