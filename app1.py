from dotenv import load_dotenv
load_dotenv()
import signal
import sys
import atexit
import json
from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pyresparser import ResumeParser

from config import UPLOAD_FOLDER, TOP_N_RESULTS
from file_validator import is_allowed_file, extract_text, is_resume
from db_connection import create_table, insert_applicant_data, create_connection
from accuracy_evaluation import evaluate_ranking_accuracy, evaluate_and_render_accuracy_report

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
model = SentenceTransformer('all-mpnet-base-v2')
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Ensure DB table exists
create_table()

# Clean up function
def cleanup_resources():
    # Force cleanup of any Tkinter resources
    try:
        import tkinter as tk
        if 'tk' in sys.modules:
            if tk._default_root:
                tk._default_root.destroy()
    except:
        pass
    
    print("Application shutting down...")

# Register cleanup function
atexit.register(cleanup_resources)

# Handle SIGINT (Ctrl+C)
def signal_handler(sig, frame):
    print('Received SIGINT, shutting down gracefully...')
    cleanup_resources()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# 🏠 Homepage + Dashboard
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        jd_text = request.form.get('job_description')
        jd_text = jd_text.lower()
        uploaded_files = request.files.getlist('resumes')
        top_n = int(request.form.get('top_n', TOP_N_RESULTS))

        if not jd_text or not uploaded_files:
            return render_template("dashboard.html", message="Job description and resumes are required.")

        top_resumes, file_errors, all_resume_texts = rank_resumes_from_input(jd_text, uploaded_files, top_n)

        # Generate accuracy metrics directly (don't just return HTML)
        metrics = None
        if all_resume_texts:  # Only generate metrics if we have resumes
            metrics = evaluate_ranking_accuracy(jd_text, all_resume_texts, top_n)    

        return render_template("dashboard.html", 
                              resumes=top_resumes, 
                              file_errors=file_errors, 
                              metrics=metrics)

    return render_template("dashboard.html")

# 🧠 Core Matching Logic
def rank_resumes_from_input(jd_text, uploaded_files, top_n=TOP_N_RESULTS):
    jd_text = jd_text.lower()
    jd_vector = model.encode([jd_text])[0]
    texts = []
    metadata = []
    file_errors = []

    for file in uploaded_files:
        if not is_allowed_file(file.filename):
            file_errors.append({'filename': file.filename, 'error': "Unsupported file type."})
            continue

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size > 2 * 1024 * 1024:
            file_errors.append({'filename': file.filename, 'error': "File exceeds 2MB size limit."})
            continue

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        if not is_resume(save_path):
            file_errors.append({'filename': filename, 'error': "File doesn't appear to be a valid resume."})
            os.remove(save_path)
            continue

        text = extract_text(save_path)
        if not text:
            file_errors.append({'filename': filename, 'error': "Failed to extract text from resume."})
            continue

        parsed_data = ResumeParser(save_path).get_extracted_data()
        name = parsed_data.get('name') or 'N/A'
        email = parsed_data.get('email') or 'N/A'
        phone = parsed_data.get('mobile_number') or 'N/A'

        texts.append(text.lower())
        metadata.append({'filename': filename, 'name': name, 'email': email, 'phone': phone})

    if not texts:
        return [], file_errors, []

    # Compute similarities
    all_docs = [jd_text] + texts
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_docs)
    tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    resume_vectors = model.encode(texts)
    semantic_scores = cosine_similarity([jd_vector], resume_vectors)[0]

    # Normalize and combine scores
    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min()) if arr.max() > arr.min() else np.ones_like(arr)

    tfidf_scores = normalize(tfidf_scores)
    semantic_scores = normalize(semantic_scores)
    combined_scores = 0.5 * tfidf_scores + 0.5 * semantic_scores

    results = []
    for meta, score in zip(metadata, combined_scores):
        score = float(score) * 100  # Convert from numpy float to Python float
        insert_applicant_data(meta['name'], meta['email'], meta['phone'], meta['filename'], score)
        results.append({**meta, 'similarity_score': score})

    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:top_n], file_errors, texts


@app.route('/view_resumes', methods=['GET'])
def view_resumes():
    page = int(request.args.get('page', 1))
    per_page = 10
    offset = (page - 1) * per_page

    conn = create_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT COUNT(*) AS total FROM resumes")
    total = cursor.fetchone()['total']

    cursor.execute("SELECT * FROM resumes ORDER BY similarity_score DESC LIMIT %s OFFSET %s", (per_page, offset))
    resumes = cursor.fetchall()
    conn.close()

    total_pages = (total + per_page - 1) // per_page

    return render_template("view_resumes.html", resumes=resumes, page=page, total_pages=total_pages)


@app.route('/evaluate_system', methods=['GET', 'POST'])
def evaluate_system():
    """Endpoint for standalone system evaluation"""
    if request.method == 'POST':
        jd_text = request.form.get('job_description')
        uploaded_files = request.files.getlist('resumes')
        top_n = int(request.form.get('top_n', TOP_N_RESULTS))
        
        if not jd_text or not uploaded_files:
            return render_template("evaluate.html", message="Job description and resumes are required.")
            
        # Use existing function to process files
        _, file_errors, all_resume_texts = rank_resumes_from_input(jd_text, uploaded_files, top_n)
        
        if not all_resume_texts:
            return render_template("evaluate.html", message="No valid resumes found for evaluation.")
            
        # Get full evaluation metrics
        metrics = evaluate_ranking_accuracy(jd_text, all_resume_texts, top_n)
        
        return render_template("evaluate.html", metrics=metrics, file_errors=file_errors)
        
    return render_template("evaluate.html")

# ℹ️ About page
@app.route('/about')
def about():
    return render_template("about1.html")

if __name__ == '__main__':
    app.run(debug=True)