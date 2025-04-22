from flask import Flask, render_template, request, redirect, url_for
import os
import json
from werkzeug.utils import secure_filename

from dotenv import load_dotenv
load_dotenv()

# ✅ Debug prints to check if .env is being read
print("DB_USER:", os.environ.get("DB_USER"))
print("DB_PASS:", os.environ.get("DB_PASS"))

from model_loader import model
from db_connection import create_table, insert_applicant_data
from file_validator import is_allowed_file, is_resume
from config import UPLOAD_FOLDER, TOP_N_RESULTS

from pyresparser import ResumeParser
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB file limit

# Create DB table on startup
create_table()


# 🧠 Extract data from resume
def parse_resume_data(file_path):
    try:
        data = ResumeParser(file_path).get_extracted_data()
        return json.dumps(data)
    except Exception as e:
        print(f"Failed to parse {file_path}: {e}")
        return None


# 🧠 Core Matching Logic
def rank_resumes_from_input(jd_text, uploaded_files, top_n=TOP_N_RESULTS):
    jd_embedding = model.encode([jd_text])[0]
    results = []
    file_errors = []

    for file in uploaded_files:
        if not is_allowed_file(file.filename):
            file_errors.append({'filename': file.filename, 'error': "Unsupported file type."})
            continue

        # ✅ New: Check file size manually
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset pointer for saving

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

        resume_json = parse_resume_data(save_path)
        if resume_json:
            resume_embedding = model.encode([resume_json])[0]
            similarity = cosine_similarity([jd_embedding], [resume_embedding])[0][0]
            score = round(float(similarity * 100), 2)

            resume_data = json.loads(resume_json)
            name = resume_data.get('name', 'Unknown')
            email = resume_data.get('email', 'N/A')
            phone = resume_data.get('mobile_number', 'N/A')

            insert_applicant_data(name, email, phone, filename, score)

            results.append({
                'filename': filename,
                'name': name,
                'email': email,
                'phone': phone,
                'similarity_score': score
            })
        else:
            file_errors.append({'filename': filename, 'error': "Parsing failed."})

    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:top_n], file_errors



# 🏠 Homepage + Dashboard
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        jd_text = request.form.get('job_description')
        uploaded_files = request.files.getlist('resumes')
        top_n = int(request.form.get('top_n', TOP_N_RESULTS))

        if not jd_text or not uploaded_files:
            return render_template("dashboard.html", message="Job description and resumes are required.")

        top_resumes, file_errors = rank_resumes_from_input(jd_text, uploaded_files, top_n)
        return render_template("dashboard.html", resumes=top_resumes, file_errors=file_errors)

    return render_template("dashboard.html")



# 📄 View matched resumes from DB
@app.route('/view_resumes')
def view_resume():
    from db_connection import create_connection
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT resume_file, similarity_score FROM resumes ORDER BY similarity_score DESC LIMIT 5")
    data = cursor.fetchall()
    conn.close()

    resumes = [{'filename': row[0], 'similarity_score': row[1]} for row in data]
    return render_template("view_resumes.html", resumes=resumes)


# ℹ️ About page
@app.route('/about')
def about():
    return render_template("about1.html")


# 📛 Error handling
@app.errorhandler(413)
def file_too_large(e):
    return render_template("dashboard.html", message="❌ File too large. Max allowed size is 2 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)
