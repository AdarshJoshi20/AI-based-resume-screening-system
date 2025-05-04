import os

UPLOAD_FOLDER = "static/uploads"
TOP_N_RESULTS = 5
SIMILARITY_THRESHOLD = 0.30
MIN_WORD_COUNT = 300

DB_CONFIG = {
    'host': os.environ.get("DB_HOST", "localhost"),
    'user': os.environ.get("DB_USER"),
    'password': os.environ.get("DB_PASS"),
    'database': "resume_db"
}
