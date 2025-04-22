import mysql.connector
from config import DB_CONFIG

def create_connection():
    return mysql.connector.connect(**DB_CONFIG)

def create_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            email VARCHAR(255),
            phone VARCHAR(20),
            resume_file VARCHAR(255),
            similarity_score FLOAT
        )
    """)
    conn.commit()
    conn.close()

def insert_applicant_data(name, email, phone, resume_file, similarity_score):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO resumes (name, email, phone, resume_file, similarity_score)
        VALUES (%s, %s, %s, %s, %s)
    """, (name, email, phone, resume_file, similarity_score))
    conn.commit()
    conn.close()
