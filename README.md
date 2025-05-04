# ResumeRadar

## Overview

ResumeRadar is an AI-powered platform designed to match resumes with job descriptions using advanced natural language processing techniques. The system helps recruiters and hiring managers efficiently identify the most suitable candidates for any position by analyzing and ranking resumes based on their relevance to a given job description.

Developed as a Major Project by final year students of Graphic Era University, Dehradun (B.Tech CSE Batch of 2025).


## Features

- **Resume Matching**: Upload multiple resumes and a job description to get instant matches
- **AI-Powered Analysis**: Uses sentence transformers and TF-IDF vectorization for accurate matching
- **Detailed Metrics**: View comprehensive evaluation metrics for the matching process
- **Visualization**: See t-SNE visualizations of document embeddings
- **Database Storage**: Save applicant information and match scores for future reference
- **Resume Validation**: Intelligent filtering to ensure uploaded files are valid resumes

## How It Works

ResumeRadar uses a combination of traditional NLP techniques and modern deep learning to match resumes with job descriptions:

1. **Text Extraction**: Extracts text from various document formats (PDF, DOCX, TXT)
2. **Dual Scoring System**:
   - TF-IDF vectorization for keyword matching
   - Sentence transformers (all-mpnet-base-v2) for semantic understanding
3. **Combined Ranking**: Normalizes and combines both scores for a balanced assessment
4. **Evaluation Metrics**: Provides detailed accuracy metrics including:
   - Method consistency (agreement between different scoring approaches)
   - Keyword coverage
   - Normalized Discounted Cumulative Gain (NDCG)
   - Perturbation robustness
   - Score distribution analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- MySQL database
- Required Python packages (listed in requirements.txt)

### Setup Instructions

1. **Clone the repository**:
   ```
   git clone https://github.com/AdarshJoshi20/AI-based-resume-screening-system.git
   cd AI-based-resume-screening-system
   ```

2. **Create a virtual environment**:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root with the following:
   ```
   DB_HOST=localhost
   DB_USER=your_mysql_username
   DB_PASS=your_mysql_password
   ```

5. **Create the MySQL database**:
   ```
   mysql -u your_username -p
   CREATE DATABASE resume_db;
   exit;
   ```

6. **Create upload directory**:
   ```
   mkdir -p static/uploads
   ```

7. **Create reference resume directory**:
   ```
   mkdir reference_resumes
   ```
   Add a few sample resumes to this directory for the system to use as reference.

## Usage

1. **Start the application**:
   ```
   python app1.py
   ```

2. **Access the web interface**:
   Open your browser and navigate to `http://localhost:5000`

3. **Using the system**:
   - Enter a job description in the text area
   - Upload multiple resumes (PDF, DOCX, or TXT format)
   - Specify how many top results to display
   - Click "Match Resumes" to get results

4. **View Results**:
   - See a ranked list of matching resumes with similarity scores
   - Access detailed evaluation metrics and visualizations
   - View stored resumes via the "View Resumes" page

## System Architecture

ResumeRadar consists of the following main components:

- **Web Interface** (Flask): Provides user interface for uploading and viewing results
- **Preprocessing**: Extracts and validates text from various document formats
- **Core Matching Engine**: Combines TF-IDF and semantic embedding approaches
- **Evaluation System**: Calculates various metrics to assess ranking quality
- **Database Layer**: Stores resume information and match results

## Troubleshooting

- **Database Connection Issues**: 
  - Ensure MySQL is running
  - Verify credentials in your `.env` file
  - Check that the database `resume_db` exists

- **File Upload Problems**:
  - Maximum file size is 2MB
  - Only PDF, DOCX, and TXT formats are supported
  - Files must contain sufficient text to be recognized as resumes

- **Model Loading Errors**:
  - Ensure internet connectivity for initial model download
  - Check that the sentence-transformers library is properly installed

## Contributors

- Adarsh Joshi
- Vasu Kush
- Apoorva Rana
- Vipul Bora

## License

© 2025 ResumeRadar. All rights reserved.

---

For any questions or support, please contact the project contributors.