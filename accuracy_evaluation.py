def evaluate_ranking_accuracy(job_description, resumes, top_n=10):
    """
    Evaluates the accuracy of the resume ranking system using multiple approaches.
    
    Args:
        job_description (str): The job description text
        resumes (list): List of resume texts
        top_n (int): Number of top results to consider
        
    Returns:
        dict: Dictionary containing various accuracy metrics
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sentence_transformers import SentenceTransformer
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import io
    import base64
    
    # Initialize models (should match your main ranking model)
    model = SentenceTransformer('all-mpnet-base-v2')
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Get embeddings and similarities (same as your main ranking function)
    jd_vector = model.encode([job_description.lower()])[0]
    resume_texts = [resume.lower() for resume in resumes]
    
    # TFIDF calculation
    all_docs = [job_description.lower()] + resume_texts
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_docs)
    tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    
    # Semantic similarity
    resume_vectors = model.encode(resume_texts)
    semantic_scores = cosine_similarity([jd_vector], resume_vectors)[0]
    
    # Normalize scores (same as main function)
    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min()) if arr.max() > arr.min() else np.ones_like(arr)
    
    tfidf_scores_norm = normalize(tfidf_scores)
    semantic_scores_norm = normalize(semantic_scores)
    combined_scores = 0.5 * tfidf_scores_norm + 0.5 * semantic_scores_norm
    
    # Get rankings
    ranked_indices = np.argsort(combined_scores)[::-1]
    
    # 1. Consistency Analysis: Calculate agreement between different scoring methods
    # Spearman's rank correlation between TFIDF and semantic scores
    tfidf_rank = np.argsort(tfidf_scores_norm)[::-1]  
    semantic_rank = np.argsort(semantic_scores_norm)[::-1]
    
    # Convert to rankings
    tfidf_ranking = np.zeros_like(tfidf_rank)
    tfidf_ranking[tfidf_rank] = np.arange(len(tfidf_rank))
    
    semantic_ranking = np.zeros_like(semantic_rank)
    semantic_ranking[semantic_rank] = np.arange(len(semantic_rank))
    
    # Calculate Spearman correlation
    n = len(tfidf_ranking)
    d_squared = np.sum((tfidf_ranking - semantic_ranking) ** 2)
    spearman_corr = 1 - (6 * d_squared) / (n * (n**2 - 1))
    
    # 2. Keyword Overlap Analysis
    # Extract keywords from job description using TFIDF
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    jd_tfidf = vectorizer.fit_transform([job_description.lower()])
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top keywords from job description
    jd_tfidf_sorted = np.argsort(jd_tfidf.toarray()[0])[::-1]
    top_keywords = feature_names[jd_tfidf_sorted[:20]]  # Top 20 keywords
    
    # Check keyword presence in top ranked resumes
    keyword_coverage = []
    for idx in ranked_indices[:top_n]:
        resume_text = resume_texts[idx]
        keywords_present = sum(1 for keyword in top_keywords if keyword in resume_text)
        keyword_coverage.append(keywords_present / len(top_keywords))
    
    avg_keyword_coverage = np.mean(keyword_coverage) if keyword_coverage else 0
    
    # 3. Distribution Analysis
    # Calculate statistics on the similarity scores
    score_mean = np.mean(combined_scores)
    score_std = np.std(combined_scores)
    score_range = np.max(combined_scores) - np.min(combined_scores)
    
    # Calculate interquartile range for outlier detection
    q1 = np.percentile(combined_scores, 25)
    q3 = np.percentile(combined_scores, 75)
    iqr = q3 - q1
    
    # 4. Perturbation Testing (basic implementation)
    perturbation_robustness = []
    
    if len(resume_texts) > 0:
        # Take the highest ranked resume
        top_resume = resume_texts[ranked_indices[0]]
        top_resume_words = top_resume.split()
        
        # Create perturbations by removing words (simulating missing skills)
        perturbations = []
        if len(top_resume_words) > 100:  # Only perturb if resume is long enough
            for i in range(1, 6):  # 5 levels of perturbation
                # Remove i*5% of words randomly
                removal_count = int(len(top_resume_words) * i * 0.05)
                if removal_count >= len(top_resume_words):
                    continue
                    
                import random
                perturbed_words = top_resume_words.copy()
                for _ in range(removal_count):
                    if perturbed_words:  # Check if list is not empty
                        random_index = random.randint(0, len(perturbed_words) - 1)
                        perturbed_words.pop(random_index)
                        
                perturbed_text = " ".join(perturbed_words)
                perturbations.append(perturbed_text)
            
            # Get scores for perturbations
            if perturbations:
                pert_texts = [job_description.lower()] + perturbations
                pert_tfidf = tfidf_vectorizer.transform(pert_texts)
                pert_tfidf_scores = cosine_similarity(pert_tfidf[0:1], pert_tfidf[1:])[0]
                
                pert_vectors = model.encode(perturbations)
                pert_semantic_scores = cosine_similarity([jd_vector], pert_vectors)[0]
                
                pert_tfidf_norm = normalize(pert_tfidf_scores)
                pert_semantic_norm = normalize(pert_semantic_scores)
                pert_combined = 0.5 * pert_tfidf_norm + 0.5 * pert_semantic_norm
                
                # Calculate how much scores decrease with increasing perturbation
                original_score = combined_scores[ranked_indices[0]]
                expected_decrease = np.linspace(0, 0.5, len(pert_combined))  # Expected proportional decrease
                actual_decrease = [original_score - score for score in pert_combined]
                
                # Correlation between expected and actual decrease
                if len(pert_combined) > 1:
                    perturbation_corr = np.corrcoef(expected_decrease, actual_decrease)[0, 1]
                    perturbation_robustness = float(perturbation_corr) if not np.isnan(perturbation_corr) else 0
                else:
                    perturbation_robustness = 0
    
    # 5. DCG/NDCG calculation with keyword overlap as relevance proxy
    relevance_scores = []
    for idx in range(len(resume_texts)):
        resume_text = resume_texts[idx]
        keywords_present = sum(1 for keyword in top_keywords if keyword in resume_text)
        relevance_scores.append(keywords_present / len(top_keywords))
    
    # DCG calculation
    def calculate_dcg(scores, k):
        """Calculate DCG@k for given relevance scores."""
        dcg = 0
        for i in range(min(k, len(scores))):
            dcg += scores[i] / np.log2(i + 2)  # i+2 because i is 0-indexed
        return dcg
    
    # Calculate DCG for our ranking
    our_ranking_dcg = calculate_dcg([relevance_scores[i] for i in ranked_indices[:top_n]], top_n)
    
    # Calculate ideal DCG (relevance scores sorted in descending order)
    ideal_order = np.argsort(relevance_scores)[::-1]
    ideal_dcg = calculate_dcg([relevance_scores[i] for i in ideal_order[:top_n]], top_n)
    
    # Calculate NDCG
    ndcg = our_ranking_dcg / ideal_dcg if ideal_dcg > 0 else 0
    
    # Visualize embeddings (save as base64 image)
    plt.figure(figsize=(10, 8))
    
    # Combine job description vector with resume vectors for visualization
    all_vectors = np.vstack([jd_vector.reshape(1, -1), resume_vectors])
    
    # Use t-SNE for dimensionality reduction
    n_samples = all_vectors.shape[0]
    perplexity = min(30, max(5, n_samples - 1))  # Ensure perplexity is at least 5 but less than n_samples
    
    # Skip visualization if too few samples
    if n_samples >= 5:  # t-SNE typically needs at least 5 samples to work well
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        all_vectors_2d = tsne.fit_transform(all_vectors)
        
        # Plot job description (red star)
        plt.scatter(all_vectors_2d[0, 0], all_vectors_2d[0, 1], color='red', s=200, marker='*', label='Job Description')
        
        # Plot top N resumes (green)
        top_n_indices = ranked_indices[:min(top_n, len(ranked_indices))]
        plt.scatter(
            all_vectors_2d[1:, 0][top_n_indices], 
            all_vectors_2d[1:, 1][top_n_indices], 
            color='green', 
            label=f'Top {len(top_n_indices)} Resumes'
        )
        
        # Plot remaining resumes (gray)
        remaining_indices = [i for i in range(len(resume_texts)) if i not in top_n_indices]
        if remaining_indices:
            plt.scatter(
                all_vectors_2d[1:, 0][remaining_indices], 
                all_vectors_2d[1:, 1][remaining_indices], 
                color='gray', 
                alpha=0.5, 
                label='Other Resumes'
            )
        
        plt.title('t-SNE Visualization of Job Description and Resume Embeddings')
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Not enough samples for t-SNE visualization", 
                ha='center', va='center', fontsize=12)
        plt.title('Visualization not available')
    
    # Save plot to a base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    embedding_viz = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Return all metrics
    return {
        "method_consistency": float(spearman_corr),
        "keyword_coverage": float(avg_keyword_coverage),
        "score_distribution": {
            "mean": float(score_mean),
            "std": float(score_std),
            "range": float(score_range),
            "iqr": float(iqr),
        },
        "perturbation_robustness": float(perturbation_robustness) if isinstance(perturbation_robustness, (int, float)) else 0,
        "ndcg": float(ndcg),
        "embedding_visualization": embedding_viz
    }


def evaluate_and_render_accuracy_report(job_description, top_resumes, all_resume_texts, top_n=10):
    """
    Evaluates accuracy and renders results as HTML for display in Flask.
    
    Args:
        job_description (str): The job description text
        top_resumes (list): List of top resume dictionaries from main ranking
        all_resume_texts (list): List of all resume texts
        top_n (int): Number of top results to consider
        
    Returns:
        str: HTML representation of the accuracy report
    """
    # Run evaluation
    metrics = evaluate_ranking_accuracy(job_description, all_resume_texts, top_n)
    
    # Prepare HTML report
    html = """
    <div class="card mt-4">
        <div class="card-header bg-info text-white">
            <h4>Ranking System Evaluation Report</h4>
        </div>
        <div class="card-body">
            <div class="row">
                <div class="col-md-6">
                    <h5>Key Metrics</h5>
                    <table class="table table-bordered">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Interpretation</th>
                        </tr>
                        <tr>
                            <td>Method Consistency</td>
                            <td>{:.2f}</td>
                            <td>{}</td>
                        </tr>
                        <tr>
                            <td>Keyword Coverage</td>
                            <td>{:.2f}</td>
                            <td>{}</td>
                        </tr>
                        <tr>
                            <td>NDCG@{}</td>
                            <td>{:.2f}</td>
                            <td>{}</td>
                        </tr>
                        <tr>
                            <td>Perturbation Robustness</td>
                            <td>{:.2f}</td>
                            <td>{}</td>
                        </tr>
                    </table>
                    
                    <h5>Score Distribution</h5>
                    <table class="table table-bordered">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Mean Score</td>
                            <td>{:.3f}</td>
                        </tr>
                        <tr>
                            <td>Standard Deviation</td>
                            <td>{:.3f}</td>
                        </tr>
                        <tr>
                            <td>Range</td>
                            <td>{:.3f}</td>
                        </tr>
                        <tr>
                            <td>Interquartile Range</td>
                            <td>{:.3f}</td>
                        </tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h5>Embedding Visualization</h5>
                    <img src="data:image/png;base64,{}" class="img-fluid" alt="t-SNE visualization">
                </div>
            </div>
        </div>
    </div>
    """.format(
        metrics["method_consistency"],
        get_consistency_interpretation(metrics["method_consistency"]),
        metrics["keyword_coverage"],
        get_keyword_coverage_interpretation(metrics["keyword_coverage"]),
        top_n,
        metrics["ndcg"],
        get_ndcg_interpretation(metrics["ndcg"]),
        metrics["perturbation_robustness"],
        get_perturbation_interpretation(metrics["perturbation_robustness"]),
        metrics["score_distribution"]["mean"],
        metrics["score_distribution"]["std"],
        metrics["score_distribution"]["range"],
        metrics["score_distribution"]["iqr"],
        metrics["embedding_visualization"]
    )
    
    return html


def get_consistency_interpretation(value):
    """Interpret the consistency score"""
    if value > 0.8:
        return "Excellent consistency between ranking methods"
    elif value > 0.6:
        return "Good consistency between ranking methods"
    elif value > 0.4:
        return "Moderate consistency between ranking methods"
    else:
        return "Low consistency - ranking methods disagree"


def get_keyword_coverage_interpretation(value):
    """Interpret the keyword coverage score"""
    if value > 0.8:
        return "Excellent keyword coverage in top resumes"
    elif value > 0.6:
        return "Good keyword coverage in top resumes"
    elif value > 0.4:
        return "Moderate keyword coverage in top resumes"
    else:
        return "Low keyword coverage in top resumes"


def get_ndcg_interpretation(value):
    """Interpret the NDCG score"""
    if value > 0.9:
        return "Excellent ranking quality"
    elif value > 0.7:
        return "Good ranking quality"
    elif value > 0.5:
        return "Moderate ranking quality"
    else:
        return "Needs improvement"


def get_perturbation_interpretation(value):
    """Interpret the perturbation robustness score"""
    if value > 0.8:
        return "Excellent robustness to missing information"
    elif value > 0.6:
        return "Good robustness to missing information"
    elif value > 0.4:
        return "Moderate robustness to missing information"
    else:
        return "Low robustness - sensitive to missing information"