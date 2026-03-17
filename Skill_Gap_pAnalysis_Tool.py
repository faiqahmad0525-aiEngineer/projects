# =========================================
# 🔹 Skill Gap Analysis Tool (NLP + Clustering)
# =========================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# -----------------------------------------
# 🔹 Step 1: Sample Data (You can replace with real dataset)
# -----------------------------------------

intern_data = pd.DataFrame({
    'intern_name': ['Ali', 'Sara', 'John'],
    'skills': [
        'python machine learning data analysis',
        'html css javascript frontend',
        'python deep learning tensorflow'
    ]
})

job_data = pd.DataFrame({
    'job_role': ['Data Scientist', 'Frontend Developer', 'AI Engineer'],
    'requirements': [
        'python machine learning statistics deep learning',
        'html css javascript react ui ux',
        'python deep learning pytorch ai models'
    ]
})

# -----------------------------------------
# 🔹 Step 2: Combine Text Data
# -----------------------------------------

all_text = intern_data['skills'].tolist() + job_data['requirements'].tolist()

# -----------------------------------------
# 🔹 Step 3: Convert Text → TF-IDF
# -----------------------------------------

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_text)

# -----------------------------------------
# 🔹 Step 4: Apply K-Means Clustering
# -----------------------------------------

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(tfidf_matrix)

# Split clusters
intern_clusters = kmeans.labels_[:len(intern_data)]
job_clusters = kmeans.labels_[len(intern_data):]

intern_data['cluster'] = intern_clusters
job_data['cluster'] = job_clusters

# -----------------------------------------
# 🔹 Step 5: Skill Gap Analysis
# -----------------------------------------

def find_skill_gap(intern_skills, job_skills):
    intern_set = set(intern_skills.split())
    job_set = set(job_skills.split())
    
    missing_skills = job_set - intern_set
    return list(missing_skills)

# -----------------------------------------
# 🔹 Step 6: Match Intern with Closest Job
# -----------------------------------------

results = []

for i, intern in intern_data.iterrows():
    same_cluster_jobs = job_data[job_data['cluster'] == intern['cluster']]
    
    for j, job in same_cluster_jobs.iterrows():
        gap = find_skill_gap(intern['skills'], job['requirements'])
        
        results.append({
            'Intern': intern['intern_name'],
            'Job Role': job['job_role'],
            'Missing Skills': gap
        })

# -----------------------------------------
# 🔹 Step 7: Show Results
# -----------------------------------------

result_df = pd.DataFrame(results)

print("\n🔹 Skill Gap Analysis Results:\n")
print(result_df)
