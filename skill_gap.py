# 🔹 Skill Gap Analysis Tool (ML Version)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1️⃣ Create Sample Dataset

data = {
    "python": [8, 7, 3, 9, 4, 6, 2, 10, 5, 7],
    "sql": [7, 6, 4, 8, 3, 5, 2, 9, 5, 6],
    "machine_learning": [6, 5, 2, 9, 2, 7, 1, 10, 4, 6],
    "communication": [7, 8, 5, 9, 4, 8, 6, 10, 6, 7],
    "job_ready": ["Yes", "Yes", "No", "Yes", "No", 
                  "Yes", "No", "Yes", "No", "Yes"]
}

df = pd.DataFrame(data)


# 2️⃣ Define Features & Target

X = df.drop("job_ready", axis=1)
y = df["job_ready"]


# 3️⃣ Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4️⃣ Train Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# 5️⃣ Evaluate Model

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# 6️⃣ Skill Gap Analysis Function

def skill_gap_analysis(candidate_skills):
    
    prediction = model.predict([candidate_skills])[0]
    
    print("\nPredicted Job Readiness:", prediction)
    
    if prediction == "Yes":
        print("✅ Candidate is job ready.")
    else:
        print("❌ Candidate not job ready.")
        print("⚠ Skill Gaps Found:")
        
        skill_names = X.columns
        for skill, score in zip(skill_names, candidate_skills):
            if score < 5:
                print(f"- Improve {skill}")


# 7️⃣ Test New Candidate

# Format: [python, sql, machine_learning, communication]
new_candidate = [5, 4, 3, 6]

skill_gap_analysis(new_candidate)


#  Show Feature Importance

importance = model.feature_importances_

print("\nSkill Importance:")
for skill, score in zip(X.columns, importance):
    print(f"{skill}: {round(score, 3)}")