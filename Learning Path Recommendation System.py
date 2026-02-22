import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1️⃣ Create Dataset

data = {
    "skill_level": [1, 2, 3, 1, 2, 3, 2, 1, 3, 2],
    "interest_area": [1, 1, 2, 2, 3, 3, 2, 3, 1, 2],
    "study_hours": [5, 10, 15, 6, 12, 18, 9, 7, 20, 11],
    "learning_path": [
        "Web Development",
        "Web Development",
        "Data Science",
        "Data Science",
        "Cybersecurity",
        "Cybersecurity",
        "Data Science",
        "Cybersecurity",
        "Web Development",
        "Data Science"
    ]
}

df = pd.DataFrame(data)


# 2️⃣ Define Features & Label

X = df[["skill_level", "interest_area", "study_hours"]]
y = df["learning_path"]


# 3️⃣ Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4️⃣ Train Random Forest

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# 5️⃣ Evaluate Model

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# 6️⃣ Predict for New User

# skill_level → 1=Beginner, 2=Intermediate, 3=Advanced
# interest_area → 1=Programming, 2=Data, 3=Security

new_user = [[2, 2, 12]]
prediction = model.predict(new_user)

print("Recommended Learning Path:", prediction[0])