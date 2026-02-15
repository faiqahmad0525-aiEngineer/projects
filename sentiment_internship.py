import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. Sample Dataset (Feedback)

data = {
    "feedback": [
        "The internship was amazing and I learned a lot",
        "Great mentors and supportive environment",
        "It was a terrible experience and very stressful",
        "I gained valuable skills and confidence",
        "Workload was too high and poorly managed",
        "Fantastic learning opportunity",
        "No proper guidance and very disappointing",
        "Really enjoyed working with the team"
    ],
    "sentiment": [
        "Positive",
        "Positive",
        "Negative",
        "Positive",
        "Negative",
        "Positive",
        "Negative",
        "Positive"
    ]
}

df = pd.DataFrame(data)


# 2. Features & Labels

X = df["feedback"]
y = df["sentiment"]


# 3. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Convert Text → Numbers

vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


# 5. Train Model

model = LogisticRegression()
model.fit(X_train_vectors, y_train)



# 6. Predictions

y_pred = model.predict(X_test_vectors)


# 7. Accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


# 8. Test New Feedback

new_feedback = ["The training was very helpful and enjoyable"]
new_feedback_vector = vectorizer.transform(new_feedback)

prediction = model.predict(new_feedback_vector)
print("Predicted Sentiment:", prediction[0])
