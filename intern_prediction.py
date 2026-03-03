import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1. Create Sample Dataset

data = {
    "hours_studied": [2, 5, 1, 8, 7, 3, 6, 4],
    "projects_completed": [1, 3, 0, 5, 4, 1, 4, 2],
    "attendance": [60, 80, 50, 95, 90, 70, 85, 75],
    "communication_score": [4, 7, 3, 9, 8, 5, 8, 6],
    "performance": [0, 1, 0, 2, 2, 1, 2, 1]
}

df = pd.DataFrame(data)


# 2. Split Features & Label

X = df.drop("performance", axis=1)
y = df["performance"]


# 3. Train / Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Train Random Forest Model


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



# 5. Predictions


y_pred = model.predict(X_test)



# 6. Accuracy


accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)



# 7. Predict New Intern


new_intern = [[6, 3, 85, 7]]

prediction = model.predict(new_intern)
print("Predicted Performance:", prediction[0])
