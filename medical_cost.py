import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# read the file
df = pd.read_csv(r"C:\Users\ADMIN\Downloads\archive (7)\insurance.csv")
print(df.columns)

# select feature and target variable
X = df[["age", "bmi", "children"]]
y = df["charges"]

# split data into train adn test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# prediction 
y_pred = model.predict(X_test)


#accuracy
acc = model.score(X_test, y_test)
print("Model Accuracy:", acc)

# Evaluate model Performance
mse = mean_squared_error(y_test , y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)           
print("R-squared Score:", r2)

# Predict for new data
age, bmi, children = map(float, input("Enter age, bmi and number of children separated by commas: ").split(","))
new_data = [[age, bmi, children]]           
predicted_charges = model.predict(new_data)
print("Predicted Medical Charges:", predicted_charges[0])