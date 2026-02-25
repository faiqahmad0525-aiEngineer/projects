import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# import dataset
url = r"C:\Users\ADMIN\Downloads\archive (6)\data.csv"
df = pd.read_csv(url)

# print head and tail of data
print(df.head())
print(df.tail())
print(df.columns)

# select features and target variable
x = df[["bedrooms", "bathrooms", "sqft_living", "sqft_lot"]]
y = df["price"]

#split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#train linear regression model
model = LinearRegression()
model.fit(x_train, y_train)


#Prediction
y_pred = model.predict(x_test)

#new data for prediction
bedrooms, bathrooms, sqft_living, sqft_lot = map(float, input("Enter number of bedrooms, bathrooms, sqft_living and sqft_lot separated by commas: ").split(","))
new_data = [[bedrooms, bathrooms, sqft_living, sqft_lot]]
predicted_price = model.predict(new_data)
print("Predicted House Price:", predicted_price[0])

# accuracy
accuracy = model.score(x_test, y_test)
print("Model Accuracy:", accuracy)

#Evaluate model performance

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)