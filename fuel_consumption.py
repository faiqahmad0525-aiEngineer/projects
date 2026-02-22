#import Libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

# upload and read dataset 
url = r"C:\Users\ADMIN\Downloads\FuelConsumptionCo2.csv"
df = pd.read_csv(url)

# select features and target variable
X = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY"]]
y = df["CO2EMISSIONS"]

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#train linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# evaluate model peroformance
