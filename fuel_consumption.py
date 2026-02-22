#import Libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# upload and read dataset 
url = r"C:\Users\ADMIN\Downloads\FuelConsumptionCo2.csv"
df = pd.read_csv(url)