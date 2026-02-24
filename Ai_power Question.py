import pandas as pd 
import random

# dataset create

data ={
    "category": [
        "Python", "Python", "Python",
        "Machine Learning", "Machine Learning", "Machine Learning",
        "Data Science", "Data Science", "Data Science"
    ],
    "difficulty": [
        "Easy", "Medium", "Hard",
        "Easy", "Medium", "Hard",
        "Easy", "Medium", "Hard"
    ],
    "question": [
        "What is a list in Python?",
        "Explain the difference between deep copy and shallow copy.",
        "Explain how Python's memory management works.",
        "What is supervised learning?",
        "Explain bias-variance tradeoff.",
        "Explain how gradient descent works mathematically.",
        "What is data preprocessing?",
        "Explain feature engineering techniques.",
        "How do you handle imbalanced datasets?"
    ]}

df = pd.DataFrame(data)



# Function to get question based on category and difficulty
def Questiom_maker(Category_1, Difficulty_1):
    filtered_df = df[(df["category"].str.lower()) == (Category_1.lower()) &
                      (df["difficulty"].str.lower() == (Difficulty_1.lower()))]
    
    if filtered_df.empty:
        return "No question found for the given category and difficulty."
    
    return random.choice(filtered_df["question"].tolist())



print("Welcome to Interview")
category = input("Choose a category (Python, Machine Learning, Data Science): ")
difficulty = input("Choose a difficulty level (Easy, Medium, Hard): ")

question = Questiom_maker(category, difficulty)
print(question)