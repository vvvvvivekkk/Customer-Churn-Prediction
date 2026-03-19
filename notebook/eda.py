import pandas as pd

df = pd.read_csv(r"C:\Users\11020\Desktop\churn-prediction\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nInfo:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())



print(df["Churn"].value_counts())


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="Churn", data=df)
plt.show()


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print(df.isnull().sum())

df.dropna(inplace=True)


df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


df = pd.get_dummies(df, drop_first=True)

print(df.head())
print("Shape after cleaning:", df.shape)

import os

output_path = "../data/cleaned_churn.csv"

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df.to_csv(output_path, index=False)

df.to_csv("C:/Users/11020/Desktop/churn-prediction/data/cleaned_churn.csv", index=False)