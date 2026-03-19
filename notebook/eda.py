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