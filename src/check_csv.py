import pandas as pd

df = pd.read_csv("data_raw/train.csv")
print("Columns in CSV:")
print(df.columns)
print(df.head())
