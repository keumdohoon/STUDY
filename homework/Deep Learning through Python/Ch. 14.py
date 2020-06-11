#14-1
import pandas as pd
df = pd.read_csv("https://archive.ics.uci,edu/ml/machine-learning-databases/wine/wine.data", header=None)

df.columns = ["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash","Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "0D280/0D315 of diluted wines", "Proline" ]

print(df)