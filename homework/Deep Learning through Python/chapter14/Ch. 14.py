#14-1
import pandas as pd

df = pd.read_csv("./data/csv/wine_data.csv", header=None, index_col=None)

# df.columns = ["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash","Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "0D280/0D315 of diluted wines", "Proline" ]

print(df.shape)