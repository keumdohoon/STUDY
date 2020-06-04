import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from keras.models import Model
from keras.layers import Dense, Input, Concatenate
from sklearn.metrics import r2_score, mean_squared_error as mse
from keras.callbacks import EarlyStopping

for epochs in range(20, 220, 5):
    hite_df=pd.read_csv("./하이트 주가.csv", index_col=0, header=0, encoding= "cp949", sep= ",")
    samsung_df= pd.read_csv("./삼성전자 주가.csv", index_col=0, header=0, encoding="cp949", sep="," )

    hite_df = hite_df[:509]
    samsung_df = samsung_df[:509]

    # print(hite_df.tail())
    # print(samsung_df.tail())

    # print(hite_df.head())
    # print(samsung_df.head())

    hite_df = hite_df.sort_values(['일자'], ascending =[True])
    samsung_df = samsung_df.sort_values(['일자'], ascending=[True])

    #print(hite_df.tail)
    #print(samsung_df.tail())

    print(f"")
