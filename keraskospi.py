#코스피 파일을 불러온다.

import numpy as np
import pandas as pd

datasets = pd.read_csv("./data/csv/kospi200.csv", 
                        index_col = 0, 
                        header = 0, encoding = 'cp949',sep = ',')

print(datasets)       
print(datasets.shape)                 