from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datestrs = ['6/1/2020', '6/3/2020', '6,4,2020', '6/8/2020', '6/10/2020']
dates = pd.to_datetime(datestrs)
print(dates)
print("===============================")

ts = Series([1,np.nan, np.nan, 8,10], index = dates)
print(ts)

ts_intp_linear = ts.interpolate()
print(ts_intp_linear)
#보간법 중에 선형보간이다. 그래서 이거는 기존의 데이터를 가지고 비어있는 것들을 태워준다는 것이다.  