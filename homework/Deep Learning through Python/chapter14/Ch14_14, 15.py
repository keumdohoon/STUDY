# 전체 데이테 프레임에서 출력하되 0열과 2열을 남기고 nan을 포함하는 행은 삭제하고 출력하라

import numpy as np
from numpy import nan as NA
import pandas as pd
np.random.seed(0)

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA


#이렇게 작성하면 전체 데이터프레임에서 0열과 2열 만을 프린트하고 nan이 포함된 행은 다 지워진체로 출력된다.ㅜ 
sample_data_frame[[0, 2]].dropna()