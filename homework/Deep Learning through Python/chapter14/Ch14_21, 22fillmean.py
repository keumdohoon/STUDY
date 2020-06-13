#fillna 에서 sample_data_frame.fillna(sample_data_frame.mean())
#이ㅓㅀ게 파일명과 뒤에 점을 붙여주고 난다음에 mean을 붙여주게 되면 nan위아래에 있는 평균값으로 nan을 채워주게 된다. 
import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

# 데이터 누락
sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA

#fillna로 NaN 부분에 평균값을 대입합니다
sample_data_frame.fillna(sample_data_frame.mean())
