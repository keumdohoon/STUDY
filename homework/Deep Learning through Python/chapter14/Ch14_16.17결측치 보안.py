#데이터에서 일부러 결측치를 만들고 그것을 0으로 채워지는 결측치
#보안법
import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

#데이터를 누락시킨다. 
sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA

sample_data_frame.fillna(0)
#이렇게 적게 되면 nan값을 0으로 채워주라는 뜻으로 결측치가 
#제거가 되는데 모든 ㄱㄹ측치를 0으로 채워지=ㅜ는 것은 그다지 현명한 방법이 아니다. 