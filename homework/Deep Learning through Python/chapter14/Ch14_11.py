import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA 

sample_data_frame

#nan 이 포함된 데이터가 출력된다. 


#      0	        1           2	        3
# 0	0.841679	0.940392	0.124771	0.270782
# 1	NaN	        0.257470	0.800159	0.770554
# 2	0.228424	0.357201	NaN	        0.190702
# 3	0.305640	0.358955	0.236532	0.681229
# 4	0.487733	0.725023	0.560171	0.311290
# 5	0.723736	0.711835	0.084667	NaN
# 6	0.704934	0.876740	0.042586	NaN
# 7	0.624276	0.985274	0.327018	NaN
# 8	0.696359	0.276943	0.311615	NaN
# 9	0.545223	0.092706	0.100455	NaN