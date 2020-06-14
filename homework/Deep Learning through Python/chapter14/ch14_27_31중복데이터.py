#중복데이터의 예, 중복데이터를 삭제하는 방법
#밑에는 중복데이터 작성
import pandas as pd
from pandas import DataFrame

dupli_data = DataFrame({"col1":[1, 1, 2, 3, 4, 4, 6, 6], 
                        "col2":["a", "b", "b", "b", "c", "c", "b", "b"]})

dupli_data 

# col1	col2
# 0	1	a
# 1	1	b
# 2	2	b
# 3	3	b
# 4	4	c
# 5	4	c
# 6	6	b
# 7	6	b

#여기서는 만약 행의 중복된 데이터가 나타나면 그 행을 true로 표시해준다. 
#위의 데이터에서는 4,5행과 67행이 중복된다. 

dupli_data.duplicated()
# 0    False
# 1    False
# 2    False
# 3    False
# 4    False
# 5     True
# 6    False
# 7     True
# dtype: bool
#이렇게 중복된 데이터를 true로 표시해준다. 
dupli_data.drop_duplicates()
#얘를 써주면 중복된 데이터를 지워주는 역할을한다.
# 5번과 7번 행이 없어진것을 확인할 수 있다. 
# col1	col2
# 0	1	a
# 1	1	b
# 2	2	b
# 3	3	b
# 4	4	c
# 6	6	b 

#연습문제
import pandas as pd
from pandas import DataFrame

dupli_data = DataFrame({"col1":[1, 1, 2, 3, 4, 4, 6, 6, 7, 7, 7, 8, 9, 9],
                        "col2":["a", "b", "b", "b", "c", "c", "b", "b", "d", "d", "c", "b", "c", "c"]})


dupli_data.drop_duplicates()
#   col1	col2
# 0	  1	     a
# 1	  1     	b
# 2	  2  	b
# 3   3	    b
# 4	  4	    c
# 6	  6  	b
# 8	  7  	d
# 10	7	c
# 11	8	b
# 12	9	c
#중복된 데이터를 찾아서 지우시오
#5번과 7번과 9번행이 없어졌다. 