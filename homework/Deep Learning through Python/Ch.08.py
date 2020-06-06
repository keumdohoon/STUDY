'''
#import pandas
#8_1
import pandas as pd

#data
fruits = {"orange": 2, "banana":3}
print(pd.Series(fruits))

# orange    2
# banana    3
# dtype: int64

#8_2
import pandas as pd

data = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"], 
        "years"  : [2001, 2002, 2001, 2008, 2006], 
        "time"   : [1,4,5,6,3]}
df = pd.DataFrame(data)
print(df)    

#        fruits  years  time
# 0       apple   2001     1
# 1      orange   2002     4
# 2      banana   2001     5
# 3  strawberry   2008     6
# 4   kiwifruit   2006     3

#8_3
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10,5,8,12,3]
series = pd.Series(data, index = index)

data = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"], 
        "years"  : [2001, 2002, 2001, 2008, 2006], 
        "time"   : [1,4,5,6,3]}
df = pd.DataFrame(data)

print("Series 데이터")
print(series)
print("/n")
print("DataFrame 데이터")
print(df)


#8_4
# Series 데이터
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# dtype: int64
# /n
# DataFrame 데이터
#        fruits  years  time
# 0       apple   2001     1
# 1      orange   2002     4
# 2      banana   2001     5
# 3  strawberry   2008     6
# 4   kiwifruit   2006     3
#8_5
import pandas as pd
fruits ={"banana": 3, "orange":2}
print(pd.Series(fruits))
# banana    3
# orange    2
# dtype: int64
#8_6
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10,5,8,12,3]
print(series)

#8_7
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10,5,8,12,3]
series = pd.Series(data, index = index)
print(series)
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# dtype: int64
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# dtype: int64
#8_8
import pandas as pd
fruits = {"banana": 3, "orange": 4, "grape": 1, "peach":5}
series = pd.Series(fruits)
print(series[0:2])
# banana    3
# orange    4
# dtype: int64

#8_9
print(series[["orange", "peach"]])
# orange    4
# peach     5
# dtype: int64
#8_10, #8_11 QNA
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10,5,8,12,3]
series = pd.Series(data, index = index)
items1 = series[1:4]
items2 = series[["apple", "banana", "kiwifruit"]]
print(items1)
print()
print(items2)

# orange         5
# banana         8
# strawberry    12
# dtype: int64

# apple        10
# banana        8
# kiwifruit     3



#8_12, #8_13
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10,5,8,12,3]
series = pd.Series(data, index = index)

series_values = series.values

series_index = series.index

print(series_values)
print(series_index)


# [10  5  8 12  3]
# Index(['apple', 'orange', 'banana', 'strawberry', 'kiwifruit'], dtype='object')

#8_14
import pandas as pd
fruits = {"banana": 3, "orange": 2}
series = pd.Series(fruits)
series = series.append(pd.Series([3], index=["grape"]))
#인덱스가 파인애플이고 데이터가 12인 요소를 series에 추가 
#8_15
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10,5,8,12,3]
series = pd.Series(data, index = index)
print(series)

#8_16
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10,5,8,12,3]
series = pd.Series(data, index = index)
pineapple = pd.Series([12], index=["pineapple"])
series = series.append(pineapple)
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# dtype: int64
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# pineapple     12
# dtype: int64
print(series)
#8_17, #8_18
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10,5,8,12,3]
series = pd.Series(data, index = index)
series = series.drop("strawberry")
print(series)

# apple        10
# orange        5
# banana        8
# kiwifruit     3
# dtype: int64
#8_19
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10,5,8,12,3]
series = pd.Series(data, index = index)

conditions = [True, True, False, False, False]
print(series[conditions])
# apple     10
# orange     5
# dtype: int64

#8_20
import pandas as pd
print(series[series>=5])
# apple         10
# orange         5
# banana         8
# strawberry    12
# dtype: int64
#8_21, #8_22
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10,5,8,12,3]
series = pd.Series(data, index = index)
print(series[series>=5][series<10])
# orange    5
# banana    8
# dtype: int64


#8_23, #8_24
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10,5,8,12,3]
series = pd.Series(data, index = index)
items1 = series.sort_index()
items2 = series.sort_values()

print(items1)
print(items2)
# apple         10
# banana         8
# kiwifruit      3
# orange         5
# strawberry    12
# dtype: int64
# kiwifruit      3
# orange         5
# banana         8
# apple         10
# strawberry    12



#8_25
import pandas as pd
data = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"], 
        "years"  : [2001, 2002, 2001, 2008, 2006], 
        "time"   : [1,4,5,6,3]}
df = pd.DataFrame(data)
print(df)
#        fruits  years  time
# 0       apple   2001     1
# 1      orange   2002     4
# 2      banana   2001     5
# 3  strawberry   2008     6
# 4   kiwifruit   2006     3


#8_26, #8_27
import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10,5,8,12,3]
data2 = [30,25,12,10,8]
series1 = pd.Series(data1, index = index)
series2 = pd.Series(data2, index = index)
df = pd.Dataframe([series1, series2])

print(df)

#8_28, #8_29
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10,5,8,12,3]
data2 = [30,25,12,10,8]
series1 = pd.Series(data1, index = index)
series2 = pd.Series(data2, index = index)
df = pd.Dataframe([series1, series2])

df.index = [1,2]

#8_30 #행을 추가하는 얘
import pandas as pd
data = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"], 
        "time"   : [1,4,5,6,3]}
df = pd.DataFrame(data)
series = pd.Series(["mango", 2008, 7], index=["fruis", "year", "time"])
df = df.append(series, ignore_index=True)
print(df)

#8_31, #8_32
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10,5,8,12,3]
data2 = [30,25,12,10,8]
data3 = [30,12,10,8, 25, 3]

series1 = pd.Series(data1, index = index)
series2 = pd.Series(data2, index = index)
index.append("pineapple")
series3 = pd.Series(data3, index = index)
df = pd.Dataframe([series1, series2])
df = df.append(series3, ignore_index=True)
print(df)



#8_33
import pandas as pd
data = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"], 
        "years"  : [2001, 2002, 2001, 2008, 2006], 
        "time"   : [1,4,5,6,3]}
df = pd.DataFrame(data)

df["price"]= [150,120,100,300,150]
print(df)

#        fruits  years  time  price     
# 0       apple   2001     1    150     
# 1      orange   2002     4    120     
# 2      banana   2001     5    100     
# 3  strawberry   2008     6    300     
# 4   kiwifruit   2006     3    150     

#8_34 문제 Dataframe 생성 #8_35
import pandas as pd
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10,5,8,12,3]
data2 = [30,25,12,10,8]
series1 = pd.Series(data1, index = index)
series2 = pd.Series(data2, index = index)


new_column = pd.DataFrame([series1, series2])

df = pd.DataFrame ([series1, series2])

df["mango"] = new_column


'''
#8_36
import pandas as pd
data = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"], 
        "years"  : [2001, 2002, 2001, 2008, 2006], 
        "time"   : [1,4,5,6,3]}
df = pd.DataFrame(data)

print(df)

#8_37
df = df.loc[[1,2], ["time", "year"]]
print(df)

#        fruits  years  time
# 0       apple   2001     1
# 1      orange   2002     4
# 2      banana   2001     5
# 3  strawberry   2008     6
# 4   kiwifruit   2006     3

#8_38
import numpy as np
import pandas as pd
np.random.seed(0)
columns = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"]

#add data frames and add rows
df = pd.DataFrame()
for column in columns:
    df[column]= np.random.choice(range(1,11), 10)

#range (starting row, ending row-1)
df.index = range(1, 11)

print(df)



#8_39
import numpy as np
import pandas as pd
np.random.seed(0)
columns = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"]

#add data frames and add rows
df = pd.DataFrame()
for column in columns:
    df[column]= np.random.choice(range(1,11), 10)

#range (starting row, ending row-1)
df.index = range(1, 11)
df = df.loc[range(2,6), ["banana", "kiwifruit"]]

print(df)

#8_40
import pandas as pd
data = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"], 
        "years"  : [2001, 2002, 2001, 2008, 2006], 
        "time"   : [1,4,5,6,3]}
df = pd.DataFrame(data)
print(df)
#8_41
df = df.iloc[[1,3],[0,2]]
print(df)



#8_42, #8_43
print numpy as np
print pandas as pd 
np.random.seed(0)
columns = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df= pd.Dataframe()
for column in columns:
    df[column] = np.random.choice(range(1,11)), 10)
df.index = range(1,11)
df = df.iloc[range(1,5), [2,4]]

print(df)


#8_44
import pandas as pd
data = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"], 
        "years"  : [2001, 2002, 2001, 2008, 2006], 
        "time"   : [1,4,5,6,3]}
df = pd.DataFrame(data)

df_1 = df.drop(range(0,2))
df_2 = df.drop("year", axis=1)

print(df_1)
print()
print(df_2)



#8_45, #8_46
print numpy as np
print pandas as pd 
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df= pd.Dataframe()
for column in columns:
    df[column] = np.random.choice(range(1,11)), 10)
df.index = range(1,11)
df = df.drop(np.arange(2,11,2))
df= df.drop("strawberry", axis =1)

print(df)


#8_47
import pandas as pd
data = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"], 
        "years"  : [2001, 2002, 2001, 2008, 2006], 
        "time"   : [1,4,5,6,3]}
df = pd.DataFrame(data)
print(df)

#데이터를 오름차순으로 정렬시킨다 
df = df.sort_values(by="year", ascending = True)
print(df)

df = df.sort_values(by=["time", "year"], ascending = True)
print(df)

#8_48
print numpy as np
print pandas as pd 
np.random.seed(0)
columns =  ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1,11), 10)
df.index = range(1,11)

print(df)
#8_49
print numpy as np
print pandas as pd 
np.random.seed(0)
columns =  ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1,11), 10)
df.index = range(1,11)
df = df.sort_values(by=columns)

print(df)

#8_50
data = {"fruits" : ["apple", "orange", "banana", "strawberry", "kiwifruit"], 
        "years"  : [2001, 2002, 2001, 2008, 2006], 
        "time"   : [1,4,5,6,3]}
df = pd.DataFrame()

print(df.index % 2 ==0)
print()
print(df[df.index % 2 ==0])

#8_51
import numpy as np
import pandas as pd

np.random.seed(0)
columns =  ["apple", "orange", "banana", "strawberry", "kiwifruit"]
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1,11), 10)
df.index = range(1,11)

print(df)
#8_52
import numpy as np
import pandas as pd

np.random.seed(0)
columns =  ["apple", "orange", "banana", "strawberry", "kiwifruit"]
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1,11), 10)
df.index = range(1,11)
df = df.iloc[df["apple"]>=5]
df = df.iloc[df["kiwifruit"]>=5]

print(df)
#8_53
import pandas as pd 
import numpy as np

index = ["growth", "mission", "ishikawa", "pro"]
data = [50,7,26,1]

#write down the series
series = 
#input series to aidemy which is in alphabetical order
aidemy =

aidemy1 =
aidemy2= series.append(aidemy1)

print(aidemy)
print()
print(aidemy2)

#dataframe을 추가하고 열을 추가합니다. 
df = pd.Dataframe()
for index in index :
    df[index] = np.random.choice(range(1,11), 10)
    df.index = range(1,11)

    aidemy3 = 
    print()
    print(aidemy3)


#8_54
import pandas as pd 
import numpy as np

index = ["growth", "mission", "ishikawa", "pro"]
data = [50,7,26,1]

#write down the series
series = pd.Series(data, index=index)
#input series to aidemy which is in alphabetical order
aidemy =series.sort_index()

aidemy1 = pd.Series([30], index=["tutor"])
aidemy2= series.append(aidemy1)

print(aidemy)
print()
print(aidemy2)

#dataframe을 추가하고 열을 추가합니다. 
df = pd.Dataframe()
for index in index :
    df[index] = np.random.choice(range(1,11), 10)
    df.index = range(1,11)

    aidemy3 = df.loc[range(2,6), ["ishikawa"]]
    print()
    print(aidemy3)

