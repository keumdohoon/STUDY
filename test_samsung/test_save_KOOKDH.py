import numpy as np
import pandas as pd

datasets1 = pd.read_csv("./data/csv/하이트 주가.csv", 
                        index_col = 0, 
                        header = 0,encoding = 'cp949', sep =',')

datasets2 = pd.read_csv("./data/csv/삼성전자 주가.csv", 
                        index_col = 0, 
                        header = 0,encoding = 'cp949', sep =',')

print(datasets1.head(508))       

print(datasets1.shape) 
print(datasets2)       
print(datasets2.shape)        

datasets1 = datasets1.dropna()
datasets2 = datasets2.dropna()

print(datasets1)       
print(datasets1.shape) 
print(datasets2)       
print(datasets2.shape)  # (509, 1)

# datasets2 = pd.DataFrame(datasets2,index=[i for i in range(508)], columns=['날짜','시가'])
# datasets2 = pd.DataFrame(datasets2, index = []) # 이것 때문에 삼성전자 주가가 2차원으로 변경된듯?
# print(datasets2) # Empty DataFrame 
# print(datasets2.shape) # (0, 1)


# datasets2.drop(['일자']['2020-06-02'])
# datasets2 = datasets2.drop(datasets2.index[0])
print(type(datasets2)) # <class 'pandas.core.frame.DataFrame'>
datasets2 = datasets2.drop(['2020-06-02'])

print(datasets2)
print(datasets2.shape)


#거래량str->int 변경
#hite
print(datasets1.index) # 날짜 나옴
print(datasets1.iloc) # iloc는 판다스 데이터 프레임 구조에서 위치정수를 기반으로 인덱싱을 한다.

for i in range(len(datasets1.index)): # 행 만큼 반복 508-1 총  0 ~ 507
    datasets1.iloc[i,0:5] = datasets1.iloc[i,0:5].replace(',','') # i행의 1열
print(datasets1)
print(datasets1.shape)


#samsung
for i in range(len(datasets2.index)): 
    for j in range(len(datasets2.iloc[i])):
        datasets2.iloc[i,j] = int(datasets2.iloc[i,j].replace(',',''))

# datasets1=np.transpose(datasets1)

print(datasets1)
print(datasets2)

datasets1 = datasets1.sort_values(['일자'], ascending=[True])
datasets2 = datasets2.sort_values(['일자'], ascending=[True])
print('dataset1:', datasets1)
print('dataset2:', datasets2)


#일자순으로 깔끔하게 나오게 분류해준다.

print(datasets1.values)
datasets2 = datasets2.values
#pandas를 numpy로 바꿔준다

print('======================================')
print('x',datasets1)
print('x',datasets2)



print(datasets1.shape)  #(426, 5)(508, 5)
print(datasets2.shape)  #(426, 5)(508, 1)
print(type(datasets1), type(datasets2))
#<class 'numpy.ndarray'> <class 'numpy.ndarray'>로 우리가 원하는 방향으로 변경된걸 알 수 있다.



# datasets1=np.transpose(datasets1)
# print(datasets1)
# print(datasets1.shape)
# print(len(datasets1), '시가')

# datasets1 = pd.datasets1({'시가': ['1', '2', '3'], 

#                    '고가': ['4.1', '5.5', '6.0'], '저가': ['4.1', '5.5', '6.0'], '종가': ['4.1', '5.5', '6.0'], '거래량': ['4.1', '5.5', '6.0']}) 



# print(datasets1)


# from pandas import DataFrame
# import numpy as np
 
# data = [
    
#     ]
 
# columns = ["시가", "고가", "저가", "종가", "거래량"]
# datasets1 = DataFrame(data=data, columns=columns)
# print(datasets1)


def remove_comma(x):
         return x.replace(',', '')
datasets1['시가'] = datasets1['시가'].apply(remove_comma)
datasets1['저가'] = datasets1['저가'].apply(remove_comma)
datasets1['종가'] = datasets1['종가'].apply(remove_comma)
datasets1['거래량'] = datasets1['거래량'].apply(remove_comma)
print(datasets1)
print(datasets1.dtypes)

datasets1 = datasets1.astype({'시가': np.int64})
print(datasets1)
print(datasets1.dtypes)
datasets1 = datasets1.astype({'고가':np.int64})
datasets1 = datasets1.astype({'저가': np.int64})
datasets1 = datasets1.astype({'종가': np.int64})
datasets1 = datasets1.astype({'거래량': np.int64})
print(datasets1)
print(datasets1.dtypes)


print('x',datasets1)
print('x',datasets2)

#데이터를 넘파이로 저장하기
np.save('./data/hite.npy', arr = datasets1) 
np.save('./data/samsung_electronics.npy', arr = datasets2) 
print(datasets1)
print(datasets2)

#실행시키면 저장까지 다 한것이다.
'''