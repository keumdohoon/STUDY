print(samsung)
print(hite.head())
#print(samsung.shape)
#print(hite_shape)



#Non 제거 1
samsung = samsung.dropna(axis=0)
#print(samsung)
#print(samsung.shape)

hite = hite.fillna(method='bfill')
hite = hite.dropna(axis=0)
#bfill 은 데이터의 전의 값을 가져와서 비어있는 셀에 집어 넣어 준다는 것이다. 



#Non 제거
hite = hite[0:509]
hite.iloc[0,1:5] = [10, 20, 30, 40]
#iloc는 0행의 1~5컬럼에 각각 10, 20, 30,40 을 적용시켜준다는 것이다. 

hite.loc["2020-06-02", '고가':'거래량']= ['10', '20', '30', '40']
#행은 날짜를 서주고 열은 고가부터 거래량까지의 데이터슷 쓴다는 것이다. 
#고가부터 거래량까지 가각의 항목에 10, 20, 30, 40을 각각 자리에 넣어준다. 

print(hite)
