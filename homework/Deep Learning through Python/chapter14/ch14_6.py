# 14_6 pandas 로 csv파일 만들기
import pandas as pd

data = {"city": ["Nagano", "Sydney", "Salt lake city", "Athens", "Torino", "Beijing", "Vancouver", "London", "Sochi", "Rio de Janeiro"],
        "year" : [1998, 2000, 2002, 2004, 2006,2008,2010,2012,2014,2016],
        "season": ["winter", "summer", "winter", "summer", "winter","winter", "summer", "winter","summer", "winter"]}

df= pd.DataFrame(data)
df.to_csv("csv1.csv")


#실행을 시키면 같ㅇ느 폴더에 csv파일이 만들어 진다. 