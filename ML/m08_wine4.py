import pandas as pd
import matplotlib.pyplot as plt
#y값에 대한 데이터 분포를 분류하는 방식
#너무 많은 분류가 있으니 오히려 힘들다. 
#와인 데이터 읽기 
wine = pd.read_csv("./data/csv/winequality-white.csv", sep=';', header=0)
#컬럼 안에 들어가 있는걸 그룹별로 모아주는 것

count_data = wine.groupby('quality')['quality'].count()
#1,2,3,4,5,6,7,8,9, 을 행별로 세겠다는 뜻이다 pandas groupby 검색해복시
print(count_data)

count_data.plot()
plt.show()

y= wine['quality']
x = wine.drop('quality', axis = 1)
#y레이블 축소
#와인이 현재는 3등급에서 9등급까지 맞출 확률이 90프로 였는데 우리는 지금 그 등급을 3등급으로 나누었다 왜냐하면 9등급하면 너무 등급이 많기에..
newlist = []
for i in list(y):
    if i <=4:
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else:
        newlist +=[2]
# 이런식으로 나누어주면 9등급이 3등급안으로 몰리게 된다. 
y = newlist

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x ,y,test_size = 0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print("acc_score :", accuracy_score)
print('acc: ', acc)