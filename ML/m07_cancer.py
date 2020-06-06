#cancer(이진)데이터로 오늘 배운 SVC, linearSVC, KNeighborsClassifier, KNeighborsRegressor
#중 하나를 사용하여 만든다.



from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_breast_cancer
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_breast_cancer()
print("data: ",  dataset.data)
print('target: ', dataset.target)
x = dataset.data
y = dataset.target
print(x.shape) # (569, 30)
print(y.shape) # (569,)
print(x)
print(y)



print(type(dataset))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True,
    train_size = 0.8)
# y= np_utils.to_categorical(y)

scale = StandardScaler()
x = scale.fit_transform(x)
#2. 모델
# model = LinearSVC()    #acc =  0.903508, R2 : 0.5810223855
# model = SVC()    #acc =  0.894736, R2 : 0.54293
# model = KNeighborsClassifier()   #acc = 0.9210526, R2 : 0.65720
# model = KNeighborsRegressor()
model = RandomForestClassifier()  #acc =  0.9561403 ,R2 :  0.8095556
# model = RandomForestRegressor()

#이렇게 여러가지 결과치가 나오긴하는데 우리는 이중에서 어느결과치를 사용할지를 잘 고민해야한다. 

#3. 실행
model.fit(x_train,y_train)
score = model.score(x_test, y_test)

#4, 평가와 예측
y_pred = model.predict(x_test)
print("x_test : \n",x_test,"\npred values : \n",y_pred)


acc = accuracy_score(y_test, y_pred)
print(x_test, "의 예측 결과 :", y_pred)

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)


print("R2 : ", r2)
print("acc : ",acc)
print('score: ', score)
