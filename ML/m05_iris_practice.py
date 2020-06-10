from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
#1. Data 
dataset = load_iris()
print('data: ', dataset.data)
print('target: ', dataset.target)
x = dataset.data
y = dataset.target
print(x.shape)
print(y.shape)
print(x)
print(y)
#데이터를 불러와 준다. 엑스를 데이터에 와이를 타겟으로해서 설정해준다. 

y = np_utils.to_categorical(y)
x_train, x_test,  y_train, y_test = train_test_split(
    x, y, shuffle= True, random_state = 66, train_size = 0.5)
scale = StandardScaler()
x = scale.fit_transform(x)
#와이를 카테고리컬로해서 원핫 인코딩을 해준다.
# 스텐다드 스케일러로 인해서 스케일 핏을 해준다. 
#  

# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier(n_neighbors=1)
# model = KNeighborsRegressor()
model = RandomForestClassifier()
# model = RandomForestRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
#모델을 랜덤 포레스트 클레시파이어로 설정해주고 모델 핏을 통해서 엑스트레인과 와이 트레인 을 핏해준다.
# 모델 스코어를 통해서 엑스 테스트와 와이테스트를 스코어를 구해준다. 


#3. activate
from sklearn.metrics import accuracy_score, r2_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

#4. Evaluation 
print(x_test, "의 예측 결과 :", y_predict)
print('r2 :', r2)
print("acc = ", acc)
print("score : ", score)




#ML 에서는 score이라는 함수를 써주면 자기가 자동으로 결과치가 acc나r2둘중에 선형인지 분류형인지를 알아서 계산해주고 이를 제일 잘 맞는 결과치를 
#내어준다. 여기서 우리가 R2와 ACC를 찍어주고 Score도 해 준이유는 3개의 결과치를 비교하여 Score가 R2나 ACC중에 어떠한 결과치를 내어주게 되었는지를 비교해주기 위함이다. 