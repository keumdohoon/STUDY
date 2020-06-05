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



y = np_utils.to_categorical(y)

x_train, x_test,  y_train, y_test = train_test_split(
    x, y, shuffle= True, random_state = 66, train_size = 0.5)

scale = StandardScaler()
x = scale.fit_transform(x)


# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier(n_neighbors=1)
# model = KNeighborsRegressor()
model = RandomForestClassifier()
# model = RandomForestRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)


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
