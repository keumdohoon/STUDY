from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score

# datasets = load_boston()
# x = dataset.data
# y = dataset.target

datasets = load_breast_cancer()


x, y = load_breast_cancer(return_X_y=True)# 이렇게 적어줘도됨

print(x.shape) #(569, 30)
print(y.shape) #(569,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
                  shuffle = True, random_state = 66)

model = XGBClassifier(n_estimators = 400, learning_rate = 0.1)

model.fit(x_train, y_train, verbose=True, eval_metric="error",
 eval_set=[(x_train, y_train), (x_test, y_test)])




import pickle
model2= pickle.load(open("./model/xgb_save/cancer.pickle.data", "rb"))
print("LOADED!!!!불러왔다. ")
y_predict = model.predict(x_test)
acc = accuracy_score(y_predict, y_test)
print("acc : ", acc)