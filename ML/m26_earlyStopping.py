#xgbooster에도 evaluation이 있다. 
from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, r2_score

# datasets = load_boston()
# x = dataset.data
# y = dataset.target

datasets = load_iris()


x, y = load_iris(return_X_y=True)# 이렇게 적어줘도됨

print(x.shape) #(569, 30)
print(y.shape) #(569,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBRegressor(n_estimators = 1000, learning_rate = 0.1)

model.fit(x_train, y_train, verbose=True, eval_metric="rmse",
 eval_set=[(x_train, y_train), (x_test, y_test)],
            early_stopping_rounds=20)
#rmse, mae, logloss, error, auc
#error은 회기 모델 지표가 아니다
result = model.evals_result()
print("eval's results :", result)


y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print("r2 Score : %.2f%%" %(r2 * 100.0))
print("r2 :", r2)
# Stopping. Best iteration:
# [28]    validation_0-rmse:0.06268       validation_1-rmse:0.28525
#validation 이 올라가기 시작하면서 끊겼다. loss랑 validation 중에 중요한것은  validation 이다. 