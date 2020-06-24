#책에 없는 내용이니 주의를 매우 기우릴것
from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, r2_score
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
# datasets = load_boston()
# x = dataset.data
# y = dataset.target

datasets = load_iris()


x, y = load_iris(return_X_y=True)# 이렇게 적어줘도됨

print(x.shape) #(150, 4)
print(y.shape) #(150,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
                                                    shuffle = True, random_state = 66)



#모델정의
model = LGBMClassifier(objective = 'multi:softmax',n_estimators = 300, learning_rate = 0.1, n_jobs= 1)


model.fit(x_train, y_train)
score = model.score(x_test, y_test)    
print('acc :', score)      #acc : 0.9333333333333333  
#rmse, mae, logloss, error, auc
#error은 회기 모델 지표가 아니다
#eval metric을 두가지 이상으로 할때는 리스트 형식으로 쓴다. 

print("r2 Score : %.2f%%" %(score * 100.0))     #r2 Score : 93.33%

# Stopping. Best iteration:
# [28]    validation_0-rmse:0.06268       validation_1-rmse:0.28525
#validation 이 올라가기 시작하면서 끊겼다. loss랑 validation 중에 중요한것은  validation 이다. 

import matplotlib.pyplot as plt





thresholds = np.sort(model.feature_importances_)
             #정렬 #중요도가 낮은 것부터 높은것 까지

print(thresholds)  #[574 594 634 915]

#for문을 전체 컬럼수 만큼 돌리면 총 13번 돌리게 된다. 

for thresh in thresholds:     #총 컬럼수 만큼 돌게 된다 빙글빙글. 
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                            #median
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    print(select_x_train.shape) #결과를 보면 컬럼이 13->1로 쭈욱 내려가는데 이것을 컬럼의 중요도가 없는 컬럼을 하나씩 지워주는 것이다.최종 1 

    
    selection_model = LGBMClassifier(objective='multi:softmax', n_estimators=300, learning_rate=0.1, n_jobs= -1)
    selection_model.fit(select_x_train, y_train, verbose = False, eval_metric = ['multi_logloss', 'multi_error'],
          eval_set = [(select_x_train, y_train), (select_x_test, y_test)], early_stopping_rounds=20)



    y_predict = selection_model.predict(select_x_test)

    acc = accuracy_score(y_test, y_predict)

    print("Thresh= %.3f,n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
                acc*100.0))
    

# import pickle#파이썬에서 제공하는 피클
# pickle.dump(model, open("./model/xgb_save/iris.pickle.data", "wb"))
# print("SAVED!!!!")
# fig, ax = plt.subplots()
# ax.plot(x_axis, result['validation_0']['logloss'], label = 'Train')
# ax.plot(x_axis, result['validation_1']['logloss'], label = 'Test')
# ax.legend()
# plt.ylabel('Log Loss')
# plt.title('XGBoost Log Loss')
# plt.show()


# fig, ax = plt.subplots()
# ax.plot(x_axis, result['validation_0']['rmse'], label = 'Train')
# ax.plot(x_axis, result['validation_1']['rmse'], label = 'Test')
# ax.legend()
# plt.ylabel('Rmse')
# plt.title('XGBoost Rmse')
# plt.show()
