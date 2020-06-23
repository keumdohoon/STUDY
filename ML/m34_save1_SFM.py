'''
m29_eval1_SFM.py
m29_eval2_SFM.py
m29_eval3_SFM.py 에 save를 적용하시오.

save이름에는 평가지표를 첨가해서
가장 좋은 SFM용 save파일을 나오도록 하는거ㅏ

'''


#m24_SelectFromModel1.boston 에서 Gridsearch를 사용해준 모델
#XG + select from model + gridsearch
#책에 없는 내용이니 주의를 매우 기우릴것
from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

# datasets = load_boston()
# x = dataset.data
# y = dataset.target

datasets = load_boston()

##데이터###
x, y = load_boston(return_X_y=True)# 이렇게 적어줘도됨

print(x.shape) #(506, 13)
print(y.shape) #(506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
                                                    shuffle = True, random_state = 66)


###기본 모델###
model = XGBRegressor(n_estimators = 100, learning_rate = 0.1, n_jobs = -1)


model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2: ', score) #R2:  0.9323782715918234
'''
###feature engineerg###
thresholds = np.sort(model.feature_importances_)
print('threshold:',thresholds)
 #   [0.00165952 0.00253551 0.01019184 0.01064387 0.01437491 0.01552134
 #   0.01602842 0.02064591 0.0255824  0.04278987 0.04454341 0.25299022
 #   0.54249275]


models = []
res = np.array([])

for thresh in thresholds:      
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                            #median
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test) 
    model2 = XGBRegressor(n_estomators=100, learning_rate = 0.1, n_jobs = -1)
    model2.fit(select_x_train, y_train, verbose = False, eval_metric=['logloss', 'rmse'], 
            eval_set = [(select_x_train, y_train), (select_x_test, y_test)], early_stopping_rounds = 20)

    y_pred = model2.predict(select_x_test)
    score = r2_score(y_test, y_pred)
    shape = select_x_train.shape
    models.append(model2)               #모델을 배열에 저장

    print(thresh, score)
    res = np.append(res,score)      #결과값 전부를 배열에 저장
print(res.shape)
best_idx = res.argmax()                     #결과값 최대값의 index wjwkd
score = res[best_idx]                       #위 인덱스 기반으로 점수 호출
total_col = x_train.shape[1] - best_idx   #전체 컬럼 계산
models[best_idx].save_model(f'./model/xgb_save/boston--{score}--{total_col}--.model') #인덱스 기반으로 모델 저장
'''