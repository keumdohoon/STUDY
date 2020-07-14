import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_absolute_error as MAE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from lightgbm import LGBMRegressor
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)

x_train = np.load('./dacon/comp1/x_train.npy')
y_train = np.load('./dacon/comp1/y_train.npy')
x_pred = np.load('./dacon/comp1/x_pred.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 66
)
print(x_train.shape)
print(y_train.shape)
# print(x_test.shape)

# 2. model
final_y_test_pred = []
final_y_pred = []
parameter = [
    {'n_estimators': [1],
    'learning_rate': [0.05,0.06,0.07,0.08,0.09],
    'max_depth': [6], 
    'boosting_type': ['dart'], 
    'drop_rate' : [0.3],
    'objective': ['regression'], 
    'metric': ['logloss','mae'], 
    'is_training_metric': [True], 
    'num_leaves': [144], 
    'colsample_bytree': [0.7], 
    'subsample': [0.7]
    }
]

settings = {
    'verbose': False,
    'eval_set' : [(x_train, y_train), (x_test,y_test)]
}

kfold = KFold(n_splits=5, shuffle=True, random_state=66)
# 모델 컬럼별 4번
for i in range(4):
    model = LGBMRegressor()
    settings['eval_set'] = [(x_train, y_train[:,i]), (x_test,y_test[:,i])]
    model.fit(x_train,y_train[:,i], **settings)
    y_test_pred = model.predict(x_test)
    score = model.score(x_test,y_test[:,i])
    mae = MAE(y_test[:,i], y_test_pred)
    print("r2 : ", score)
    print("mae :", mae)
    thresholds = np.sort(model.feature_importances_)[ [i for i in range(0,len(model.feature_importances_), 20)] ]
    print("model.feature_importances_ : ", model.feature_importances_)
    print(thresholds)
    best_mae = mae
    best_model = model
    best_y_pred = model.predict(x_pred)
    best_y_test_pred = y_test_pred
    print(best_y_pred.shape)
    for thresh in thresholds:
        if(thresh == 0): continue
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median 이 둘중 하나 쓰는거 이해하면 사용 가능
                                                ## 이거 주어준 값 이하의 중요도를 가진 feature를 전부 자르는 파라미터
        select_x_train = selection.transform(x_train)
        select_x_test = selection.transform(x_test)
        select_x_pred = selection.transform(x_pred)

        print(select_x_train.shape)

        selection_model = RandomizedSearchCV(LGBMRegressor(), parameter, cv = kfold,n_iter=4)
        settings['eval_set'] = [(select_x_train, y_train[:,i]), (select_x_test,y_test[:,i])]
        selection_model.fit(select_x_train, y_train[:,i], **settings)

        y_pred = selection_model.predict(select_x_test)
        r2 = r2_score(y_test[:,i],y_pred)
        mae = MAE(y_test[:,i],y_pred)
        print(selection_model.best_params_)
        if mae <= best_mae:
            print("예아~")
            best_mae = mae
            best_model = selection_model
            best_y_pred = selection_model.predict(select_x_pred)
            best_y_test_pred = y_pred
        print("Thresh=%.3f, n=%d, MAE: %.5f R2: %.2f%%" %(thresh, select_x_train.shape[1], mae, r2*100))
    final_y_pred.append(best_y_pred)
    final_y_test_pred.append(best_y_test_pred)

print('MAE :', MAE(y_test, np.array(final_y_test_pred).T))

final_y_pred = np.array(final_y_pred)

submissions = pd.DataFrame({
    "id": np.array(range(10000,20000)),
    "hhb": y_pred[:, 0],
    "hbo2": y_pred[:, 1],
    "ca": y_pred[:, 2],
    "na": y_pred[:, 3]
})
print(submissions)
submissions.to_csv('./Dacon/comp1/comp1_sub.csv', index = False)