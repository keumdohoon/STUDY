
from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
# datasets = load_boston()
# x = dataset.data
# y = dataset.target

datasets = load_breast_cancer()


x, y = load_breast_cancer(return_X_y=True)# 이렇게 적어줘도됨

print(x.shape) #(569, 30)
print(y.shape) #(569,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = LGBMClassifier(n_estimators= 300, learning_rate = 0.1, n_jobs = -1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("acc: ", score) #acc:  0.9736842105263158


thresholds = np.sort(model.feature_importances_)
print(thresholds)  


for thresh in thresholds:     #총 컬럼수 만큼 돌게 된다 빙글빙글. 
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                            #median
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = LGBMClassifier(n_estimatore=300, learning_rate=0.1, n_jobs=-1)
    selection_model.fit(select_x_train, y_train, verbose=False, eval_metric='logloss', eval_set = [(select_x_train, y_train), (select_x_test, y_test)]
                        , early_stopping_rounds=20)


    print(select_x_train.shape) #결과를 보면 컬럼이 13->1로 쭈욱 내려가는데 이것을 컬럼의 중요도가 없는 컬럼을 하나씩 지워주는 것이다.최종 1 


    y_pred = selection_model.predict(select_x_test)

    acc = accuracy_score(y_test, y_pred)
    print("acc : ", acc)
    print("Thresh= %.3f,n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
    
# import pickle#파이썬에서 제공하는 피클
# pickle.dump(model, open("./model/xgb_save/breast_cancer.pickle.data", "wb"))
# print("SAVED!!!!")

from sklearn.metrics import accuracy_score, r2_score, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix

def get_clf_eval(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    print('오차행렬:\n', confusion)
    print('\n정확도: {:.4f}'.format(accuracy))
    print('정밀도: {:.4f}'.format(precision))
    print('재현율: {:.4f}'.format(recall))
    print('F1: {:.4f}'.format(F1))
    print('AUC: {:.4f}'.format(AUC))

get_clf_eval(y_test, y_pred)    