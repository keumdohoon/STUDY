#책에 없는 내용이니 주의를 매우 기우릴것
from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

# datasets = load_boston()
# x = dataset.data
# y = dataset.target

datasets = load_breast_cancer()


x, y = load_breast_cancer(return_X_y=True)# 이렇게 적어줘도됨

print(x.shape) #(569, 30)
print(y.shape) #(569,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBClassifier(n_jobs = -1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)



print("R2 :", score) #R2 : 0.9221188544655419

thresholds = np.sort(model.feature_importances_)
             #정렬 #중요도가 낮은 것부터 높은것 까지

print(thresholds)  
                   
for thresh in thresholds:     #총 컬럼수 만큼 돌게 된다 빙글빙글. 
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                            #median
    select_x_train = selection.transform(x_train)
    print(select_x_train.shape) #결과를 보면 컬럼이 13->1로 쭈욱 내려가는데 이것을 컬럼의 중요도가 없는 컬럼을 하나씩 지워주는 것이다.최종 1 

    parameters = [
    {"n_estimators":[1000], "learning_rate":[0.1],
     "max_depth":[5]}]
     
    selection_model = RandomizedSearchCV(XGBClassifier(n_jobs= -1), parameters , cv=5)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    x_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, x_pred)
    print("R2 : ", score)
    import pickle
    for i in thresholds:
        pickle.dump(model, open("./model/xgb_save/breast_cancer.pickle{}.dat".format(select_x_train.shape[1]), "wb"))

    print("Thresh= %.3f,n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
                score*100.0))
    
import pickle#파이썬에서 제공하는 피클
pickle.dump(model, open("./model/xgb_save/breast_cancer.pickle.data", "wb"))
print("SAVED!!!!")
    
