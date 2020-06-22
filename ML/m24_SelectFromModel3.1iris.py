#책에 없는 내용이니 주의를 매우 기우릴것
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

model = XGBClassifier()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("R2 :", score) #R2 : 0.9221188544655419

thresholds = np.sort(model.feature_importances_)
             #정렬 #중요도가 낮은 것부터 높은것 까지

print(thresholds)  

#for문을 전체 컬럼수 만큼 돌리면 총 13번 돌리게 된다. 

for thresh in thresholds:     #총 컬럼수 만큼 돌게 된다 빙글빙글. 
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                            #median
    select_x_train = selection.transform(x_train)
    print(select_x_train.shape) #결과를 보면 컬럼이 13->1로 쭈욱 내려가는데 이것을 컬럼의 중요도가 없는 컬럼을 하나씩 지워주는 것이다.최종 1 

    
    selection_model = XGBClassifier()
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    x_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, x_pred)
    print("R2 : ", score)

    print("Thresh= %.3f,n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
                score*100.0))
    
# R2 : 0.9
# [0.01759811 0.02607087 0.33706376 0.6192673 ]
# (120, 4)
# R2 :  0.8489932885906041
# Thresh= 0.018,n=4, R2: 84.90%
# (120, 3)
# R2 :  0.8489932885906041
# Thresh= 0.026,n=3, R2: 84.90%
# (120, 2)
# R2 :  0.9496644295302014
# Thresh= 0.337,n=2, R2: 94.97%
# (120, 1)
# R2 :  0.8993288590604027
# Thresh= 0.619,n=1, R2: 89.93%

#n=2 에서 가장 높은 값을 나타내어준다. 