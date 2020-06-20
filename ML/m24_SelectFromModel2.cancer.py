#책에 없는 내용이니 주의를 매우 기우릴것
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

model = XGBClassifier()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("R2 :", score) #R2 : 0.9221188544655419

thresholds = np.sort(model.feature_importances_)
             #정렬 #중요도가 낮은 것부터 높은것 까지

print(thresholds)   #[0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
                    #0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643, 0.42848358]
                    #점점 값이 올라가는 오름차순의 형태인걸 확인.

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
    

    
# (455, 30)
# R2 :  0.885733377881724
# Thresh= 0.000,n=30, R2: 88.57%
# (455, 30)
# R2 :  0.885733377881724
# Thresh= 0.000,n=30, R2: 88.57%
# (455, 28)
# R2 :  0.885733377881724
# Thresh= 0.000,n=28, R2: 88.57%
# (455, 27)
# R2 :  0.885733377881724
# Thresh= 0.002,n=27, R2: 88.57%
# (455, 26)
# R2 :  0.885733377881724
# Thresh= 0.003,n=26, R2: 88.57%
# (455, 25)
# R2 :  0.885733377881724
# Thresh= 0.003,n=25, R2: 88.57%
# (455, 24)
# R2 :  0.885733377881724
# Thresh= 0.003,n=24, R2: 88.57%
# (455, 23)
# R2 :  0.885733377881724
# Thresh= 0.003,n=23, R2: 88.57%
# (455, 22)
# R2 :  0.8476445038422986
# Thresh= 0.004,n=22, R2: 84.76%
# (455, 21)
# R2 :  0.8476445038422986
# Thresh= 0.004,n=21, R2: 84.76%
# (455, 20)
# R2 :  0.885733377881724
# Thresh= 0.005,n=20, R2: 88.57%
# (455, 19)
# R2 :  0.885733377881724
# Thresh= 0.005,n=19, R2: 88.57%
# (455, 18)
# R2 :  0.8476445038422986
# Thresh= 0.005,n=18, R2: 84.76%
# (455, 17)
# R2 :  0.8476445038422986
# Thresh= 0.006,n=17, R2: 84.76%
# (455, 16)
# R2 :  0.8476445038422986
# Thresh= 0.006,n=16, R2: 84.76%
# (455, 15)
# R2 :  0.885733377881724
# Thresh= 0.008,n=15, R2: 88.57%
# (455, 14)
# R2 :  0.885733377881724
# Thresh= 0.008,n=14, R2: 88.57%
# (455, 13)
# R2 :  0.9238222519211493
# Thresh= 0.009,n=13, R2: 92.38%
# (455, 12)
# R2 :  0.9238222519211493
# Thresh= 0.012,n=12, R2: 92.38%
# (455, 11)
# R2 :  0.9238222519211493
# Thresh= 0.014,n=11, R2: 92.38%
# (455, 10)
# R2 :  0.9238222519211493
# Thresh= 0.014,n=10, R2: 92.38%
# (455, 9)
# R2 :  0.885733377881724
# Thresh= 0.018,n=9, R2: 88.57%
# (455, 8)
# R2 :  0.885733377881724
# Thresh= 0.023,n=8, R2: 88.57%
# (455, 7)
# R2 :  0.9238222519211493
# Thresh= 0.024,n=7, R2: 92.38%
# (455, 6)
# R2 :  0.885733377881724
# Thresh= 0.033,n=6, R2: 88.57%
# (455, 5)
# R2 :  0.8095556298028733
# Thresh= 0.066,n=5, R2: 80.96%
# (455, 4)
# R2 :  0.8476445038422986
# Thresh= 0.097,n=4, R2: 84.76%
# (455, 3)
# R2 :  0.7714667557634479
# Thresh= 0.116,n=3, R2: 77.15%
# (455, 2)
# R2 :  0.6191112596057466
# Thresh= 0.222,n=2, R2: 61.91%
# (455, 1)
# R2 :  0.5048446374874705
# Thresh= 0.285,n=1, R2: 50.48%