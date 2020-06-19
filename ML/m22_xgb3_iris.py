#iris 모델 다중분류

#feature을 xgboost로 건들겠다 
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import matplotlib.pyplot as plt

dataset = load_iris()
x= dataset.data
y = dataset.target

print(x.shape) #(150, 4)
print(y.shape) #(506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
                                                    shuffle = True, random_state = 120)






n_estimators = 220
learning_rate = 0.035      
colsample_bytree = 0.7   
colsample_bylevel = 0.7  
# 점수:  1.0


#cv는 꼭 써서 결과치를 보고 feature importance도 봐줘야한다. 
#위에 이것 4개만 조정해주면 된다. 
max_depth = 5
n_jobs = -1
#tree model 이 ensemble 된게 forest 이다
#boosting계열에서는 tree구조에서 쓰던 특성을 다 가져온다 
#특성에는 : 전처리가 필요없다, 결측치 제거를 안해도된다, 
#xgboost는 앙상블이니까 decision tree보다는 많이 느리다 .

model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, colsample_bytree=colsample_bytree
                                    , n_estimators=n_estimators, n_jobs=n_jobs, colsample_bylevel= colsample_bylevel)
                    
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('점수: ', score)

print(model.feature_importances_)
# print('========================================')
# print(model.best_estimator_)
# print('========================================')

# print(model.best_params_)
# print('========================================')

plot_importance(model)
# plt.show()
#어떤 feature이 중요한지 쉽게 볼수가 있다. 딱 한 줄로 정리가 가능 