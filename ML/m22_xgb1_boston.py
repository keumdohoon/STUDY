#과적함 방지
#1. 훈련데이터량을 늘린다. 
#2. 피처수를 줄인다. 
#3. regularization
#4. boston 회기모델

#feature을 xgboost로 건들겠다 
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import matplotlib.pyplot as plt

dataset = load_boston()
x= dataset.data
y = dataset.target

print(x.shape) #(506, 13)
print(y.shape) #(506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,
                                                    shuffle = True, random_state = 66)



# n_estimators = 1510
# learning_rate = 0.023   #deeplearning의 loss부분이랑 연관되어있다.학습률이라는 뜻이고 보통 디폴트 0.01 이다. 이걸 바꾸면 많은것들이 바뀜 바뀌는 핵심 키워드중에 하나이다.   
# colsample_bytree = 0.712  #0.0부터 1 까지인데 우승 모델은 0.6~0.9를 사용한다. 디폴트는 1이 나온다 왜냐하면 전부를 활용하기 때문이다. 
# colsample_bylevel = 0.712   #
# 점수:  0.9405326446015613

# n_estimators = 10000
# learning_rate = 0.05      
# colsample_bytree = 0.9   
# colsample_bylevel = 0.7  
# 점수:  0.9430531781646333

n_estimators = 2000
learning_rate = 0.05      
colsample_bytree = 0.9   
colsample_bylevel = 0.6  
# 점수:  0.9482512438175588




#cv는 꼭 써서 결과치를 보고 feature importance도 봐줘야한다. 
#위에 이것 4개만 조정해주면 된다. 
max_depth = 5
n_jobs = -1
#tree model 이 ensemble 된게 forest 이다
#boosting계열에서는 tree구조에서 쓰던 특성을 다 가져온다 
#특성에는 : 전처리가 필요없다, 결측치 제거를 안해도된다, 
#xgboost는 앙상블이니까 decision tree보다는 많이 느리다 .

model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, colsample_bytree=colsample_bytree
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