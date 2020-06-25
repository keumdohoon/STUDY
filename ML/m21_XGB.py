from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np


cancer = load_breast_cancer()
#cancer 를 로드해준다 하지만 이 데이터셋에서는 이미 data와 target으로 나뉘어져 있다. 
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size =0.8, random_state= 42
)
#이미 data와 target으로 나뉘어져 있으니 이를 x와 y를 대신해서 넣어준다 





# model = DecisionTreeClassifier(max_depth = 4 )
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()
# 위 4개의 모델들중 XGBClassifier을 사용하게 될것이다. 



###XGBClassifier에 들어가는 파라미터들 중 일부#####
# max_features : 기본값 쓸것!
# n_estimators : 클수록 좋다!, 단점 메모리를 많이 차지 한다는것을 인지하고 기본값은 100이다. 
# n_jobs = -1  : 병렬 처리( *주의 gpu 같이 돌릴땐 2이상 값을 주면 터진다)


#max_features : 기본값 써라!
# n_estimators : the bigger the better클수록 좋은거다, 단점으로는 메모리를 짱 차지한다 , 기본값 100
#n_jobs = -1 : 병렬처리 (gpu 같이 돌릴때는 2이상의 값을 쓸때는 터진다.우리는 12코어를 사용하고 12개를 다 사용해주기 위해서 -1즉 전체를 다써준다는 얘기다. )

model.fit(x_train, y_train)
acc = model.score(x_test , y_test)

#위에서 명시해준 모델 중에서 XGB모델을 사용하여 주었기 때문에 이를 보델로 사용하여 fit을 해주고 또 model.score을 해준다. 특히나 
#여기서는 compile 을 사용해줄 필요가 없다. 
print(model.feature_importances_)
print(acc)
#feature importance라고 가장 좋은 feature을 뽑아다 쓰는 것이다. 


def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
    align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

#이거를 분석을 하고 이것을 데이콘에 써먹을수 있으니 적용하도록 하자
#직접 찾아보기 
#barh, stick , lim 이 뭔지 찾아서 주석처리하기 
plot_feature_importances_cancer(model)
plt.show()

#cancer 모델에서 중요한 feature importance 를 뽑아서 plot으로 그려준다. 

#과제: 데이콘에 이 featureimportance를 적용하여 확인을 할것 
#feature importances는 feature enginerring 이라고 한다.



