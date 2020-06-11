from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=3, test_size=0.2
)

model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1) # depth 깊으면 과대적합됨
# max_depth 디폴트
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

print(model.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model) :
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("feature importance")
    plt.ylabel("features")
    plt.ylim(-1, n_features)
plot_feature_importances_cancer(model)
plt.show()
