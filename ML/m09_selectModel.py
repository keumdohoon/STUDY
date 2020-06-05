import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

# ./data/csv/winequality-white.csv'
iris = pd.read_csv('./data/csv/iris.csv', header = 0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]
print(type(iris))
print(x)
print(y)

# warnings.filterwarning('ignore')


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state = 66)
# warnings.filterwarning('ignore')


allAlgorithms = all_estimators(type_filter = 'classifier')

for (name, algorithm) in allAlgorithms:
        model = algorithm()
    
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, "의 정답률 =", accuracy_score(y_test, y_pred))

import sklearn
print(sklearn.__version__)
# warnings.filterwarning('ignore')
