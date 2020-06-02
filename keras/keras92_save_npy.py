from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

print(type(iris))

x_data = iris.data
y_data = iris.target
#가져온 sklearn.utils파일을 사용할수 있는 numpy 파일로 변환
print(type(x_data))
print(type(y_data))

np.save('./data/iris_x.npy', arr=x_data)
np.save('./data/iris_y.npy', arr=y_data)
#위에 이게 저장하는법




x_data_load = np.load('./data/iris_x.npy')
y_data_load = np.load('./data/iris_y.npy')
#밑에 이게 불러오는 법


print(type(x_data_load))
print(type(y_data_load))
print(x_data_load.shape)
print(y_data_load.shape)
