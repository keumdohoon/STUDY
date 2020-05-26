from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape= (10,10,1)))   #(9,9,10)
#가로세로 2,2로 잘라 버리겠다는 것이다. the third number in the input shape parameter means the color of the object so it would be , length, height, and color black
# CNN은 4차원이다.CNN은 한번 자른것 가지고 데이터의 특성을 알아내지 못하니 한번 자른걸 자르고 또 자른다. 
model.add(Conv2D(7, (3,3))) ##(7,7,7)
model.add(Conv2D(5, (2,2), padding = 'same'))       #(7,7,5)
model.add(Conv2D(5, (2, 2)))              #(3, 3, 5)
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten)
model.add(Dense(1))
model.summary()
#지금 댄스의 레이어가 4차
# 여기서 padding의 디폴트 값은 valid 이다. 
# stride띄어주는 것 은 항상 디폴트값 1로 되어있다. 
# model.add(Conv2D(10, kernel_size=2, input_shape = (5,5,1)))
#flatten은 데이터를 4차원의 데이터를 쭈욱 펴준다는 의미이다. 그위에 있는 (n,3,3,5)를 다 곱해준 값이다 즉 45가 된다. convolutionlayer에 끝은 항상 flatten이다. 