#VGG16 이라는 뜻이다. 
from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152V2
from keras.applications import ResNet152, ResNet50, ResNet50V2, InceptionResNetV2, InceptionV3
from keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam



# vg16 = VGG16()  #(None, 224, 224, 3)
# vg19 = VGG19()
# Xception = Xception()
# ResNet101 =ResNet101()
# ResNet101V2 =ResNet101V2()
# ResNet152V2 =ResNet152V2()
# ResNet152 =ResNet152()
# ResNet50 =ResNet50()
# ResNet50V2 =ResNet50V2()
# InceptionResNetV2 =InceptionResNetV2()
# InceptionV3 =InceptionV3()
# MobileNet =MobileNet()
# MobileNetV2 =MobileNetV2()
# DenseNet121 = DenseNet121()
# DenseNet169 = DenseNet169()
# DenseNet201 =DenseNet201()

applications = [VGG19, Xception, ResNet101, ResNet101V2, ResNet152,ResNet152V2, ResNet50, 
                ResNet50V2, InceptionV3, InceptionResNetV2,MobileNet, MobileNetV2, 
                DenseNet121, DenseNet169, DenseNet201]

for i in applications:
    take_model = i()
# vg19.summary()
# vg16.summary()
# Xception.summary()
# ResNet101.summary()
# ResNet101V2.summary()
# ResNet152V2.summary()
# ResNet152.summary()
# ResNet50.summary()
# ResNet50V2.summary()
# InceptionResNetV2.summary()
# InceptionV3.summary()
# MobileNet.summary()
# MobileNetV2.summary()
# DenseNet121.summary()
# DenseNet169.summary()
# DenseNet201.summary()

#프린트해서 나온 모델을 그대로 갖다 붙여넣기 하여서 우리가 직접 조금씩 튜닝하여 사용할 수 있다. 
# model = Sequential()
# model.add(vg16)
# model.add(Flatten())
# model.add(Dense(256))
# model.add(BatchNormalization())
# model.add(Activation())
# model.add(Dense(10, activation = 'softmax'))

# model.sumamry()