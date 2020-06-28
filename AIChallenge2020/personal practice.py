# USAGE
# python train.py --dataset dataset

# import the necessary packages

from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from PIL import Image
import os, glob
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from keras.preprocessing.image import img_to_array


### 이미지 파일 불러오기 및 카테고리 정의
pic_dir = './dataset'

categories = ['covid', 'normal']
nb_classes = len(categories)


### 가로, 세로, 채널 쉐이프 정의
image_w = 150
image_h = 150
# pixels = image_h * image_w * 3
### 이미지 파일 Data화
X = []
Y = []

for index, categories in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[index] = 1
    image_dir = pic_dir + '/' + categories        #예를들면, './miniprojectdata/data/kangdongwon 
    files = glob.glob(image_dir + "/*.*")   #확장자가 jpg인 모든 이미지를 불러오는 라이브러리
    print(categories, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        Y.append(label)
#numpy

x = np.array(X)
y = np.array(Y)
print(x.shape)
print(y.shape)
'''
### 데이터 train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
xy = (x_train, x_test, y_train, y_test)
### 데이터 SAVE
print('>>> data 저장중 ...')
# np.save('./miniprojectdata/data/datasetNPY/datasets.npy', xy)
'''