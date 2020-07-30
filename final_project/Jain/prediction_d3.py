import numpy as np
import cv2
import re
import os
from keras.models import load_model

# data
path = 'D:/data/project/testset'  #예측을 하기 위해서는 우리가 예측을 하게 되는 대상이 있어야하는데 그 파일을 준비해 주어야 한다. 
'''
image_w = 112
image_h = 112

x_pred = []
for top, dir, f in os.walk(path): 
    for filename in f:
        img = cv2.imread(top+'/'+filename)
        img = cv2.resize(img, dsize = (image_w, image_h), interpolation = cv2.INTER_LINEAR)
        x_pred.append(img/255)     

x_pred = np.array(x_pred)         
np.save('./project/project02/data/pred_img.npy', x_pred)
print(x_pred.shape)
print('----------- x_pred save complete ---------------')
'''
x_pred = np.load('D:\Study\Bitcamp\\final_project\Jain/pred_img.npy')

#우리가 prediction을 위해서 만들어둔 pred_img.npy파일을 불러 와준다. 이것을 하는 방법은 위에 나와있다. 원래의 이미지를 

# load_model
model = load_model("D:\Study\Bitcamp\\final_project\model_save\\best_Xception_32.hdf5")

#우리가 만들어 놓은 가중치에서 우리가 가지고 와주고 싶은 가중치를 가지고 와준다. 

# predict
prediction = model.predict(x_pred)

#우리가 가져온 가중치랑 우리가 만들어놓은 프리딕션 사진을 predict해준다. 

number = np.argmax(prediction, axis = 1)
#
#argmax를 통해서 가장 높은 확률을 가지고 있는 견종의 카테고리를 출력한다. 

# 카테고리 불러오기
categories = ['Bichon_frise', 'Border_collie', 'Bulldog', 'Chihuahua', 'Corgi', 'Dachshund', 'Golden_retriever', 
                'Huskey', 'Jindo_dog', 'Maltese', 'Pug', 'Yorkshire_terrier']

#카테고리의 순서가 중요한 이유는 아르그 맥스에서 나온 자리의 숫자에 카테고리를 맞춰줘야하기 때문이다. 

filename = os.listdir(path)

#우리가 테스트 폴더 안에 있는 이미지의 이름들이 각각의 매칭되는 견종 이름으로 되어 있다 그 견종이 뭔지 확인을 하기위해서 이렇게 써 주었다. 
#파일 이름이 견종 이름이여야지만 사용할 수 있는 , 그 파일이 뭐로 예측되어 있는지 알려고 할때 사용할수 있다. 
for i in range(len(number)):
    idex = number[i]
    true = filename[i].replace('.jpg', '').replace('.png','')
    pred = categories[idex]
    print('실제 :', true, '\t예측 견종 :', pred)

