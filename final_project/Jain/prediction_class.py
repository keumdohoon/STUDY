import numpy as np
import cv2
import re
import os
from tensorflow.keras.models import load_model
from class_project import Project

# data
#------------- only one file -------------------
# path = 'D:/data/project/testset/bichon.jpg'
# img = cv2.imread(path)
# img = cv2.resize(img, dsize = (128, 128), interpolation = cv2.INTER_LINEAR)
# x_pred = np.array(img/255).reshape(-1, 128, 128, 3)                 # 픽셀 = 0~255값가짐, 학습을 위해 0~1사이의 소수로 변환

#---------------- load_data -------------------
path = 'D:\Study\Bitcamp\\final_project\\testset' 

# 테스트 사진 세트를 가지고 와 준다. 




x_pred = np.load('D:\Study\Bitcamp\\final_project\Jain\pred_img.npy')
                                 # filename
print(x_pred.shape)

#npy로 저장해둔 예상 이미지를  불러 와준다. 그리고 그것을 x_pred로 지정해주고 난 다음에. 



# load_model
model = load_model('D:\Study\Bitcamp\\final_project\model_save\\best_Xception_32.hdf5')

#모델을 가져오게 되는데 이는 우리가 best_model에 저장해둔 hdf5가중치를 말하는 것이다. 



# predict
prediction = model.predict(x_pred)
number = np.argmax(prediction, axis = 1)
print(len(number))
#가중치에다가 우리가 가지고 와준 사진 이미지를 넣은 다음에 이중에서 가장 정확도가 높은것을 뽑아주기 위해서 argmax를 사용해준다. 



# # 카테고리 불러오기
# categories = ['Bichon_frise', 'Border_collie', 'Bulldog', 'Chihuahua', 'Corgi', 'Dachshund', 'Golden_retriever', 
#                 'Huskey', 'Jindo_dog', 'Maltese', 'Pug', 'Yorkshire_terrier']


# f = open('./project/project02/data/pred_image_name.txt', 'r')
# filename = f.readlines()

# # filename = ['Jindo_dog']

# for i in range(len(number)):
#     idex = number[i]
#     true = filename[i].replace('\n', '')
#     pred = categories[idex]
#     print('실제 :', true, '\t예측 견종 :', pred)