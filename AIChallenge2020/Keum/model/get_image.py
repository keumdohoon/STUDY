import numpy as np
import cv2
import os
import glob
from fourier_trans import fourier_trans

# os로 불러오기 : 파일 형식 상관 X
def load_image_os(image_dir):
    X = []
    # resize 크기
    image_w = 384
    image_h = 384

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            # print(image_dir+filename)
            img = cv2.imread(image_dir + filename)
            img = cv2.resize(img, dsize = (image_w, image_h), interpolation = cv2.INTER_LINEAR)
            X.append(img/255)     # 픽셀 = 0~255값가짐, 학습을 위해 0~1사이의 소수로 변환
    
    return np.array(X)



# glob으로 불러오기 : 지정한 파일 형식만 불러오기
def load_image(image_dir):
    X = []
    # resize 크기
    image_w = 384
    image_h = 384

    image_dir = glob.glob(image_dir+'*.png')

    for filename in image_dir:
        # print(filename)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize = (image_w, image_h), interpolation = cv2.INTER_LINEAR)
        X.append(img/255)     # 픽셀 = 0~255값가짐, 학습을 위해 0~1사이의 소수로 변환
        # print(img.shape)
    
    return np.array(X)


def load_image_fourier(image_dir):
    X = []
    # resize 크기
    image_w = 384
    image_h = 384   

    image_dir = glob.glob(image_dir+'*.png')

    for filename in image_dir:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize = (image_w, image_h), interpolation = cv2.INTER_LINEAR)

        # print(img)
        img_back = fourier_trans(img)
        X.append(img_back/255)     # 픽셀 = 0~255값가짐, 학습을 위해 0~1사이의 소수로 변환
    
    return np.array(X)

train_path = '/tf/notebooks/Keum/data/train/'
test_path = '/tf/notebooks/Keum/data/test/'
val_path = '/tf/notebooks/Keum/data/validate/'

# #os#에러
# x_train = load_image_os(train_path)
# x_test = load_image_os(test_path)
# x_val = load_image_os(val_path)

##load image
x_train = load_image(train_path)
x_test = load_image(test_path)
x_val = load_image(val_path)


# #fourier
# x_train = load_image_fourier(train_path)
# x_test = load_image_fourier(test_path)
# x_val = load_image_fourier(val_path)

print(x_train.shape)
print(x_test.shape)
print(x_val.shape)

# print(x_val[10,:])


# np.save('/tf/notebooks/Keum/data/x_train.npy', arr = x_train) 
# np.save('/tf/notebooks/Keum/data/x_test.npy', arr = x_test) 
np.save('/tf/notebooks/Keum/data/x_val.npy', arr = x_val) 

print('save complete')