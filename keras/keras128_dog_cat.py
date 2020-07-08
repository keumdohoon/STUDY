from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img


img_dog = load_img('./data/dog.cat/dog.jpg')
img_cat = load_img('./data/dog.cat/cat.jpg')
img_suit = load_img('./data/dog.cat/suit.jpg')
img_yang = load_img('./data/dog.cat/yang.jpg')

plt.imshow(img_yang)
plt.imshow(img_cat)
plt.imshow(img_suit)

# plt.show()

from keras.preprocessing.image import img_to_array

arr_dog = img_to_array(img_dog)

img_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_yang = img_to_array(img_yang)

print(arr_dog)
print(type(arr_dog))
print(arr_dog.shape) #(485, 729, 3)


# RGB->BGR
from keras.applications.vgg16 import preprocess_input


arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_yamg = preprocess_input(arr_yang)

print(arr_dog)
print(arr_dog.shape) #(485, 729, 3)

print(arr_cat)
print(arr_cat.shape)

print(arr_suit)
print(arr_suit.shape)

print(arr_yang)
print(arr_yang.shape)
# 이미지를 하나로 합친다
import numpy as np
arr_input = np.stack([arr_dog, arr_cat, arr_suit, arr_yang])

print(arr_input.shape)

#모델 구성
model = VGG16()
probs = model.predict(arr_input)

print(probs)
print('probs.shape : ', probs.shape)

#이미지 결과 
from keras.applications.vgg16 import decode_predictions

results =decode_predictions(probs)


print(results[0])
print('-----------------------------------------------------------------------------------------------------------')
print(results[1])
print('-----------------------------------------------------------------------------------------------------------')
print(results[2])
print('-----------------------------------------------------------------------------------------------------------')
print(results[3])
