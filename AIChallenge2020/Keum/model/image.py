import cv2
import glob
train_path = '/tf/notebooks/Keum/data/train/'
image_dir = glob.glob(train_path+'*.png')
image_dir = image_dir[0]

img = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
print(img.shape)
#(384, 384)