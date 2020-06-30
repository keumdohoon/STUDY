import numpy as np
import re

def labeling(path):  
    f = open(path, 'r')
    txt = f.read()
    png_num = re.findall('png\s\d', txt) 
    label = []
    for num in png_num:
        num = int(num.replace('png ', ''))
        label.append(num)
    return(np.array(label))

path_train = '/tf/notebooks/Keum/data/train/train_label_COVID.txt'
path_val = '/tf/notebooks/Keum/data/validate/validate_label_COVID.txt'
# path_test = '/tf/notebooks/Ja/data/test/test_label.txt'

y_train= labeling(path_train)
y_val = labeling(path_val)

print(len(y_train)) # 546
print(len(y_val))   # 100

np.save('/tf/notebooks/Keum/data/y_train.npy', arr = y_train)
np.save('/tf/notebooks/Keum/data/y_val.npy', arr= y_val)
print('save_complete')


