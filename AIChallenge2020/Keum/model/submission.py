
import glob
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, f1_score
from keras.models import load_model
from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2) 
leaky.__name__ = 'leaky'

#1. data
x_train = np.load('/tf/notebooks/Keum/data/x_train.npy').reshape(-1, 384, 384, 1)
x_pred = np.load('/tf/notebooks/Keum/data/x_test.npy').reshape(-1, 384, 384, 1)
x_val = np.load('/tf/notebooks/Keum/data/x_val.npy').reshape(-1, 384, 384, 1)

y_train = np.load('/tf/notebooks/Keum/data/y_train.npy')
y_val = np.load('/tf/notebooks/Keum/data/y_test.npy')



# load_model
model = load_model('/tf/notebooks/Keum/save_model/model02.h5')

# y_pred = model.predict(x_test)
# f1_score = f1_score(y_test, y_pred)
# print('f1_score : ', f1_score)

# y_predict = model.predict(x_pred)

# submit_data
y_predict = model.predict(x_pred)
y_predict = np.where(y_predict >= 0.5, 1, 0)
#0.5보다 크면 1로 설정해주고 0.5보다 작으면 0으로 출력을 해준다
y_predict = y_predict.reshape(-1,)

def submission(y_sub):
    for i in range(len(y_sub)):
        path = '/tf/notebooks/Keum/data/test/test_label_COVID.txt'
        #pat에서 test_label파일을 가져와준다 이거는 실제 우리가 isi해줘야하는 모델
        f1 = open(path, 'r')
        title = f1.read().splitlines()
        f = open('/tf/notebooks/Keum/sub/submission.txt', 'a', encoding='utf-8')
        #submission.txt라는 파일을 만들고 거기에 우리가 편집할수 있게하는 'a'라는 것을 명시해준다.
        #주의할점은 저게 이미 있는 파일이면 거기에다가 붙여넣기를 해줌으로 주의할것
        #만약 없는 파일이면 저 파일을 새로 생성하게 된다 그리고 그 파일은 a로 편집할 수 있는 파일이다. 
        f.write(title[i]+' '+str(y_sub[i]) + '\n')
        #이제 위에서 생성해준 f라는 파일에 write로 적어주겠다. title을 가져오고 그 사이를 ' ' 로 띄워주고
        #-뒤에는 문자형으로 y_sub[i]를 적어주겠다 뒤에 +/n은 '엔터' 즉 다음줄로 이동해 주겠다는 것이다. 
        #그러면 결과값이 나오게 된다. 
    print('complete')

submission(y_predict)