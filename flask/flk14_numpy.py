import pymssql as ms
import numpy as np

conn = ms.connect(server ='127.0.0.1', user='bit2', password ='1234', 
                    database='bitdb',port = 65009)


cursor = conn.cursor()

cursor.execute('SELECT * FROM iris;')
#sql의 가장 끝에는 항상 ';'를 넣어 줘야 함

row = cursor.fetchall()
print(row)
conn.close()
print('=============================================================')
aaa =np.asarray(row)
print(aaa)
print(aaa.shape)
print(type(aaa))

np.save('test_flask_iris.npy', aaa)