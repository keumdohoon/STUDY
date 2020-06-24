# import p31_sample
#from 을 사용하여서 p31_sample에 test만을 import해준다 그렇게 되면 나중에 test()만을 써줘도 나오게 된다. 
from p31_sample import test
                        #함수명만쓰면
x = 222

def main_func():
    print('x:', x)

# p31_sample.test()
test()
#이렇게 test()만 써도된다.
main_func()