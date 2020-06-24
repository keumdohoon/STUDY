from machine.car import drive
from machine.tv import watch

drive()
watch()
#만약from machine.car이라고 적고  drive를 바로 임포트해준것이라면 drive()이라고 적으면 된다. 


from machine import car
from machine import tv

car.drive()
tv.watch()
#만약 from machine이라고 해주고 import카를 해주면 그것에 하위인 drive()까지 해줘야 결과가 나온다 

print("########################################################")
#위에랑 다른 부분은 test폴더 안에 들어가있다는 것임
from machine.test.car import drive
from machine.test.tv import watch
#현재 폴더에 machine폴더에 test폴더안에 car을 from 해주고 drive를 임포트 해주면 바로 drive()이라고 적어줘도 작동이 잘된다. 
drive()
watch()

from machine.test import car
from machine.test import tv
#만약 
car.drive()
tv.watch()

from machine import test
from machine import tv

test.car.drive()
test.tv.watch()