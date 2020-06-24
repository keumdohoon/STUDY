from machine.car import drive
from machine.tv import watch

drive()
watch()
#만약from machine.car이라고 적고  drive를 바로 임포트해준것이라면 drive()이라고 적으면 된다. 

from machine import car
from machine import tv
#만약 from machine이라고 해주고 import카를 해주면 그것에 하위인 drive()까지 해줘야 결과가 나온다 

car.drive()
tv.watch()