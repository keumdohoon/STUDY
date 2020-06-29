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
#결국 from을 뭘적어주냐와 import를 어디까지 해주냐에 따라서 나중에 출력할때 어디까지 적어줘야되는지가 결정이 난다. 