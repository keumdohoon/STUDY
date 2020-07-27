#datetime package
'''
1. datetime: 날짜 시간 저장
2. date    : 날짜
3. time    : 시간
4. timedelta: 시간구간 정보 저장
'''

from datetime import datetime

# .now()
# 현재시각 출력
now = datetime.now()
print(now)    #2020-07-17 15:14:02.449012

#year-2020
print(now.year)

#month-7
print(now.month)

#day -17
print(now.day)

#hour-15
print(now.hour)

#minute-15
print(now.minute)

#second-58
print(now.second)

#microsecond-481484
print(now.microsecond)



#weekday()
#요일변환 : (0:월, 1:화, 2:수, 3:목, 4:금, 5:토, 6:일)
print(now.weekday())















