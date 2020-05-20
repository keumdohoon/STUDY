#3. 딕셔너리 #중복x
# {키 : 벨류}
# {key : value}


a = { 1:'hi', 2: 'hello'}
print(a)
print(a[1])
#a = {'hi':1, 'hello':2}
#print(a['hello'])#2를 나오게 하려면



b = {'hi': 1, 'hello': 2}
print(b['hello'])

#딕셔너리 요소 삭제
#del a[1]
print(a)

del a[2]
print(a)

a = {1:'a', 1:'b', 1:'c'}#마지막것만 남는다 주가 되는 1은 항상 같기 때문에 1=a->b->c로 되기 때문에 덮어씌어져서 c가 결과가 된다.
print(a)
b = {1:'a', 2:'a', 3:'a'}#1,2,3,은 각각 다르기때문에 다 프린트 되게 되어있다. 
print(b)

a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}
print(a.keys())
print(a.values())
print(type(a))
print(a.get('name'))
print(a['name'])
print(a.get('phone'))
print(a['phone'])

