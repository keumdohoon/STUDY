#scikitlearn패키지가 제공하는 fetch_20newsgroups 데이터를 사용한다 이 데이터는 20개 카테고리에 대한 18000개의 뉴스 제보가 수록되어 있다. 

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset = 'train')
newsgroups_test = fetch_20newsgroups(subset = 'test')

#클래스 이름 줄이기
class_names = [x, split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:])
                for x in newsgroups_train.target_names]

print(class_names)

class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'

print(class_names)

