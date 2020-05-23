#p.04_finding core talents
users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" },
    { "id": 10, "name": "Jen" }
]

#id와 쌍으로 구성된 friendship_pairs, (0,1)은 Hero 와 Dunn 이 친구라는 뜻이다. 
friendship_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]
#사용자별로 비어 있는 친구 목록 리스트?를 지정하여 딕셔너리를 초기화
friendships  = {user["id"] : [] for user in users}

for i, j in friendship_pairs:
    # 각각의 friendshipdl 이 어디에 연결되어있는건지를 지정해주는 것이다. 
    friendships[i].append(j)#j를 사용자 i의 친구로 추가
    friendships[j].append(i)

def number_of_friends(user) :
    #user의 친구는 몇명일까
    user_id = user["id"]
    friend_ids = friendships[user_id]
    return len(friend_ids)

total_connections = sum(number_of_friends(user) for user in users)

print(total_connections) #24

num_users = len(users)
avg_connections = total_connections / num_users



# p.07_DATA SCIENTISTS YOU MAY KNOW #

def foaf_ids_bad(user):
    # foaf 는friend of a friend의 줄인 말로써 친구의 친구라는 뜻이다. 
    return [foaf_id
            for friend_id in friendships[user["id"]] # 각 유저의 친구
            for foaf_in friendships[friend_id]] # 각 친구의 친구를 구하여라
#Hero=[0,2,3,0,1,3]

from collections import Counter

def friends_of_friends(user):
    user_id = user["id"]
    return Counter(
        foaf_id
        for friend_id in friendships[user_id]
        for foaf_id in friendships[friend_id]
        if foaf_id !=user_id
        and foaf_id not in friendships[user_id]

print(friends_of_friends_ids(users[3])) # Counter({0: 2, 5: 1})

interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
    (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"),
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
    (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
    (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
    (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

def data_scientists_who_like(target_interest):
    """특정 관심사를 가지고 있는 모든 사용자 아이디를 변환"""
    return [user_id
            for user_id, user_interest in interests
            if user_interest == target_interest]

from collections import defaultdict

# 키가 관심사이고, 값이 사용자 id이다.keys=interests, values = lists of user_ids with that interest
user_ids_by_interest = defaultdict(list)

for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)

# 키가 사용자 id이고 관심사가 값이 되는 모형.keys = user_ids, values = lists of interests for that user_id
interests_by_user_id = defaultdict(list)

for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

def most_common_interests_with(user_id):
    return Counter(
        interested_user_id
        for interest in interests_by_user_id[user["id"]]
        for interested_user_id in user_ids_by_interest[interest]
        if interested_user_id != user_id)


#p.10_SALARIES AND EXPERIENCE
#밑에 그래프는 연봉과 근속 년수를 정해준 것이다. 이데이터를 가지고 연봉과 경력의 상관관계를 알아보는 단계.
salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
                        (48000, 0.7), (76000, 6),
                        (69000, 6.5), (76000, 7.5),
                        (60000, 2.5), (83000, 10),
                        (48000, 1.9), (63000, 4.2)]


# keys 는 년수로 측정해 줄것이고.
# values(값) 은 연봉으로 분류할 것이다. 
salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

#이번에도 비슷한 방법으로 할것인데 위에 모델과의 차별점은 연봉을 평균연봉으로 바꾼것이다. 
average_salary_by_tenure = {
    tenure : sum(salaries) / len(salaries)
    for tenure, salaries in salary_by_tenure.items()
}
print(average_salary_by_tenure)

#하지만 위의 형식으로 하면 연봉과 근속년수의 상관관계보다는 데이터가 부족하기에 개개인의 연봉과
#근속년수를 리스트화 해둔것 밖에 되지 않는다. 이 데이터를 더욱 효과적인 데이터로 만들기 위해서는 
#각각의 구간으로 나누고 데이터화시키겠다. 
def tenure_bucket(tenure):
    if tenure < 2: 
        return "less than two"
    elif tenure < 5: 
        return "between two and five"
    else: return "more than five"
#키는 근속 연수 구간, 값은 해당 구간에 속하는 사용자들의 연봉으로 설정해두겠다. 
salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)
#이것도 위의 모델과 유사한데 연봉을 평균 연봉으로 바꾸어 주었다. 
average_salary_by_bucket = {
  tenure_bucket : sum(salaries) / len(salaries)
  for tenure_bucket, salaries in salary_by_tenure_bucket.iteritems()
}

print(average_salary_by_bucket)

#p.12_PAID_ACCOUNTS

def predict_paid_or_unpaid(years_experience):
  if years_experience < 3.0:
    return "paid"
  elif years_experience < 8.5:
    return "unpaid"
  else: 
    return "paid"

#p.13_TOPICS OF INTEREST

words_and_counts = Counter(word
                           for user, interest in interests
                           for word in interest.lower().split())


if __name__ == "__main__":

    print
    print ("######################")
    print ("#")
    print ("# FINDING KEY CONNECTORS")
    print ("#")
    print ("######################"
    print


    print "total connections", total_connections
    print "number of users", num_users
    print "average connections", total_connections / num_users
    print

    # create a list (user_id, number_of_friends)
    num_friends_by_id = [(user["id"], number_of_friends(user))
                         for user in users]

    print "users sorted by number of friends:"
    print sorted(num_friends_by_id,
                 key=lambda (user_id, num_friends): num_friends, # by number of friends
                 reverse=True)                                   # largest to smallest

    print
    print "######################"
    print "#"
    print "# DATA SCIENTISTS YOU MAY KNOW"
    print "#"
    print "######################"
    print


    print "friends of friends bad for user 0:", friends_of_friend_ids_bad(users[0])
    print "friends of friends for user 3:", friends_of_friend_ids(users[3])

    print
    print "######################"
    print "#"
    print "# SALARIES AND TENURES"
    print "#"
    print "######################"
    print

    print "average salary by tenure", average_salary_by_tenure
    print "average salary by tenure bucket", average_salary_by_bucket

    print
    print "######################"
    print "#"
    print "# MOST COMMON WORDS"
    print "#"
    print "######################"
    print

    for word, count in words_and_counts.most_common():
        if count > 1:
            print word, count
'''