#lr  가 어떤 방식으로 적용 되는지. 

weight = 0.5
input = 0.5
goal_prediction = 0.8
# lr = 0.001 #Error : 1.0799505792475652e-27/tPrediction : 0.7999999999999672
# lr = 0.1    #Error : 0.0024999999999999935/tPrediction : 0.7500000000000001
# lr = 1     #Error : 0.20249999999999996/tPrediction : 1.25
lr = 0.0001 #Error : 0.24502500000000604/tPrediction : 0.30499999999999394

for iteration in range(1101):#1100번 돌려라 

    prediction = input * weight
    error = (prediction - goal_prediction) **2

    print("Error : " +str(error) + "/tPrediction : " + str(prediction))

    up_prediction = input* (weight +lr)
    up_error = (goal_prediction - up_prediction) **2

    down_prediction = input* (weight - lr)
    down_error = (goal_prediction - down_prediction) **2

    if(down_error < up_error) :
        weight = weight  - lr
    if(down_error > up_error) :
        weight = weight  + lr 