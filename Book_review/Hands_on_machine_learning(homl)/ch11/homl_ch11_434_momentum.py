#page434 모멘텀 최적화 momentum optimization

#모멘텀 최적화는 이전 그레이디언트가 얼마였는지를 상당히 중요하게 생각핟낟.
# 매 반복시 현재 그레이디언트를 모멘텀벡터(m)에 더하고 이 값을 빼는 방식으로 가중치를 갱신시킨다. 
# 
# 케라스에서 모멘텀 최적화를 구현하는 방법 
# cara pakai momentum optimizer di keras sangat gampang. 
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

#436네스트로프 가속 경사 -- Nesterov accelerated gradient(NAG)
#경사에서는 항상 이동중이니까 지금당장 현재의 위치가 아닌 그 방향의 조금 앞인 위치를 계산해 주는 방법이 있다. 
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

#438 AdaGrad
optimizer = keras.optimizers.Adagrad(lr=0.001)
#Adagrad 알고리즘은 가장 가파른 차원을 따라서 그레이디언트 백터의 스케일을 감소시켜 이 문제를 해결한다. 

#440 RMSprop --
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
#AdaGrad는 너무 빨리 느려져서 전역 최적점에 수렴하지 못하는 위험이 있습니다. 
# RMSprop알고리즘은 가장 최근 반복에서 비롯된 그래이디언틈난 누적함으로써 이 문제를 해결했다ㅣ 

#440 Adam과 Nadam의 최전화 
#적응적 모멘트 추정 adaptive momentum estimation을 의미하는 adam 모멘텀 최적화와 RMSProp 을 합친것이다.  
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
