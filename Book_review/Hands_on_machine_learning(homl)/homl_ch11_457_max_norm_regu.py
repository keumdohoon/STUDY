#page 457 MAX-norm regularization
#맥스 노름 규제는 전체 손실 함수에 규제 손실 항을 추가하지 않습니다. 
#r을 줄이면 규제의 양이 증가하여 과대적합을 감소시키는 데 도움이 됩니다. 
#맥스 노름 규제는 불안전한 그레이디언트 문제를 완화하는데 도움을 줄 수 있습니다.

keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal",
                           kernel_constraint=keras.constraints.max_norm(1.))

#매 훈련 반복이 끝난 후 모델의 fit() 매서드가 층의 가중치와 함께 max_norm()이 반환한 
#객체를 호출하고 스케일이 조정된 가중치를 반환받습니다. 
# 이 값을 사용하여 층의 가중치를 바꿉니다. 


