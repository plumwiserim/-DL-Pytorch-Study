# 03. 선형 회귀(Linear Regression)
# 03-04. nn.Module로 구현하는 선형회귀 

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim = 1, output_dim = 1
model = nn.Linear(1, 1)

# model에 저장되어있는 가중치 W와 편향 b 출력
print(list(model.parameters()))

# 두 값 모두 현재는 랜덤 초기화되어있다. 
# 그리고 학습의 대상이므로 requires_grad=True로 설정되어있다. 
# [Parameter containing: tensor([[0.5153]], requires_grad=True), 
#  Parameter containing: tensor([-0.4414], requires_grad=True)]

# optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 전체 훈련 데이터에 대해 경사 하강법을 2000회 반복 
nb_epochs = 2000
for epoch in range(nb_epochs+1): 
    # H(x) 계산 
    prediction = model(x_train)

    # cost 계산 
    cost = F.mse_loss(prediction, y_train)  # <-- 파이토치에서 제공하는 평균 제곱 오차 함수 

    # cost로 H(x) 개선 
    # gradient를 0으로 초기화 
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산 
    # backward 연산 
    cost.backward()
    # W와 b 업데이트 
    optimizer.step()

    # 100번마다 로그 출력 
    if epoch % 100 == 0: 
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# 예측 결과 확인 
# 임의의 입력 4를 선언
new_var = torch.FloatTensor([[4.0]])
# 입력한 값 4에 대해서 예측값 y를 리턴받아서 pred_y에 저장
# forward 연산
pred_y = model(new_var)
# y = 2x 이므로 입력이 4라면 y가 8에 가까운 값이 나와야 제대로 학습이 된 것 
print("훈련 후 입력이 4일 때 예측값: ", pred_y)

# 학습 후 가중치와 편향 값 출력
print(list(model.parameters()))

# 다중 선형 회귀 구현 
import torch 
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터 
x_train = torch.FloatTensor([[73, 80, 75], 
                             [93, 88, 93], 
                             [89, 91, 80], 
                             [96, 98, 100], 
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=3, output_dim=1
model = nn.Linear(3, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs+1): 
    # H(x) 계산 
    # model(x_train)은 model.forward(x_train)과 동일함. 
    prediction = model(x_train)
    
    # cost 계산 
    cost = F.mse_loss(prediction, y_train)  # <-- 파이토치에서 제공하는 평균 제곱 오차 함수 

    # cost 로 H(x) 개선
    # gradient를 0으로 초기화 
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산 
    cost.backward()
    # 가중치와 편향 업데이트 
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0: 
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# 학습 결과 확인 
# 임의의 입력 [73, 80, 75]를 선언
new_var = torch.FloatTensor([[73, 80, 75]])
# 입력한 값 [73, 80, 75]에 대해서 에측값 y를 리턴받아서 pred_y에 저장 
pred_y = model(new_var)
print('훈련 후 입력이 73, 80, 75일 때 예측값: ', pred_y)

# 훈련 후 가중치와 편향 출력
print(list(model.parameters()))
