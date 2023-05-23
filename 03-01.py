# 03. 선형 회귀(Linear Regression)
# 03-01. 선형 회귀(Linear Regression)

import torch

# 1. Train Data Set 구성 
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 2. 가설(Hypothesis) 수립 
# y = Wx + b 
# H(x) = Wx + b 

# 3. 비용 함수(Cost Function)
# Cost Function = Loss Function = Error Function = Objective Function 

# 4. Optimizer - 경사하강법(Gradient Descent)

# 5. Pytorch 로 선형 회귀 구현 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

# 현재 실습하고 있는 파이썬 코드를 재실행해도 다음에도 같은 결과가 나오도록 랜덤 시드(random seed)를 준다. 
torch.manual_seed(1)

# 변수 선언
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 가중치와 편향의 초기화 
# 가중치를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시 
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 가설 세우기 
hypothesis = x_train * W + b 

# 비용 함수 선언하기 
# Cost Function, 잔차 제곱의 평균 
cost = torch.mean((hypothesis - y_train)**2)

# 경사하강법 구하기 
optimizer = optim.SGD([W, b], lr=0.01)

# gradient를 0으로 초기화
optimizer.zero_grad()

# 비용 함수를 미분하여 gradient 계산 
cost.backward()

# W와 b 업데이트 
optimizer.step()


# 전체 코드 
# 데이터 
x_train = torch.FloatTensor([1], [2], [3])
y_train = torch.FloatTensor([2], [4], [6])

#  모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 1999 # 원하는만큼 경사 하강법을 반복 
for epoch in range(nb_epochs + 1): 
    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산 
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0: 
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))

# optimizer.zero_grad() 가 필요한 이유
# 파이토치는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적시키는 특징이 있다. 
import torch 
w = torch.tensor(2.0, requires_grad=True)

np_epochs = 20
for epoch in range(nb_epochs + 1): 
    z = 2*w
    z.backward()
    print('수식을 w로 미분한 값: {}'.format(w.grad))

# torch.manual_seed()를 하는 이유 
import torch 

torch.manual_seed(3)
print('랜덤 시드가 3일 때')
for i in range(1, 3):
    print(torch.rand(1))

torch.manual_seed(5)
print('랜덤 시드가 5일 때')
for i in range(1, 3):
    print(torch.rand(1))

# requires_grad 라는 속성을 True로 설정하면
# 자동 미분 기능 적용 
# 선형 회귀부터 신경망과 같은 복잡한 구조에서 파라미터들에 모두 적용 가능 
# requires_grad = True 가 적용된 텐서에 연산을 하면, 계산 그래프가 생성되며 
# backward 함수를 호출하면 그래프로부터 자동으로 미분이 계산된다. 
