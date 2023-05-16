# 03. 선형 회귀(Linear Regression)
# 03-01. 선형 회귀(Linear Regression)

import torch

# Train Data Set 구성 
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 가설(Hypothesis) 수립 
# y = Wx + b 
# H(x) = Wx + b 

# 비용 함수(Cost Function)
# Cost Function = Loss Function = Error Function = Objective Function 

# Optimizer - 경사하강법(Gradient Descent)

# Pytorch 로 선형 회귀 구현 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

# 현재 실습하고 있는 파이썬 코드를 재실행해도 다음에도 같은 결과가 나오도록 랜덤 시드(random seed)를 준다. 
torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 가중치를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시 
W = torch.zeros(1, requires_grad=True)

b = torch.zeros(1, requires_grad=True)

hypothesis = x_train * W + b 

# Cost Function, 잔차 제곱의 평균 
cost = torch.mean((hypothesis - y_train)**2)

optimizer = optim.SGD([W, b], lr=0.01)

# gradient를 0으로 초기화
optimizer.zero_grad()

# 비용 함수를 미분하여 gradient 계산 
cost.backward()

# W와 b 업데이트 
optimizer.step()
