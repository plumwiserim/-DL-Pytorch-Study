# 03. 선형 회귀(Linear Regression)
# 03-02. 자동 미분(Autograd)

# 자동미분(Autograd)이란? 
# 경사하강법 코드에서 
# requires_grad = True
# backward()

# 자동미분(Autograd) 실습 
import torch 

# 값이 2인 임의의 스칼라 텐서에 대한 기울기를 저장하겠다. 
w = torch.tensor(2.0, requires_grad=True)

# 임의로 2w**2 + 5 라는 수식 세우기 
y = w**2
z = 2* y + 5

# w에 대해 미분하기 
z.backward()

# w에 대해 미분한 값이 저장되었는지 확인하기 
print('수식을 w로 미분한 값: {}'.format(w.grad))
