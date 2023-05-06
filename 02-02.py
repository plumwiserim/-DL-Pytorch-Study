# 02. 파이토치 기초(Pytorch Basic)
# 02-02. 텐서 조작하기(Tensor Manipulationi) 1

# Numpy 로 텐서 만들기(벡터와 행렬 만들기)
import numpy as np 

# 1차원 텐서인 벡터 생성 
# list 생성해서 1차원 array 로 변환 
t = np.array([0., 1., 2., 3., 4., 5., 6.])

# 벡터의 차원과 크기 출력 
print('Rank of t: ', t.ndim)    # Rank of t: 1 
print('Shape of t: ', t.shape)  # Shape of t: (7, )

# 파이토치 텐서 선언(Pytorch Tensor Allocation)
import torch 

# 1차원 텐서인 벡터 생성 
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])

print(t.dim())  # rank.
print(t.shape)  # shape
print(t.size()) # shape

# 벡터의 원소에 접근 
print(t[0], t[1], t[-1]) # tensor(0.) tensor(1.) tensor(6.)

# 브로드캐스팅(BroadCasting)
'''
행렬의 덧셈과 뺄셈을 수행할 때에는 두 행렬의 크기가 같아야한다. 
하지만, 딥러닝을 하게 되면 불가피하게 다른 행렬 또는 텐서에 대해서 사칙연산을 수행할 경우가 생긴다. 
이를 위해 자동으로 크기를 맞춰서 연산을 수행하게 만드는 기능을 제공한다. 
'''

# 같은 크기일 때 행렬 연산 
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# Vector + Scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3])
print(m1 + m2)  # tensor([[4., 5.]])

m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)  # tensor([4., 5.], [5., 6.])

# 행렬 곱셈 
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print(m1.matmul(m2))

# element-wise 곱셈
print(m1 * m2)
print(m1.mul(m2))

# 평균 
t = torch.FloatTensor([1, 2])
print(t.mean()) # tensor(1.5000)

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.mean()) # 2.5000

# 행렬에서 첫번째 차원(dim=0)은 행을 의미
# 행을 제외하고 열을 기준으로 평균을 구하겠다는 의미 
print(t.mean(dim=0))    # tensor([2, 3])

# 마지막 차원(dim=-1)은 열을 의미
# 열을 제외하고 행을 기준으로 평균을 구하겠다는 의미 
print(t.mean(dim=-1))   # tensor([1.5000, 3.5000])

# 덧셈 
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.sum())          # tensor(10.)
print(t.sum(dim=0))     # tensor([4., 6.])
print(t.sum(dim=1))     # tensor([3., 7.])
print(t.sum(dim=-1))    # tensor([3., 7.])

# 최대(Max), Argmax
print(t.max())  # Returns one value: max    # tensor(4.)
print(t.max(dim=0)) # Returns two values: max and argmax    # tensor([3., 4.]), tensor([1, 1])
print('Max: ', t.max(dim=0)[0])     # Max: tensor([3., 4.])
print('Argmax: ', t.max(dim=0)[1])  # Argmax: tensor([1, 1])
