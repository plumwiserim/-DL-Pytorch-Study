# 02. 파이토치 기초(Pytorch Basic)
# 02-02. 텐서 조작하기(Tensor Manipulationi) 2

import numpy as np
import torch

t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape) # torch.Size([2, 2, 3])

# 3차원 텐서 --> 2차원 텐서로 변경
print(ft.view([-1, 3])) # ft라는 텐서를 (?, 3)의 크기로 변경 
print(ft.view([-1, 4]).shape)   # torch.Size([4, 3])

# 2차원 텐서 --> 3차원 텐서로 변경 
print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape()) # torch.Size([4, 1, 3])

# 스퀴즈(Squeeze) - 1인 차원을 제거
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape) # torch.Size([3, 1])

print(ft.squeeze()) # tensor([0., 1., 2.])
print(ft.squeeze().shape)   # torch.Size([3])

# 언스퀴즈(Unsqueeze) - 특정 위치에 1인 차원을 추가 
ft = torch.Tensor([0, 1, 2])
print(ft.shape) # torch.Size([3])

# 0번쨰 차원에 1인 차원 추가 
print(ft.unsqueeze(0))          # tensor([[0., 1., 2.]])
print(ft.unsqueeze(0).shape)    # torch.Size([1, 3])

print(ft.unsqueeze(1))          # tensor([[0.], [1.], [2.]])
print(ft.unsqueeze(1).shape)    # torch.Size([3, 1])

lt = torch.LongTensor([1, 2, 3, 4])

# Long --> float 
print(lt.float())

# Byte type Tensor 
bt = torch. ByteTensor([True, False, False, True])
print(bt)   # tensor([1, 0, 0, 1], dtype=torch.unit8)

print(bt.long())    # byte --> long 
print(bt.float())   # byte --> float

# 텐서 연결하기(concatenate)
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

# 첫번째 차원(dim=0)을 늘리기 
print(torch.cat([x, y], dim=0)) # tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

# 두번째 차원(dim=1)을 늘리기 
print(torch.cat([x, y], dim=1)) # tensor([[1, 2, 5, 6], [3, 4, 7, 8]])

# 스택킹(Stacking)
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))   # tensor([[1., 4.], [2., 5.], [3., 6.]])

# 스택킹은 아래 연산을 축약한 것 
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))

print(torch.stack([x, y, z], dim=1))    # tensor([[1., 2., 3.], [4., 5., 6.]])

x = torch.FloatTensor([[0, 1, 2], [2, 1,  0]])

# 입력 텐서와 동일한 크기(shape)이지만 값이 1로만 채워진 텐서 생성 
print(torch.ones_like(x))

# 입력 텐서와 동일한 크기(shape)이지만 값이 0으로만 채워진 텐서 생성
print(torch.zeros_like(x))

x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.))    # tensor([[2, 4], [6, 8]])
print(x)            # tensor([[1, 2], [3, 4]])

# 연산 덮어쓰기
print(x.mul_(2.))   # tensor([[2, 4], [6, 8]])
print(x)            # tensor([[2, 4], [6, 8]])
