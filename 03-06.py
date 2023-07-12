# 03. 선형 회귀(Linear Regression)
# 03-06. 미니 배치와 데이터 로드(Mini Batch and Data Load)

import torch 
import torch.nn

# 1. 미니 배치와 배치 크기(Mini Batch and Batch Size)
# 데이터 
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 위 데이터의 샘플의 수는 5개입니다. 
# 전체 데이터에 대해서 경사 하강법을 수행하여 학습할 수도 있지만 
# 만약 데이터가 수십만개 이상이라면 전체 데이터에서 경사 하강법을 수행하는 것은 매우 느릴 뿐만 아니라 많은 계산량이 필요하다. 
# 어쩌면 메모리의 한계로 계산이 불가능한 경우도 있을 수 있다. 
# 그렇기 때문에 전체 데이터를 더 작은 단위로 나누어서 해당 단위로 학습하는 개념, 즉 미니 배치가 나오게 되었다. 

# 미니 배치 학습을 하게 되면 미니 배치만큼만 가져가서 미니 배치에 대한 비용(cost)를 계산하고, 경사하강법을 수행한다. 
# 그리고 다음 미니 배치를 가져가서 경사 하강법을 수행하고 반복한다. 
# 이렇게 전체 데이터에 대한 학습이 끝나면 1 epoch가 끝나게 된다. 

# 배치 크기는 보통 2의 제곱수를 사용합니다. 
# 그 이유는 CPU와 GPU의 메모리가 2의 배수이므로 배치 크기가 2의 제곱수일 경우에 데이터 송수신의 효율을 높일 수 있다고 한다. 

# 2. 이터레이션 (Iteration)

# 3. 데이터 로드 (Data Load)
# 파이토치에서는 데이터를 좀 더 쉽게 다룰 수 있도록 유용한 도구로서 데이터셋(Dataset)과 데이터로더(DataLoader)를 제공합니다. 
# 이를 사용하면 미니 배치 학습, 데이터 셔플, 병렬 처리까지 간단히 수행할 수 있습니다. 
# 기본적인 사용 방법은 Dataset을 정의하고, 이를 DataLoader 에 전달하는 것입니다. 

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader     # 데이터로더 

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train, y_train)

# 데이터로더는 기본적으로 2개의 인자를 입력받는다. 
# 데이터셋, 미니 배치의 크기
# 그리고 추가적으로 많이 사용되는 인자로 shuffle 이 있다. 
# shuffle=True를 선택하면 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꾼다 

dataloader = DataLoader(dataset, batch_size = 2, shuffle=True)

# 모델과 옵티마이저 설계 
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# 훈련 진행 
nb_epochs = 20 
for epoch in range(nb_epochs + 1): 
    for batch_idx, samples in enumerate(dataloader): 
        x_train, y_train = samples 
        
        # H(x) 계산 
        prediction = model(x_train)

        # cost 계산 
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 계산 
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx+1, len(dataloader),
            cost.item()
            ))

# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]]) 
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 
