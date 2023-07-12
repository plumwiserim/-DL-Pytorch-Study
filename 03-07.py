# 03. 선형 회귀(Linear Regression)
# 03-07. 커스텀 데이터셋(Custom Dataset)

import torch 

class CustomDataset(torch.utils.data.Dataset): 
    def __init__(self): 
    # 데이터셋의 전처리를 해주는 부분 
        pass 

    def __len__(self): 
    # 데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분 
        pass

    def __getitem__(self, idx): 
    # 데이터셋에서 특정 1개의 샘플을 가져오는 함수 
        pass

import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import  DataLoader 

# Dataset 상속 
class CustomDataset(Dataset): 
    def __intt__(self): 
        pass 
