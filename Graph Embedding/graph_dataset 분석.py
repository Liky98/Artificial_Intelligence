"""  Cora, Pubmed, citeseer  """
""" 분석 및 network X로 변환 """


""" cuda 버전 : 11.2 """
""" pytorch 버전 : 1.9.1 """

path = "D:/From_Github/planetoid-master/data"

import torch
print(torch.__version__)
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)

#from torch_geometric.data import Data

#%%
print('cuda index:', torch.cuda.current_device())

print('gpu 개수:', torch.cuda.device_count())

print('graphic name:', torch.cuda.get_device_name())

cuda = torch.device('cuda')

print(cuda)
#%%
edge_index = torch.tensor([[0, 1, 1, 2],
                        [1, 0, 2, 1]], dtype=torch.long)
#%%
x = torch.tensor([-1], [0], [1], dtype=torch.float)
data = data(x=x, edge_index=edge_index)

#>>> Data(edge_index=[2, 4], x=[3, 1])