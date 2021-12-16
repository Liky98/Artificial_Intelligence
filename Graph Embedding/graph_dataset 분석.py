"""  Cora, Pubmed, citeseer  """
""" 분석 및 network X로 변환 """

path = "D:/From_Github/planetoid-master/data"

import torch
from torch_geometric.data import data
#%%
edge_index = torch.tensor([[0, 1, 1, 2],
                        [1, 0, 2, 1]], dtype=torch.long)
#%%
x = torch.tensor([-1], [0], [1], dtype=torch.float)
data = data(x=x, edge_index=edge_index)

#>>> Data(edge_index=[2, 4], x=[3, 1])