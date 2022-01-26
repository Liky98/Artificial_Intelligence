import torch
""" cuda 버전 : 11.2 """
""" pytorch 버전 : 1.9.1 """
print(torch.__version__)
#%%
#  Returns a bool indicating if CUDA is currently available.
print(torch.cuda.is_available())
#  True

#  Returns the index of a currently selected device.
print(torch.cuda.current_device())
#  0

#  Returns the number of GPUs available.
print(torch.cuda.device_count())
#  1

#  Gets the name of a device.
print(torch.cuda.get_device_name(0))
#  'GeForce GTX 1060'

#  Context-manager that changes the selected device.
#  device (torch.device or int) – device index to select.
print(torch.cuda.device(0))
#%%
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