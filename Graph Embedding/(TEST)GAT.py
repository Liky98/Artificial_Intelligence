import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from dgl import DGLGraph
import dgl
from IPython.display import display
from dgl.data import CitationGraphDataset
citeseer = CitationGraphDataset('citeseer')

dir(citeseer)
#%%
# networkx 라이브러리의 오브젝트인 그래프가 citeseer.graph 로 들어가 있다.
# draw 함수를 통해 아래와 같은 끔찍한 이미지를 만들어 낼 수 있다.
import networkx as nx
#%%
nx_G = citeseer.graph.to_undirected()
pos = nx.kamada_kawai_layout(nx_G)
display(nx.draw(nx_G, pos, with_labels=False, node_size = 0.01, node_color='#00b4d9'))
