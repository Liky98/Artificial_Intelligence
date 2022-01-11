from dgl.data import CitationGraphDataset
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle as pkl  # serialize를 위한 바이너리 프로토콜 구현. 클래스를 통째로 저장 및 불러올 때 유용하다.
import networkx as nx  # 그래프를 다루기 위한 파이썬 패키지
import scipy.sparse as sp  # 행렬의 대부분의 원소가 0인 sparse matrix를 만들기 위해 불러온다
from scipy.sparse.linalg.eigen.arpack import eigsh  # 대칭행렬에서 k개의 eigen vector와 eigen value를 가져옴
import sys
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import easydict
import datasets

path = '/Graph Embedding/planetoid-master/data'
datasets.load_data_KIHOON("siteseer")
print("a")

#%%
# easydict를 이용해 hyperparameter 및 사용할 데이터를 미리 정의한다.
args = easydict.EasyDict({'dataset': 'pubmed',
                          'model': 'gcn',
                          'learning_rate': 0.01,
                          'epochs': 400,
                          'hidden': 16,
                          'dropout': 0.5,
                          'weight_decay': 5e-4,
                          'early_stopping': 10,
                          'max_degree': 3
                          })

names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
objects = []
path = '/Graph Embedding/planetoid-master/data'
for i in range(len(names)):  # 각 파일들을 열고 pickle모듈을 이용해 serialized된 데이터를 파싱한다.
    with open(path+"/ind.{}.{}".format(args.dataset, names[i]), 'rb') as f:
        objects.append(pkl.load(f, encoding='latin1'))



os.environ['KMP_DUPLICATE_LIB_OK']='True'

citeseer = CitationGraphDataset('citeseer')

dir(citeseer)

# networkx 라이브러리의 오브젝트인 그래프가 citeseer.graph 로 들어가 있다.
# draw 함수를 통해 아래와 같은 끔찍한 이미지를 만들어 낼 수 있다.

nx_G = citeseer.graph.to_undirected()

pos = nx.kamada_kawai_layout(nx_G)

nx.draw(nx_G, pos, with_labels=False, node_size = 0.01, node_color='#00b4d9')

plt.show()

