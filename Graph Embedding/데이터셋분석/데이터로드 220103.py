import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 데이터 얻어오는 함수
def load_data_KIHOON(dataset): #citeseer / cora / pubmed  택1
    data_path = 'data'
    suffixs = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph'] #접미사(마지막에 붙는것들)
    objects = [] #객체

    for suffix in suffixs:
        file = data_path + '/ind.%s.%s'%(dataset, suffix)  #파일 경로 설정
        print(file)
        objects.append(pickle.load(open(file, 'rb'), encoding='latin1')) #리스트에 추가함. pickle은 텍스트데이터X 바이너리파일로 저장.

    x, y, allx, ally, tx, ty, graph = objects
    x, allx, tx = x.toarray(), allx.toarray(), tx.toarray()
    return x, y, allx, ally, tx, ty, graph

#%%
data_cora = load_data_KIHOON('cora')
x, y, allx, ally, tx, ty, graph = data_cora
"""
x.shape
Out[23]: (140, 1433)
allx.shape
Out[24]: (1708, 1433)
y.shape
Out[25]: (140, 7)
ally.shape
Out[26]: (1708, 7)
"""
#%%
plt.plot(x)
plt.title("x")
plt.show()
plt.plot(y)
plt.title("y")
plt.show()
plt.plot(allx)
plt.title("allx")
plt.show()
plt.plot(ally)
plt.title("ally")
plt.show()
#%%
test = pd.read_pickle('data/ind.cora.graph')

#%%
import networkx as nx
#nx.draw(graph, with_labels=True, font_weigth="bold")
#graph1 = nx.DiGraph()
nx.draw(test)
plt.show()

#%%
