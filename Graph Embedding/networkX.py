#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from stellargraph import StellarGraph
#객체 생성
g1 = nx.Graph()

#노드 추가
g1.add_node("a")
g1.add_node(1)
g1.add_node(2)
g1.add_node(3)

g1.add_nodes_from([11, 22])

#노드 제거
g1.remove_node(3)

#엣지 추가
g1.add_edge(1, "a")
g1.add_edge(1, 2)
g1.add_edge(1, 22)

g1.add_edges_from([(1, 2), (1, 11)])

#엣지 제거
g1.remove_edge(1, 22)

#그래프 그리기
#nx.draw(g1, with_labels=True, font_weigth="bold")
d = dict(g1.degree)
nx.draw(g1, nodelist = d.keys(), node_size = [v * 100 for v in d.values()], with_labels = True, font_weigth = "bold")
#%%
g = nx.Graph()
g.add_edge("a", "b")
g.add_edge("b", "c")
g.add_edge("c", "d")
g.add_edge("d", "a")
# diagonal
g.add_edge("a", "c")
square = StellarGraph.from_networkx(g)
print(square.info())

