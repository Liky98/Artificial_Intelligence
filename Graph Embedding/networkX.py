#%%
from IPython.core.display import display, HTML
display(HTML("<style>.container {width:90% !important;}</style>"))
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

#%%
import networkx as nx
import matplotlib.pyplot as plt

# 그래프 생성
G = nx.DiGraph()
G.add_nodes_from([1, 2, 3, 4, 5])
G.add_edges_from([(1, 2), (2, 1), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2,4), (4, 2),
                 (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3)])
degree = nx.degree(G)
print(degree)
#%%
nx.draw(G,node_size=[500 + v[1]*500 for v in degree], with_label0s=True)

#%%
import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(graph):

    # graph 객체에서 노드들을 뽑아 nodes 에 저장한다
    nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])

    # 그래프 객체를 하나 만든다
    G=nx.Graph()

    # add_node() 를 통해 그래프에 노드를 입력한다
    for node in nodes:
        G.add_node(node)

    # add_edges() 를 통해 노드간의 엣지를 지정해 준다
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # 그리려는 그래프의 속성을 설정해 준다
    nx.draw_networkx_nodes(G,pos,node_size=5)
    nx.draw_networkx_edges(G,pos,width=1)
    nx.draw_networkx_labels(G,pos,font_size=10,font_family='sans-serif')

    # 그래프 그림을 파일로 저장한다
    plt.axis('off')
    plt.savefig("graph.png", dpi=1000) # 그림 파일 크기 지정
    plt.savefig("weighted_graph.png") # png 파일로 저장

    # 그래프 레이아웃을 저장한다
    pos = nx.shell_layout(G)
    nx.draw(G, pos)

    # matplotlib 모듈을 이용해 그래프를 시각화 한다.
    plt.show()

# 그리고자 하는 그래프 예제
graph = [(20, 21),(21, 22),(22, 23), (23, 24),(24, 25), (25, 20)]
draw_graph(graph)