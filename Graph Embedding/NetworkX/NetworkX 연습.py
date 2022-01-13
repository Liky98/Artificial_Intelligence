import networkx as nx
import json
import matplotlib.pyplot as plt
import json_open

path = "../dblp/DBLP최종.json"
data = json_open.json_open(path)

# 방향있는 그래프 생성
G = nx.DiGraph(Data="DBLP")

count = 0

# Node 설정
#G.add_nodes_from(data[0])


for i in data :
    # Edge생성
    G.add_edge(i["_id"], i["venue.sid"], relation = "1")# _id  →  venue.sid   (논문을 투고한 학회 관계)
    G.add_edge(i["venue.sid"], i["_id"], relation = "2")  # venue.sid  → _id   (학회에 포함된 논문 관계)

    for j in i["authors._id"] :
        G.add_edge(i["_id"], j, relation = "3")  # _id  →  authors._id   (논문을 작성한 저자 관계)
        G.add_edge(j, i["_id"], relation = "4")  # authors._id → _id   (저자가 쓴 논문 관계)

    for j in i["references"]:
        G.add_edge(i["_id"], j)  # _id → references (논문에서 인용한 다른 논문 관계)
        G.add_edge(j, i["_id"])  # references → _id (인용한 논문 관계)

    for j in i["keywords"] :
        G.add_edge(i["_id"], j)  # _id → keywords (논문의 키워드 관계)
        G.add_edge(j, i["_id"])  # keywords → _id (해당 키워드에 포함된 논문 관계)

    for j in i["fos"] :
        G.add_edge(i["_id"], j)       # _id → fos (논문의 분야 관계)
        G.add_edge(j, i["_id"])       # fos → _id (해당 분야에 포함된 논문 관계)

    G.add_edge(i["_id"], i["year"])  # _id →  year  (논문의 발행연도 관계)
    G.add_edge(i["year"], i["_id"])  # year → _id (해당연도에 발행된 논문 관계)

    count+=1
    if count == 5 :
        break


d = dict(G.degree) #degree 크기에 따른 NODE size 설정
nx.draw(G, node_size = [v*2 for v in d.values()],with_labels=False)
plt.show()

print(nx.info(G))
len(list(G.nodes))
list(G.edges)
