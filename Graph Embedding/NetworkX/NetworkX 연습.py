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
        G.add_edge(i["_id"], j,relation = "5")  # _id → references (논문에서 인용한 다른 논문 관계)
        G.add_edge(j, i["_id"],relation = "6")  # references → _id (인용한 논문 관계)

    for j in i["keywords"] :
        G.add_edge(i["_id"], j,relation = "7")  # _id → keywords (논문의 키워드 관계)
        G.add_edge(j, i["_id"],relation = "8")  # keywords → _id (해당 키워드에 포함된 논문 관계)

    for j in i["fos"] :
        G.add_edge(i["_id"], j,relation = "9")       # _id → fos (논문의 분야 관계)
        G.add_edge(j, i["_id"],relation = "10")       # fos → _id (해당 분야에 포함된 논문 관계)

    G.add_edge(i["_id"], i["year"],relation = "11")  # _id →  year  (논문의 발행연도 관계)
    G.add_edge(i["year"], i["_id"],relation = "12")  # year → _id (해당연도에 발행된 논문 관계)

    count +=1
    if count %100 == 0 :
        print(f"{count}번째 작업중입니다.")

# 그래프 저장
nx.write_gpickle(G,"DBLP.gexf")
nx.write_edgelist(G, "DBLP_Edge")

#그래프 읽기
test = nx.read_gpickle("DBLP.gexf")
