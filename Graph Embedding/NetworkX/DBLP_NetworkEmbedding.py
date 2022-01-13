"""
DBLP DataSet

네트워크 임베딩(Unsupervised Node classification)
Dataset	|#Nodes	|#Edges	    |#Classes	|#Degree	|#Name in Cogdl
DBLP	|51,264	|2,990,443	|60(m)	    |2	        |dblp-ne

이기종 그래프
Dataset	|#Nodes	|#Edges	|#Features	|#Classes	|#Train/Val/Test	|#Degree	|#Edge Type	|#Name in Cogdl
DBLP	|18,405	|67,946	|334	    |4	        |800 / 400 / 2857	|4	        |4	        |gtn-dblp(han-acm)
"""

import pickle
import json

def load_data(Data, path = "../Data/DBLP_네트워크임베딩/") :
    with open(path + Data, 'rb') as f:
        data = f.read()
    data_file = pickle.loads(data)
    return data_file

edges = load_data("edges.pkl")
labels = load_data("labels.pkl")
node_features = load_data("node_features.pkl")

print(f"edges \n type : {type(edges)}\n {edges}\n")
print(f"labels \nlen : {len(labels)}\n type : {type(labels)}\n{labels[1][:10]}\n")
print(f"node_features \nlen : {len(node_features)}\n type : {type(node_features)}\n {node_features[1][:10]}\n")

def Heterogenous(Data, path = "../Data/DBLP_이기종그래프/") :
    with open(path + Data, 'rb') as f :
        data = f.read()
    return data

dblp_cmty = Heterogenous("dblp.cmty")
dblp_ungraph = Heterogenous("dblp.ungraph")

print(f"cmty파일 \n len : {len(dblp_cmty)}\n type : {type(dblp_cmty)}\n{dblp_cmty[:30]}\n")
print(f"ungraph파일 \n len : {len(dblp_cmty)}\n type : {type(dblp_ungraph)}\n{dblp_ungraph[:30]}")


