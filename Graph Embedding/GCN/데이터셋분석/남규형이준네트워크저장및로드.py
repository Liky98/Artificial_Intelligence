import networkx as nx
import json

class Network:
    def __init__(self, dirpath=None, year=2020):
        self.G = nx.DiGraph()

        self.year = year
        if dirpath is not None:
            self.read_data(dirpath)

    def read_data(self, dirpath):
        papers = json.load(open(dirpath))
        print(len(papers))
        papers = [paper for paper in papers if 2008 <= int(paper['year']) <= self.year]

        for paper in papers:
            self.G.add_edge(paper['_id'], paper['venue']['_id'])  # paper - venue 연결
            self.G.add_edge(paper['venue']['_id'], paper['_id'])  # venue - paper 연결
            for author in paper['authors']:
                self.G.add_edge(paper['_id'], author['_id'])  # paper - author 연결
                self.G.add_edge(author['_id'], paper['_id'])  # author - paper 연결
            for reference in paper['references']:
                self.G.add_edge(paper['_id'], reference)  # paper - referece for other paper 연결

        for paper in papers:
            self.G.nodes[paper['_id']]['type'] = 'paper'
            self.G.nodes[paper['_id']]['year'] = paper['year']
            self.G.nodes[paper['venue']['_id']]['type'] = 'venue'
            for author in paper['authors']:
                self.G.nodes[author['_id']]['type'] = 'author'
            for reference in paper['references']:
                self.G.nodes[reference]['type'] = 'paper'

net = Network(dirpath = "./../data/network_info.json", year=2013).G
nx.write_gpickle(net, "./../data/network_mini.gpickle")

print(f"Number of nodes {net.number_of_nodes()} and number of edges {net.number_of_edges()} in network.")

#%%
net = nx.read_gpickle(network_path + f"{network_name}.gpickle")

print(f"Number of nodes {net.number_of_nodes()} and number of edges {net.number_of_edges()} in network.")