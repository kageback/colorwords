import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from copy import deepcopy

class PopulationGraph:
    def __init__(self, agents=[], n_clusters=1, cluster_connections=None, cluster_distribution=None, epsilon=0.01, csv=None):
        self.n_agents = len(agents)
        self.agents = agents
        self.n_clusters = n_clusters
        self.cluster_connections = cluster_connections
        self.cluster_distribution = cluster_distribution
        self.epsilon = epsilon
        self.csv = csv

        if self.n_agents > 0:
            self.graph, self.labels = self.generate_graph(self.n_agents, self.n_clusters, self.cluster_connections, self.cluster_distribution)
        if csv is not None:
            self.graph, self.labels = self.generate_graph_from_csv(csv)

    def generate_graph(self, n_agents=2, n_clusters=1, cluster_connections=None, cluster_distribution=None):
        graph = np.zeros([n_agents, n_agents])
        if cluster_distribution is None:
            n_members = int(np.floor(n_agents / n_clusters))
            labels = []
            # Generate clusters
            for i in range(n_clusters):
                labels += [i] * n_members

        if cluster_connections is None:
            prob = 1 / (n_members - 1) # Can't talk to oneself
            for a in range(n_agents):
                current_cluster = labels[a]
                members = [i for i, x in enumerate(labels) if x == current_cluster]
                graph[a, members] = prob
                graph[a, a] = 0

        return graph, labels

    def generate_graph_from_csv(self, csv):
        self.df = pd.read_csv(csv)
        self.graph = csv.values

    def get_graph(self):
        return self.graph

    def get_agent(self, i):
        return deepcopy(self.agents[i])

    def sample_neighbor(self, agent, batch_size=1):
        return np.random.choice(np.arange(self.n_agents), p=self.graph[agent, :], size=batch_size).tolist()


    def draw_pairs(self, n):
        training_pairs = []
        for _ in range(n):
            speaker = np.random.randint(0, self.n_agents)
            listener = self.sample_neighbor(speaker)
            training_pairs.append((speaker, listener))
        return training_pairs


    def plot_graph(self, save_string=None):
        rows, cols = np.where(self.graph > 0)
        values = self.graph[rows, cols]
        edges = zip(rows.tolist(), cols.tolist(), values.tolist())
        gr = nx.Graph()
        gr.add_weighted_edges_from(edges)
        pos = nx.spring_layout(gr)
        nx.draw_networkx(gr, pos, node_size=500)
        labels = nx.get_edge_attributes(gr, 'weight')
        nx.draw_networkx_edge_labels(gr, pos, edge_labels=labels)
        if save_string is None:
            plt.show()
        else:
            plt.savefig(save_string)



