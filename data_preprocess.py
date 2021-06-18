import pandas as pd
import numpy as np
import networkx as nx
# from datetime import datetime

class DataProcessor:
    def __init__(self, path, data_col, is_directed=True, frequent_node=False, th=10,
            connected_component=False, filter=True):
        super(DataProcessor, self).__init__()
        self.data = None
        self.matrix = None
        self.filtered_matrix = None
        self.G = None
        self.data_col = data_col
        self.is_directed = is_directed
        self.frequent_node = frequent_node
        self.th = th
        self.connected_component = connected_component
        self.filter = filter

def load_data(self):
    print('Loading data...')
    self.data = pd.read_csv(self.path, sep=r"\s+", header=None)
    # data.columns = ['source', 'target', 'time']
    # self.data.columns = ['source', 'target', 'weight', 'time']
    self.data.columns = self.data_col
    self.data['time'] = pd.to_datetime(self.data['time'], unit='s')
    # self.data['date'] = self.data['time'].apply(lambda x: x.date())

def construct_matrix(self):
    print('Constructing temporal network matrix...')
    if self.is_directed:
        self.G = nx.DiGraph()
    else:
        self.G = nx.Graph()
    for ind, row in self.data.iterrows():
        source, target = row[['source', 'target']]
        if self.G.has_edge(source, target):   
            self.G[source][target]['occurence'] += 1
        else:
            self.G.add_edge(source, target, occurence=1)
    if self.frequent_node:
        frequent_nodes = [x for x in self.G.nodes() if self.G.degree(weight='occurence')[x]>self.th]
        self.G = self.G.subgraph(frequent_nodes)
    if self.connected_component:
        if self.is_directed:
            largest = max(nx.strongly_connected_components(self.G), key=len)
        else:
            largest = max(nx.connected_components(self.G), key=len)
        self.G = self.G.subgraph(largest)
        nodes = list(self.G.nodes())
        self.data = self.data[self.data['source'].isin(nodes) & self.data['target'].isin(nodes)].reset_index(drop=True)
    nodes = list(self.G.nodes())
    num_nodes = len(nodes)
    time_span = (self.data['time'].iloc[-1] - self.data['time'].iloc[0]).days
    self.matrix = np.zeros((len(nodes)**2, time_span + 1))
    for ind, row in self.data.iterrows():
        self.matrix[nodes.index(row['source'])*num_nodes+nodes.index(row['target']), (row['time'] - self.data['time'].iloc[0]).days] += 1
    
def filter_links(self):
    print('Filtering links...')
    col = self.matrix.shape[1]
    num_nonzero = np.count_nonzero(self.matrix)
    # Sorted based on number of nonzero elements
    count_matrix = np.hstack((self.matrix, np.count_nonzero(self.matrix, axis=1)[:, None]))
    count_matrix = count_matrix[count_matrix[:, -1].argsort()[::-1]]
    matrix = count_matrix[:, :-1]
    # Links with activity rate higher than 0.1
    active_link_ind = np.where(count_matrix[:, -1] > 0.1 * col)[0]
    activity_portion = np.cumsum(count_matrix[:, -1]) / num_nonzero
    # Filtered network at least has 1000 rows or keeps 60% activities
    if len(active_link_ind) >= 1000 or activity_portion[active_link_ind[-1]] >= 0.6:
        rows_to_keep = active_link_ind
    else:
        rows_to_keep = np.arange(np.where(activity_portion >= 0.6)[0][0] + 1)
    self.filtered_matrix = matrix[rows_to_keep, :]
