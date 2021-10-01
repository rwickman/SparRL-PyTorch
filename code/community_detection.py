from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.semi_supervised import LabelPropagation
import networkx as nx 
import community
# from graph import Graph
import torch


class CommunityDetection:
    def __init__(self, args, graph):
        self._args = args
        self._graph = graph
        self._true_communities = []
        self._load_communities()

    def _load_communities(self):
        """Load communities form label file."""
        with open(self._args.com_labels) as f:
            for line in f:
                _, community = line.split()
                self._true_communities.append(int(community))
        
        
        # Make labels start at zero
        # min_com_val = min(self._true_communities)
        # for i in range(len(self._true_communities)):
        #     self._true_communities[i] = self._true_communities[i] -  min_com_val
        
        # self._true_com_dict = self._create_com_dict(self._true_communities)

        print("|V|", self._graph.num_nodes)
        print("self._true_communities[i]", len(self._true_communities))

    def ARI_louvain(self):
        """Compute the Adjusted Rand Index for Louvain."""    
        partition = self._graph.louvain()
        return adjusted_rand_score(self._true_communities, list(partition.values()))

    def NMI_louvian(self):
        partition = self._graph.louvain()
        return normalized_mutual_info_score(self._true_communities, list(partition.values()))

    def average_clustering(self):
        return nx.average_clustering(self._graph._G)

    def modularity(self):
        partition = self._graph.louvain()
        return community.modularity(partition,self._graph._G)
    
    def is_edge_same_com(self, edge):
        
        return self._true_communities[edge[0] - 1] == self._true_communities[edge[1] - 1] 


    # def fix_labels(self, found_labels):
    #     com_dict = self._create_com_dict(found_labels)

    #     # For each community, count the number of nodes that belong to the same ground truth community
         

    # def _create_com_dict(self, labels):
    #     # Create clusters
    #     clusters = {}
    #     for i, label in enumerate(found_labels):
    #         if label not in clusters:
    #             clusters[label] = [] 
    #         clusters[label].append(i)
        
    #     return clusters


        