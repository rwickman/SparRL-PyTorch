from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.semi_supervised import LabelPropagation
# from graph import Graph


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
        min_com_val = min(self._true_communities)
        # Make it start at zero
        for i in range(len(self._true_communities)):
            self._true_communities[i] -= min_com_val
        
    def ARI_louvain(self):
        """Compute the Adjusted Rand Index for Louvain."""    
        partition = self._graph.louvain()
        # print("ARI Louvain: ", adjusted_rand_score(list(partition.values()), self._true_communities))
        # print("list(partition.values())", list(partition.values()))
        # print("self._true_communities", self._true_communities)
        return adjusted_rand_score(list(partition.values()), self._true_communities)

