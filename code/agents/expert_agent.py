import networkit as nk
from networkit.sparsification import LocalDegreeScore, ForestFireScore
import torch
import random
from agents.agent import Agent

class ExpertAgent(Agent):
    def __init__(self, args, graph):
        super().__init__(args)
        self.graph = graph
        self._init_edge_scores()

    def _init_edge_scores(self):
        G = nk.nxadapter.nx2nk(self.graph.get_G())
        G.indexEdges()
        if self.args.expert_spar == "eff":
            eff_scores = ForestFireScore(G, 0.6, 5.0)
            eff_scores.run()
            edges_scores = eff_scores.scores()
        else:
            local_deg_scores = LocalDegreeScore(G)
            local_deg_scores.run()
            edges_scores = local_deg_scores.scores()
        self._edge_score_dict = {}
        for i, edge in enumerate(self.graph._G.edges()):
            self._edge_score_dict[edge] = edges_scores[i]

    def edge_score(self, edge):
        if edge not in self._edge_score_dict:
            raise Exception(f"Edge score for {edge} not found!")
        else:
            return self._edge_score_dict[edge]

    def find_min_score_edge(self, subgraph: torch.Tensor):
        """Find the edge with the minimum edge score to prune."""

        # Iterate over all the edges to find max
        edge_idxs = self._get_valid_edges(subgraph).tolist()
        assert len(edge_idxs) > 0

        # Prevent predicting same index
        #random.shuffle(edge_idxs)
        
        min_score = None
        min_edge = None
        for edge_idx in edge_idxs:
        
            edge = [subgraph[2*edge_idx], subgraph[2*edge_idx + 1]]

            score = self.edge_score((int(edge[0]) - 1, int(edge[1]) - 1))
            if min_score is None or min_score > score:
                min_score = score
                min_edge = edge_idx

        return min_edge

    def __call__(self, state, mask=None) -> int:
        """Make a sparsification decision based on the state.

        Returns:
            an edge index.
        """
        min_edge = self.find_min_score_edge(state.subgraph[0])

        return min_edge