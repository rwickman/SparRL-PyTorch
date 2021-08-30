import random

from model import SparRLNet
from conf import *
from agents.agent import Agent, State


class RandomAgent(Agent):

    def __init__(self, args):
        self.args = args
        super().__init__(args, None)

    
    def __call__(self, state) -> int:
        """Make a random sparsification decision based on the state.

        Returns:
            an edge index.
        """
        # Verify batch size is 1
        assert state.subgraph.shape[0] == 1
        
        valid_edge_idxs = self._get_valid_edges(state.subgraph[0])
        
        # Randomly sample an edge to prune
        return random.choice(valid_edge_idxs)