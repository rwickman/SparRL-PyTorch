import numpy as np

from reward_manager import RewardManager
from conf import *

@dataclass
class State:
    subgraph: torch.Tensor
    global_stats: torch.Tensor
    local_stats: torch.Tensor

@dataclass
class Experience:
    state: State
    next_state: State
    action: int
    reward: float


class Environment:
    """Handles environment related tasks/management during an episode."""
    def __init__(self, args, graph, agent):
        self.args = args
        self.graph = graph
        self.agent = agent

        self.T_lam = self.args.T_lam

        # Setup the RewardManager to manage rewards 
        self._reward_manager = RewardManager(args, graph)
        self._reward_manager.setup()    

    def reset(self):
        """Reset environment for next episode."""
        # Restore pruned edges to graph
        for pair in self._removed_edges:
            self._graph.add_edge(pair[0], pair[1])

        self._removed_edges = set()

        self.agent.reset()

    def _sample_T(self) -> int:
        T = np.random.poisson(self.T_lam)
        return np.clip(T, 1, self.args.max_prune)

    def sample_subgraph(self, subgraph_len: int):
        """Sample a subgraph of edges."""
        sampled_edges = self.graph.sample_edges(size)

        return sampled_edges

    def prune_edge(self, edge_idx: int, subgraph: torch.Tensor):
        """Prune an edge from the subgraph and the graph."""
        edge = [subgraph[2*edge_idx], subgraph[2*edge_idx + 1]]

        # Shift back to original node ids
        edge = [edge[0] - 1, edge[1] - 1]

        # Remove from graph
        self._removed_edges(edge)
        self._graph.del_edge(edge[0], edge[1])


    def create_state(self, subgraph_len: int, T: int, t: int):
        """Create the current state for the episode."""

        # Sample random edges
        subgraph = sample_subgraph(subgraph_len)
        
        # Create global statistics
        prune_left = np.log(T-t + 1)
        num_edges_left = np.log(self._graph.get_num_edges() + 1)
        global_stats = torch.tensor([[prune_left, num_edges_left]], device=device, dtype=torch.float32)

        # Create local node statistics
        node_ids = []
        for e in subgraph:
            node_ids.append(e[0])
            node_ids.append(e[1])

        # Get node degrees
        degrees = self.graph.degree(node_ids)
        if isinstance(degrees, tuple):
            # Must be directed graph
            degs = torch.tensor(list(zip(degrees[0], degrees[1])), device=device)
        else:
            degs = torch.tensor(list(zip(degrees, [0] * len(degrees))), device=device)
        
        local_stats = torch.tensor([degs], device=device, dtype=torch.float32)

        # Scale the degrees
        local_stats = np.log(local_stats + 1)


        # Add one to subgraph as node ID 0 is reserved for empty node
        subgraph = torch.tensor(subgraph, device=device, dtype=torch.int32) + 1
        subgraph.flatten().unsqueeze(0)
        
        return State(subgraph, global_stats, local_stats)

    def run(self):
        """Run an episode."""
        T = self._sample_T()

        for t in range(T):
            # Create the current state
            state = self.create_state(self.args.subgraph_len, T, t)
            
            if t > 0:
                # Add the experience to the agent
                self.agent.add_ex(
                    Experience(
                        prev_state,
                        state,
                        edge_idx,
                        reward))

            # Get edge to prune
            edge_idx = self.agent(state)
            
            # Prune the edge
            self.prune_edge(edge_idx, state.subgraph)

            # Compute the reward for the sparsification decision
            reward = self._reward_manager.compute_reward()

            prev_state = state
        
        # Add the last experience
        self.agent.add_ex(
            Experience(
                prev_state,
                None,
                edge_idx,
                reward))
        
        # Reset the environment state
        self.reset()
        

        


        

