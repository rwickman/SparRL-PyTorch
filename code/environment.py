import numpy as np
import random
from scipy.stats import betabinom

from reward_manager import RewardManager
from conf import *
from agents.storage import Experience, State

# Prevent high memory usage

class Environment:
    """Handles environment related tasks/management during an episode."""
    def __init__(self, args, agent, graph, device="cpu"):
        self.args = args        
        self.agent = agent
        self._graph = graph
        self._org_num_edges = self._graph.get_num_edges()
        #self.args.T_max = min(self._org_num_edges, self.args.T_max) 

        # Setup the RewardManager to manage rewards 
        self.reward_man = RewardManager(args, graph)
        self.reward_man.setup()
        
        self._removed_edges = set()

        self._device = device
        print("self._device", self._device)

    @property
    def num_nodes(self):
        return self._graph.num_nodes

    def reset(self):
        """Reset environment for next episode."""
        # Restore pruned edges to graph
        for edge in self._removed_edges:
            self._graph.add_edge(edge[0], edge[1])

        self._removed_edges = set()

        self.agent.reset()
        self.reward_man.reset()

    def _sample_T(self) -> int:
        """Sample the number of edges to prune from a Beta-binomial distribution.
        
        Returns:
            integer number of edges to prune.
        """

        # Predict a point in the valid range, shifting by 1
        #T = betabinom.ppf(random.random(), self.args.T_max-1, self.args.T_alpha, self.args.T_beta, loc=1)

        # Clip between valid bounds
        #return int(np.clip(T, 1, self.args.T_max))
        T = random.randint(1, self.args.T_max.value)
        # T = 1
        return T

    def sample_subgraph(self, subgraph_len: int) -> list:
        """Sample a subgraph of edges.
        
        Args:
            subgraph_len: length of the sampled subgraph

        Returns:
            the list of edges in a subgraph
        """
        sampled_edges = self._graph.sample_edges(subgraph_len)

        return sampled_edges

    def prune_edge(self, edge_idx: int, subgraph: torch.Tensor):
        """Prune an edge from the subgraph and the graph."""
        edge = [subgraph[0, 2*edge_idx], subgraph[0, 2*edge_idx + 1]]
        
        # Shift back to original node ids
        #edge = (int(edge[0] - 1), int(edge[1] - 1))
        edge = (int(edge[0]), int(edge[1]))

        if not (edge[0] >= 0 and edge[1] >= 0):
            raise Exception(f"INVALID EDGE {edge} WITH IDX {edge_idx}")
            #assert edge[0] > 0 and edge[1] > 0 

        # Remove from graph
        self._removed_edges.add(edge)
        self._graph.del_edge(edge[0], edge[1])

    def create_state(self, subgraph_len: int, T: int, t: int, num_preprune: int = 0):
        """Create the current state for the episode."""
        # Clip subgraph_len to valid range
        subgraph_len = min(subgraph_len, self._graph.get_num_edges())

        # Sanity-check
        if subgraph_len <= 0:
            print("num_preprune", num_preprune)
            print("self._graph.get_num_edges()", self._graph.get_num_edges())
            raise Exception("Zero edges in graph.")

        # Sample random edges
        subgraph = self.sample_subgraph(subgraph_len)

        # Create global statistics
        prune_left = np.log(T-t)
        #num_edges_left = np.log(self._graph.get_num_edges())
        num_edges_left = self._graph.get_num_edges()/self._org_num_edges
        preprune_pct = num_preprune / self._org_num_edges
        #global_stats = torch.tensor([[[prune_left, num_edges_left, preprune_pct]]], device=self._device, dtype=torch.float32)
        global_stats = torch.tensor([[[num_edges_left]]], device=self._device, dtype=torch.float32)
        
        # Create local node statistics
        node_ids = []
        for e in subgraph:
            node_ids.append(e[0])
            node_ids.append(e[1])

        # Get node degrees
        degrees = self._graph.degree(node_ids)
        if isinstance(degrees, tuple):
            # Must be directed graph
            degs = torch.tensor(list(zip(degrees[0], degrees[1])), device=self._device)
        else:
            # Must be undirected graph
            degs = torch.tensor(list(zip(degrees, [0] * len(degrees))), device=self._device)
        
        local_stats = degs.unsqueeze(0)

        # Scale the degrees
        local_stats = torch.log(local_stats + 1)

        # Add one to subgraph as node ID 0 is reserved for empty node
        #temp_subgraph = torch.tensor(subgraph, device=self._device, dtype=torch.int32) + 1
        temp_subgraph = torch.tensor(subgraph, device=self._device, dtype=torch.int32)
        temp_subgraph = temp_subgraph.flatten().unsqueeze(0)

        # Extend to include empty edges
        subgraph = torch.zeros(
            1, self.args.subgraph_len * 2, device=self._device, dtype=torch.int32)
        subgraph[0, :temp_subgraph.shape[1]] = temp_subgraph

        temp_local_stats = torch.zeros(
            1, self.args.subgraph_len * 2, NUM_LOCAL_STATS, device=self._device)
        temp_local_stats[0, :local_stats.shape[1]] = local_stats
        local_stats = temp_local_stats

        # Get the neighbors
        neighs = torch.zeros(1, self.args.subgraph_len*2, self.args.max_neighbors, device=self._device, dtype=torch.int32)
        mask = torch.ones(1, self.args.subgraph_len*2, self.args.max_neighbors, 1, device=self._device)
        for i, node_id in enumerate(subgraph[0]):
            if node_id != 0:
                neighs_list = list(self._graph._G.neighbors(int(node_id)))
                if len(neighs_list) > self.args.max_neighbors:
                    random.shuffle(neighs_list)
                neighs[0, i, :len(neighs_list)] = torch.tensor(neighs_list[:self.args.max_neighbors], device=self._device)
                mask[0, i, :len(neighs_list)] = 0


                    
        # # Don't mask out the global stats
        # mask[0, 0, 0, -1] = 0
        
        return State(subgraph, global_stats, local_stats, mask, neighs)

    def preprune(self, T: int, fixed_pct: float = None) -> int:
        """Preprune edges from the graph before an episode.

        Args:
            T: number of edges that are pruned in the episode.
            fixed_pct: the optional fixed percent to preprune. 
        """
        if not fixed_pct:
            # Calculate the maximum number of edges that can be prepruned
            num_edges = self._graph.get_num_edges()
            max_preprune = min(
                num_edges - T - 1,
                int(num_edges * self.args.preprune_pct))
            
            if max_preprune <= 0:
                return 0

            # Sample the number of edges to prune
            num_preprune = random.randint(0, max_preprune)

            # TODO: Consider preprune using expert instead
            subgraph = self.sample_subgraph(num_preprune)
        else:
            num_preprune = fixed_pct * num_edges
            subgraph = self.sample_subgraph(num_preprune)

        # Prune the edges
        for edge in subgraph:
            self._removed_edges.add(edge)
            self._graph.del_edge(edge[0], edge[1])

        return num_preprune

    def run(self):
        for e_i in range(self.args.episodes // self.args.workers):
            print("e_i", e_i)
         
            # Run an episode
            final_reward = self.run_episode()

            # Reset the environment state
            self.reset()
                
    def run_episode(self) -> float:
        """Run an episode."""
        # Sample number of edges to prune
        T = self._sample_T()
        
        # Preprune edges
        num_preprune = self.preprune(T)

        # Throw away prepruned reward
        self.reward_man.compute_reward()
        #print("INIT SPEARMAN: ", self.reward_man._prev_spearmanr)
        
        print("T", T)
        # Sample subgraph length
        subgraph_len = self.args.subgraph_len#random.randint(2, self.args.subgraph_len)

        # Prune edges
        for t in range(T):
            # Create the current state
            state = self.create_state(
                subgraph_len, T, t, num_preprune)
            
            if t > 0:
                # Add the experience to the agent
                self.agent.add_ex(
                    Experience(
                        prev_state,
                        state,
                        edge_idx,
                        reward))

            # Get edge to prune
            with torch.no_grad():
                edge_idx = self.agent(state)
            
            # Prune the edge
            self.prune_edge(edge_idx, state.subgraph)

            # Compute the reward for the sparsification decision
            reward = self.reward_man.compute_reward()

            prev_state = state

        
        # Acquire a next_state, if available
        if self._graph.get_num_edges() > 1:
            next_state = self.create_state(
                subgraph_len, T, t, num_preprune)
        else:
            next_state = None

        # Add the last experience
        self.agent.add_ex(
            Experience(
                prev_state,
                next_state,
                edge_idx,
                reward))

        return reward