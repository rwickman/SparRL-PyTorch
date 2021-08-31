import numpy as np
import random
from scipy.stats import betabinom

from reward_manager import RewardManager
from conf import *
from agents.agent import Experience, State
from dataclasses import dataclass



class Environment:
    """Handles environment related tasks/management during an episode."""
    def __init__(self, args, agent, graph):
        self.args = args        
        self.agent = agent
        self._graph = graph 

        # Setup the RewardManager to manage rewards 
        self.reward_man = RewardManager(args, graph)
        self.reward_man.setup()
        
        self._removed_edges = set()

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

    def _sample_T(self) -> int:
        # Sample from possion distribution
        #T = np.random.poisson(self.T_lam)

        # Predict a point in the valid range, shifting by 1
        T = betabinom.ppf(random.random(), self.args.T_max-1, self.args.T_alpha, self.args.T_beta, loc=1)

        # Clip between valid bounds
        return int(np.clip(T, 1, self.args.T_max))

    def sample_subgraph(self, subgraph_len: int):
        """Sample a subgraph of edges."""
        sampled_edges = self._graph.sample_edges(subgraph_len)

        return sampled_edges

    def prune_edge(self, edge_idx: int, subgraph: torch.Tensor):
        """Prune an edge from the subgraph and the graph."""
        edge = [subgraph[0, 2*edge_idx], subgraph[0, 2*edge_idx + 1]]
        # Shift back to original node ids
        edge = (int(edge[0] - 1), int(edge[1] - 1))

        # Remove from graph
        self._removed_edges.add(edge)
        self._graph.del_edge(edge[0], edge[1])

    def create_state(self, subgraph_len: int, T: int, t: int):
        """Create the current state for the episode."""
        
        # Clip subgraph_len to valid range
        subgraph_len = min(subgraph_len, self._graph.get_num_edges())

        # Sanity-check
        if subgraph_len <= 0:
            raise Exception("Zero edges in graph.")

        # Sample random edges
        subgraph = self.sample_subgraph(subgraph_len)

        # Create global statistics
        prune_left = np.log(T-t + 1)
        num_edges_left = np.log(self._graph.get_num_edges() + 1)
        global_stats = torch.tensor([[[prune_left, num_edges_left]]], device=device, dtype=torch.float32)

        # Create local node statistics
        node_ids = []
        for e in subgraph:
            node_ids.append(e[0])
            node_ids.append(e[1])

        # Get node degrees
        degrees = self._graph.degree(node_ids)
        if isinstance(degrees, tuple):
            # Must be directed graph
            degs = torch.tensor(list(zip(degrees[0], degrees[1])), device=device)
        else:
            degs = torch.tensor(list(zip(degrees, [0] * len(degrees))), device=device)
        
        
        local_stats = degs.unsqueeze(0)#torch.tensor(degs], device=device, dtype=torch.float32)

        # Scale the degrees
        local_stats = torch.log(local_stats + 1)

        # Add one to subgraph as node ID 0 is reserved for empty node
        subgraph = torch.tensor(subgraph, device=device, dtype=torch.int32) + 1
        subgraph = subgraph.flatten().unsqueeze(0)
        
        return State(subgraph, global_stats, local_stats)

    def preprune(self, T: int):
        """Preprune edges from the graph before an episode.

        Args:
            T: number of edges that are pruned in the episode. 
        """
        # Calculate the maximum number of edges that can be prepruned
        num_edges = self._graph.get_num_edges()
        max_preprune = min(
            num_edges - T - 1,
            int(num_edges * self.args.preprune_pct))
        
        if max_preprune <= 0:
            return

        # Sample the number of edges to prune
        num_preprune = random.randint(0, max_preprune)

        # TODO: Consider preprune using expert instead
        subgraph = self.sample_subgraph(num_preprune)       

        # Prune the edges
        for edge in subgraph:
            self._removed_edges.add(edge)
            self._graph.del_edge(edge[0], edge[1])

    def run(self):
        for e_i in range(self.args.episodes):            
            print("e_i", e_i)
            
            # TODO: Preprune before each episode, will need to update sample_T as well
            # Run an episode
            final_reward = self.run_episode()
            print("final_reward", final_reward)

            # Reset the environment state
            self.reset()

            if not self.agent.is_expert and self.agent.is_ready_to_train:
                print("TRAINING MODEL")
                # Train the model
                for _ in range(self.args.train_iter):
                    self.agent.train()

                # Save the models
                self.agent.save(final_reward)
                
    def run_episode(self) -> float:
        """Run an episode."""
        # Sample number of edges to prune
        T = self._sample_T()
        
        # Preprune edges
        self.preprune(T)

        print("T", T)

        # Prune edges
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
                        reward,
                        self.agent.is_expert))

            # Get edge to prune
            edge_idx = self.agent(state)
            
            # Prune the edge
            self.prune_edge(edge_idx, state.subgraph)

            # Compute the reward for the sparsification decision
            reward = self.reward_man.compute_reward()
            
            # ep_return += reward * self.args.gamma ** (T- t - 1)

            prev_state = state
        

        # Get a next state if there are still enough edges in the graph
        if self._graph.get_num_edges() >= 1:
            state = self.create_state(self.args.subgraph_len, T, t)
            print("self._graph.get_num_edges()", self._graph.get_num_edges())

        # Add the last experience
        self.agent.add_ex(
            Experience(
                prev_state,
                state,
                edge_idx,
                reward,
                self.agent.is_expert))
        
        return reward

        


        

