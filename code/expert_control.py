from scipy import stats

from agents.random_agent import RandomAgent
from agents.rl_agent import RLAgent
from agents.agent import Agent
from conf import *
from graph import Graph
from environment import Environment

class ExpertControl:
    def __init__(self, args):
        self.args = args

        # Create environment for testing
        self.graph = Graph(self.args)
        self.env = Environment(self.args, RandomAgent(self.args), self.graph)


        self._rand_rewards = self._run_episodes(RandomAgent(self.args))


    def _run_episodes(self, agent: Agent) -> list:
        rewards = []
        for i in range(self.args.ec_episodes):
            # Calculate preprune percent
            preprune_pct = self.args.preprune_pct * (i+1) /self.args.ec_episodes
            
            # Preprune edges
            self.env.preprune(preprune_pct)

            # Run the episode            
            for t in range(self.args.T_ec):

                # Create state and move to CUDA
                state = self.env.create_state(self.args.subgraph_len, self.args.T_ec, t)
                state.subgraph = state.subgraph.to(device)
                state.local_stats = state.local_stats.to(device)
                state.global_stats = state.global_stats.to(device)
                state.mask = state.mask.to(device)

                edge_idx = agent(state)
                self.env.prune_edge(edge_idx, state.subgraph)
                rewards.append(self.env.reward_man.compute_reward())

            # Reset the environment
            self.env.reset()            
        
        return rewards

    def _test_mean_reward(self, agent: RLAgent) -> bool:
        """Run a two-sample t-test to see if mean reward is better than random.
        
        Returns:
            boolean indicating if there is significance difference beteween the means.
        """
        rewards = self._run_episodes(agent)
        # print("self._rand_rewards", self._rand_rewards, len(self._rand_rewards))
        # print("RL Agent rewards", rewards, len(rewards))
        print("RAND MEAN", sum( self._rand_rewards)/len(self._rand_rewards))
        print("RL MEAN", sum(rewards) / len(rewards))
        test_stat, p_value = stats.ttest_ind(
            self._rand_rewards,
            rewards)

        # Check if Expert Control should be triggered
        should_trigger = p_value < self.args.ec_sig_val and test_stat < 0
        print("p_value", p_value, "test_stat", test_stat)
        return should_trigger

    