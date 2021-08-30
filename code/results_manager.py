import torch
import matplotlib.pyplot as plt

from agents.random_agent import RandomAgent

class ResultsManager:
    def __init__(self, args, agent, env):
        self.args = args
        self.agent = agent
        self.agent._sparrl_net.eval()
        self.rand_agent = RandomAgent(args)
        self.env = env

    
    def eval(self):
        final_rewards = self.eval_agent(self.agent)
        print("final_rewards", final_rewards)
        final_rewards_rand = self.eval_agent(self.rand_agent)
        print("final_rewards_rand", final_rewards_rand)
        print("sparrl avg", sum(final_rewards) / len(final_rewards))
        print("avg rand", sum(final_rewards_rand) / len(final_rewards_rand))
        #self.plot_results(final_rewards)
        
    def eval_agent(self, agent):
        final_rewards = []
        self.env.agent = agent

        for i in range(self.args.eval_episodes):
            final_rewards.append(self.run_episode(agent))
            self.env.reset()
        
        return final_rewards

    def run_episode(self, agent):
        self.env.preprune(self.args.T_eval)
        for t in range(self.args.T_eval):
            state = self.env.create_state(self.args.subgraph_len, self.args.T_eval, t)
            edge_idx = agent(state)
            self.env.prune_edge(edge_idx, state.subgraph)
        
        return self.env.reward_man.compute_reward()


    def plot_results(self, rewards):
        # def moving_average(x):
        #     return np.convolve(x, np.ones(self.args.reward_smooth_w), 'valid') / self.args.reward_smooth_w
        plt.plot(rewards)
        plt.show()
