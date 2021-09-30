import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

from agents.random_agent import RandomAgent
from agents.rl_agent import RLAgent
from conf import *

class ResultsManager:
    def __init__(self, args, agent, env):
        self.args = args
        self.agent = agent
        self.agent._sparrl_net.eval()
        self.rand_agent = RandomAgent(args)
        self.env = env
        self._spsp_freq = {}

    
    def eval(self):     
        final_rewards = self.eval_agent(self.agent)
        spsp_str = ""
        if self.args.obj == "spsp":
            num_spsp_dists = sum([v for v in self._spsp_freq.values()])
            for k in self._spsp_freq:
                self._spsp_freq[k] /= num_spsp_dists
                assert self._spsp_freq[k] <= 1

            total_prob = 0
            for dist, prob in sorted(self._spsp_freq.items()):
                spsp_str += "{}: {}\n".format(dist, prob)
                total_prob += prob
        print("spsp_str:", spsp_str)
        print("spsp_freq", self._spsp_freq)

        print("final_rewards", final_rewards)
        print("sparrl avg", sum(final_rewards) / len(final_rewards))
        final_rewards_rand = self.eval_agent(self.rand_agent)
        
        print("avg rand", sum(final_rewards_rand) / len(final_rewards_rand))
        #print("final_rewards_rand", final_rewards_rand)
        #self.plot_results(final_rewards)
        
    def eval_agent(self, agent):
        final_rewards = []
        self.env.agent = agent
        # self.env.preprune(self.args.T_max)
        for i in range(self.args.episodes):
            if isinstance(agent, RLAgent):
                final_rewards.append(self.run_rl_eval(agent, self.args.T_max))
            else:
                final_rewards.append(self.run_episode(agent, self.args.T_max))

            self.env.reset()

        return final_rewards


    # def run_rl_eval(self, agent):
    #     for t in range(self.args.T_max//32):
    #         state = self.env.create_state(self.args.subgraph_len, self.args.T_max, t*self.args.eval_batch_size)
    #         #edge_idx = agent(state, argmax=True)
    #         q_vals, valid_actions = agent.predict(state, argmax=True)
    #         max_vals = q_vals.argsort(descending=True)
    #         for edge_idx in range(32):
    #             self.env.prune_edge(max_vals[edge_idx], state.subgraph)

    #     return self.env.reward_man.compute_sparmanr()

    def run_rl_eval(self, agent, T: int,):
        if self.args.eval_batch_size == 1:
            return self.run_episode(agent, T)
        org_subgraph_len = self.args.subgraph_len 
        self.args.subgraph_len = self.args.eval_batch_size * self.args.subgraph_len
        num_left = T % self.args.eval_batch_size
        num_batches = T//self.args.eval_batch_size
        org_num_edges = self.env._graph.get_num_edges()

        if num_left >= 1:
            num_batches += 1
        for t in tqdm(range(num_batches)):
            state = self.env.create_state(self.args.subgraph_len, T, t*self.args.eval_batch_size)
            state.subgraph = state.subgraph.reshape(-1, org_subgraph_len*2)
            
            # Create new global_stats
            t_arr = torch.arange(t*self.args.eval_batch_size, (t+1)*self.args.eval_batch_size, device=device)
            T_arr = torch.tensor(T, device=device).repeat(self.args.eval_batch_size)
            num_edges_left = torch.tensor(self.env._graph.get_num_edges(), device=device).repeat(self.args.eval_batch_size)
            
            num_edges_left = num_edges_left - torch.arange(0, self.args.eval_batch_size, device=device).unsqueeze(1)
            num_edges_left = num_edges_left / org_num_edges
            #print("num_edges_left - torch.arange(0, self.args.eval_batch_size)", num_edges_left - torch.arange(0, self.args.eval_batch_size))
            
            #state.global_stats = torch.cat((num_edges_left, torch.zeros(self.args.eval_batch_size, 1)), -1).unsqueeze(1)
            state.global_stats = num_edges_left.unsqueeze(1)
            state.global_stats = state.global_stats.to(device)
            state.local_stats = state.local_stats.reshape(self.args.eval_batch_size, -1, NUM_LOCAL_STATS)

            # print("state.mask.shape", state.mask.shape)
            #print("org_subgraph_len", org_subgraph_len)
            state.mask = state.mask.reshape(-1, org_subgraph_len*2, self.args.max_neighbors, 1)
            state.neighs = state.neighs.reshape(-1, org_subgraph_len*2, self.args.max_neighbors)
            # print("")
            # print("state.mask", state.mask)
            # print("state.mask.shape", state.mask.shape)
            # print("state.global_stats.shape", state.global_stats.shape)
            # print("state.local_stats", state.local_stats.shape)
            #state.mask = torch.cat(
            #    (state.mask, torch.zeros(self.args.eval_batch_size,1,1,1, device=device)), -1)
            edge_idxs = agent(state, argmax=True)
            # Prune batch of edges
            for i, edge_idx in enumerate(edge_idxs):
                if num_left >= 1 and t + 1 == num_batches and i > num_left:
                    break
                self.env.prune_edge(edge_idx, state.subgraph[i:i+1])

        self.args.subgraph_len = org_subgraph_len
        print("E:", self.env._graph.get_num_edges())
        return self.get_final_reward()




    def run_episode(self, agent, T: int):
        #self.env.preprune(self.args.T_eval)
        for t in tqdm(range(T)):
            state = self.env.create_state(self.args.subgraph_len, T, t)
            
            if isinstance(agent, RLAgent):
                edge_idx = agent(state, argmax=True)
            else:    
                edge_idx = agent(state)

            self.env.prune_edge(edge_idx, state.subgraph)
        
        return self.get_final_reward()


    def get_final_reward(self):
        if self.args.obj == "spearman":
            return self.env.reward_man.compute_sparmanr()
        elif self.args.obj == "com":
            return self.env.reward_man._com_detect.ARI_louvain()
        elif self.args.obj == "spsp":
            if self.args.eval:
                spsp_diff = self.env.reward_man.spsp_diff()
                spsp_counter = Counter(spsp_diff)
                for k in spsp_counter:
                    assert k >= 0
                    if k not in self._spsp_freq:
                        self._spsp_freq[k] = 0
                    self._spsp_freq[k] += spsp_counter[k]
            
            return self.env.reward_man._compute_spsp_reward()
            

    def plot_results(self, rewards):
        # def moving_average(x):
        #     return np.convolve(x, np.ones(self.args.reward_smooth_w), 'valid') / self.args.reward_smooth_w
        plt.plot(rewards)
        plt.show()
