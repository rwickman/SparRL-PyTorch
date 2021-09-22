import json, os, random
import numpy as np
from multiprocessing import Lock, Value
from community_detection import CommunityDetection
from scipy import stats

class RewardManager:
    """Manages the reward acquired by the agent."""
    loaded_reward_data = Value("i", False)
    
    def __init__(self, args, graph):
        self.args = args
        self._graph = graph
        self._reward_lock = Lock()
        # self._load_reward_data()
        self._reward_json = os.path.join(self.args.save_dir, "rewards.json")
        if self.args.obj == "com":
            self._com_detect = CommunityDetection(self.args, self._graph)



    def setup(self, part=None):
        if self.args.obj == "spsp":
            # Return reward for shortest path
            self._setup_spsp(part)
        elif self.args.obj == "com":
            self._setup_com()
        elif self.args.obj == "spearman":
            self._setup_spearman()
        else:
            # By default use page rank
            # Get the prior page rank scores before training
            self._prior_pr = self._graph.get_page_ranks()
    
    def compute_reward(self):
        if self.args.obj == "spsp":
            # Return reward for shortest path
            cur_reward = self._compute_spsp_reward()
        elif self.args.obj == "com":
            cur_reward = self._compute_com_reward()
        elif self.args.obj == "spearman":
            cur_reward = self._compute_spearman_reward()
        else:
            #print("USING PR REWARD")
            # By default use page rank
            cur_reward =self._compute_page_rank_reward()
        
        # Multiple by scale factor
        cur_reward = cur_reward * self.args.reward_factor

        return cur_reward

    def _compute_page_rank_reward(self):
        """Compute the page rank reward.""" 
        cur_pr = self._graph.get_page_ranks()
        # se = 0
        # for node in cur_pr:
        #     se += (cur_pr[node] - self._prior_pr[node]) ** 2

        # mse = se/self._graph.num_nodes

        # # NOTE: negative MSE as we want to maximize reward (i.e., minimize the MSE to 0)
        # return -mse

        pr_diff = 0
        for node in cur_pr:
            pr_diff += abs(cur_pr[node] - self._prior_pr[node])
        
        l1_loss = pr_diff / len(cur_pr)
        
        return -l1_loss 

    def _compute_spsp_reward(self):
        cur_spsp_dists = self._compute_spsp_dists()
        mse = np.mean((self._spsp_dists - cur_spsp_dists) ** 2)
        return -mse

    def compute_sparmanr(self):
        cur_pr = list(self._graph.get_page_ranks().values()) 
        cur_spearmanr = stats.spearmanr(self._org_pr, cur_pr).correlation
        return cur_spearmanr

    def _compute_spearman_reward(self):
        cur_spearmanr = self.compute_sparmanr()
        reward = cur_spearmanr - self._prev_spearmanr
        self._prev_spearmanr = cur_spearmanr
        return reward#stats.spearmanr(self._org_pr, cur_pr).correlation #* self.args.reward_factor

    def _compute_com_reward(self):
        cur_ari = self._com_detect.ARI_louvain()
        reward = cur_ari - self._prev_ari
        self._prev_ari = cur_ari
        return reward
    
    def _setup_spsp(self, part=None):
        self._org_edges = set(self._graph._G.nodes())
        # Sample random spsp_pairs
        if part is None:
            node_ids = self._graph.get_node_ids()
        else:
            node_ids = part.get_node_ids()

        self._spsp_pairs = []
        self._spsp_dists = []
        random.shuffle(node_ids)
        for src_id in node_ids:
            for dst_id in node_ids:
                if src_id != dst_id:
                    path_len = len(self._graph.get_shortest_path(src_id, dst_id)) - 1
                    # Check if path exists
                    if path_len > 0:
                        self._spsp_dists.append(path_len)
                        self._spsp_pairs.append((src_id, dst_id))
                
                
                if len(self._spsp_pairs) >= self.args.num_spsp_pairs:    
                    break
            if len(self._spsp_pairs) >= self.args.num_spsp_pairs:    
                    break

        self._spsp_dists = np.array(self._spsp_dists)
        
    def _compute_spsp_dists(self, sub_one=False):
        """Compute the spsp distances between all the pairs."""
        spsp_dists = []
        for src_id, dst_id in self._spsp_pairs:
            if not sub_one:
                dist = len(self._graph.get_shortest_path(src_id, dst_id)) - 1
            else:
                dist = len(self._graph.get_shortest_path(src_id - 1, dst_id - 1)) - 1
            # Check if path exists
            if (dist == 0):
                dist = self._graph.get_num_nodes()
            spsp_dists.append(dist)
        return np.array(spsp_dists)

    def _setup_com(self):
        self._org_ari = self._com_detect.ARI_louvain()
        self._prev_ari = self._org_ari 
    
    def _setup_spearman(self):
        self._org_pr = list(self._graph.get_page_ranks().values())
        self._prev_spearmanr = 1.0


        
    def spsp_diff(self, sub_one=False):
        cur_edges = set(self._graph._G.nodes())
        # print(cur_edges)
        # print(self._org_edges)
        
        if cur_edges.difference(self._org_edges):
            cur_spsp_dists = self._compute_spsp_dists(sub_one=True)
        else:
            cur_spsp_dists = self._compute_spsp_dists(sub_one=False)
        return self._spsp_dists - cur_spsp_dists

    def reset(self, part=None):
        """Reset rewards states."""
        self.setup(part)
        
        
    # def _load_reward_data(self):
    #     with self._reward_lock:
    #         if not RewardManager.loaded_reward_data.value:
    #             if self.args.load and os.path.exists(self._reward_json):
    #                 with open(self._reward_json) as f:
    #                     reward_data = json.load(f)
    #                     self.args.mean_mse = Value("d", reward_data["mean_mse"])
    #                     self.args.var_mse = Value("d", reward_data["var_mse"])
    #                     self.args.m2 = Value("d", reward_data["m2"])
    #                     self.args.n = Value("i", reward_data["n"])
    #             else:
    #                 self.args.mean_mse = Value("d", 0.0)
    #                 self.args.var_mse = Value("d", 0.0)
    #                 self.args.m2 = Value("d", 0.0)
    #                 self.args.n = Value("i", 0)
                
    #             RewardManager.loaded_reward_data.value = True


    # def save(self):
    #     """Save the reward data."""
    #     with self._reward_lock:
    #         reward_data = {
    #             "mean_mse" : self.args.mean_mse.value,
    #             "var_mse" : self.args.var_mse.value,
    #             "m2" : self.args.m2.value,
    #             "n" : self.args.n.value
    #         }
    #         with open(self._reward_json, "w") as f:
    #             json.dump(reward_data, f)
                
