import json, os, random
import numpy as np
from multiprocessing import Lock, Value
from community_detection import CommunityDetection
from scipy import stats
import math

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
        self._valid_pairs = set()
        



    def setup(self, part=None):
        if self.args.obj == "spsp":
            # Return reward for shortest path
            self._setup_spsp(part)
        elif self.args.obj == "com":
            self._setup_com()
        elif self.args.obj == "spearman":
            self._setup_spearman()
        else:
            raise Exception("Invalid Objective.")
            # By default use page rank
            # Get the prior page rank scores before training
            self._prior_pr = self._graph.get_page_ranks()
    
    def compute_reward(self, edge=None):
        if self.args.obj == "spsp":
            # Return reward for shortest path
            cur_reward = self._compute_spsp_reward()
        elif self.args.obj == "com":
            #cur_reward = self._compute_com_reward()
            cur_reward = self.edge_com_reward(edge)
        
        elif self.args.obj == "spearman":
            cur_reward = self._compute_spearman_reward()
        else:
            #print("USING PR REWARD")
            # By default use page rank
            cur_reward =self._compute_page_rank_reward()
        
        # Multiple by scale factor
        cur_reward = cur_reward

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
        if len(self._spsp_dists) > 0:
            avg_dist = np.mean(np.array(self._spsp_dists) - cur_spsp_dists)
            self._spsp_dists = cur_spsp_dists.tolist()
            return avg_dist

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

    def edge_com_reward(self, edge):
        if edge is None:
            return 0
    
        #reward = -self._com_detect.jaccard(edge)
        reward = self. _compute_com_reward()
        if self._com_detect.is_edge_same_com(edge):
            reward += -1
        #     # for node_id in edge:
                
        #     #     neighs = self._graph._G.neighbors(int(node_id))
        #     #     for dst_node_id in neighs:
        #     #         if self._com_detect.is_edge_same_com([node_id, dst_node_id]):
        #     #             reward =+ 1
        #     #             break
            
            
        else:
            reward += 1
        
        return reward
    
    def sample_spsp_dists(self, edge):
        num_samples = 128
        node_ids = self._graph.get_node_ids()
        
        # Get distance for at least this pair
        self._spsp_pairs.append(edge)
        
        self._spsp_dists.append(len(self._graph.get_shortest_path(edge[0], edge[1])) - 1)
        

        for src_id in edge:
            dst_nodes = random.sample(node_ids, k=num_samples*2)
            for dst_id in dst_nodes:
                path_len = len(self._graph.get_shortest_path(src_id, dst_id)) - 1
                # Check if path exists
                if path_len > 0:
                    self._spsp_dists.append(path_len)
                    self._spsp_pairs.append((src_id, dst_id))
                    
                
                if len(self._spsp_pairs) >= self.args.num_spsp_pairs:    
                    break
        
        for src_id in edge:
            i = 0
            for dst_id, value in self._graph.single_source_shortest_path(src_id).items():
                if i > num_samples:
                    break
                if dst_id == src_id:
                    continue
                self._spsp_pairs.append((src_id, dst_id))
                self._spsp_dists.append(value)
                i += 1
        #self._spsp_dists = np.array(self._spsp_dists)




    def _setup_spsp(self, part=None):
        self._spsp_pairs = []
        self._spsp_dists = []
        self._org_edges = set(self._graph._G.nodes())
        # Sample random spsp_pairs
        if part is None:
            node_ids = self._graph.get_node_ids()
        else:
            node_ids = part.get_node_ids()

        random.shuffle(node_ids)
        
        
       
        # Acquire a minimum number of shortest-path distances
        while True:
            src_nodes = random.choices(node_ids, k=math.ceil(self.args.num_spsp_pairs * 0.1))
            dst_nodes = random.choices(node_ids, k=math.ceil(self.args.num_spsp_pairs * 0.1))
            for src_id, dst_id in zip(src_nodes, dst_nodes):
                pair = (src_id, dst_id)
                if pair not in self._valid_pairs:    
                    path_len = len(self._graph.get_shortest_path(src_id, dst_id)) - 1
                    # Check if path exists
                    if path_len > 0:
                        self._valid_pairs.add((pair, path_len))
            
            if len(self._valid_pairs) >= self.args.num_spsp_pairs:
                break


        valid_pairs = random.sample(self._valid_pairs, k=min(self.args.num_spsp_pairs, len(self._valid_pairs)))
        for pair, dist in valid_pairs:
            self._spsp_pairs.append(pair)
            self._spsp_dists.append(dist)            
            if len(self._spsp_pairs) >= self.args.num_spsp_pairs:    
                break
        # self._spsp_dists = np.array(self._spsp_dists
        
    def _compute_spsp_dists(self, sub_one=False):
        """Compute the spsp distances between all the pairs."""
        spsp_dists = []
        for src_id, dst_id in self._spsp_pairs:
            if not sub_one:
                #print(src_id, dst_id, self._graph.get_shortest_path(src_id, dst_id))
                dist = len(self._graph.get_shortest_path(src_id, dst_id)) - 1
            else:
                dist = len(self._graph.get_shortest_path(src_id - 1, dst_id - 1)) - 1
            # Check if path exists
            if dist == -1:
                dist = self._graph.num_nodes

            spsp_dists.append(dist)
        return np.array(spsp_dists)

    def _setup_com(self):
        self._org_ari = self._com_detect.ARI_louvain()
        self._prev_ari = self._org_ari 
    
    def _setup_spearman(self):
        self._org_pr = list(self._graph.get_page_ranks().values())
        self._prev_spearmanr = 1.0


        
    def spsp_diff(self, sub_one=False):
        cur_spsp_dists = self._compute_spsp_dists()
        return  cur_spsp_dists - self._spsp_dists

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
                
