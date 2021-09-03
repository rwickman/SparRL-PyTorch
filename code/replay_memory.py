import numpy as np
import torch
import os
import random
import math

from agents.storage import Experience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    """Simple experience replay memory."""
    def __init__(self, args):
        self.args = args
        self._memory = []
        self._memory_file = os.path.join(self.args.save_dir, "memory.pt")


        # Pointer to end of memory
        self._cur_pos = 0

        # Load the memory
        if self.args.load and not self.args.eval:
            self.load()


    def add(self, exp: Experience):
        """Add an experience."""
        if len(self._memory) >= self.args.mem_cap:
            self._memory[self._cur_pos] = exp
        else:
            self._memory.append(exp)

        # Update end of memory
        self._cur_pos = (self._cur_pos + 1) %  self.args.mem_cap 
    
    def sample(self, batch_size):
        """Sample batch size experience replay."""
        return np.random.choice(self._memory, size=batch_size, replace=False)

    def cur_cap(self):
        """Get current capacir"""
        return len(self._memory)

    def save(self):
        print("SAVING")
        model_dict = {
            "memory" : self._memory,
            "pos": self._cur_pos
        }
        print("model_dict", model_dict)
        torch.save(model_dict, self._memory_file)
        
    def load(self):
        if os.path.exists(self._memory_file):
            model_dict = torch.load(self._memory_file)
            self._memory = model_dict["memory"]
            self._cur_pos = model_dict["pos"]
    
class PrioritizedExReplay:
    """Prioritized experience replay (PER) memory."""
    def __init__(self, args):
        self.args = args
        self._sum_tree = SumTree(self.args)
        self._memory_file = os.path.join(self.args.save_dir, "memory.pt")

        if self.args.load and not self.args.eval:
            self.load()
    
    def add(self, exp: Experience, error: float):
        """Append experience."""
        
        priority = self._compute_priority(error)
        if exp.is_expert:
            priority += self.args.expert_epsilon

        self._sum_tree.add(exp, priority)

    def sample(self, batch_size: int, train_step: int):
        """Sample batch size experience replay."""
        segment = self._sum_tree.total() / batch_size
        priorities = torch.zeros(batch_size).to(device)
        exps = []
        indices = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            mass = random.uniform(a, b)
            p, e_i, tree_idx = self._sum_tree.get(mass)
            priorities[i] = p
            exps.append(e_i)
            indices.append(tree_idx)

        # Compute importance sampling weights
        sample_ps = priorities / self._sum_tree.total()
        
        # Increase per beta by the number of episodes that have elapsed
        cur_per_beta = min((train_step / self.args.train_iter)/self.args.episodes, 1) * (1-self.args.per_beta) + self.args.per_beta
        #print("cur_per_beta", cur_per_beta)

        is_ws = (sample_ps  * self.cur_cap()) ** -cur_per_beta
        

        # Normalize to scale the updates downwards
        is_ws  = is_ws / is_ws.max()
        #print("is_ws", is_ws)

        return is_ws, exps, indices

    def cur_cap(self):
        return self._sum_tree.cur_cap()

    def update_priorities(self, indices, errors, is_experts: torch.tensor):
        #print("is_experts", is_experts)
        expert_priority_bonus = is_experts * self.args.expert_epsilon
        for idx, error, bonus in zip(indices, errors, expert_priority_bonus):
            priority = self._compute_priority(error) + bonus
            self._sum_tree.update(idx, priority)

    def save(self):
        model_dict = {
            "memory" : self._sum_tree.memory,
            "expert_memory": self._sum_tree.expert_memory,
            "tree" : self._sum_tree.tree,
            "pos" : self._sum_tree._end_pos,
            "expert_pos" : self._sum_tree._expert_pos
        }
        torch.save(model_dict, self._memory_file)
        
    def load(self):
        if os.path.exists(self._memory_file):
            model_dict = torch.load(self._memory_file)
            self._sum_tree.memory = model_dict["memory"]
            self._sum_tree.tree = model_dict["tree"]            
            self._sum_tree._end_pos = model_dict["pos"]
            if "expert_pos" in model_dict:
                print(model_dict["expert_pos"])
                self._sum_tree._expert_pos = model_dict["expert_pos"]
            if "expert_memory" in model_dict:
                self._sum_tree.expert_memory = model_dict["expert_memory"]
            print(len(self._sum_tree.expert_memory), len(self._sum_tree.memory))

    def _compute_priority(self, td_error):
        return (abs(td_error) + self.args.eps) ** self.args.per_alpha 



class SumTree:
    """Sum Tree used for PER."""
    def __init__(self, args):
        self.args = args
        # Raise to next power of 2 to make full binary tree
        self.capacity = 2 ** math.ceil(
            math.log(self.args.mem_cap + self.args.expert_mem_cap,2))

        # sum tree 
        self.tree = torch.zeros(2 * self.capacity - 1).to(device)
        self.memory = []

        # Memory for expert experiences
        self.expert_memory = []

        # Pointer to end of memory
        self._end_pos = 0

        # Pointer to end of expert memory
        self._expert_pos = 0
    
    def add(self, exp, priority):
        """Add experience to sum tree."""
        
        if exp.is_expert:
            cur_mem = self.expert_memory
            end_pos = self._expert_pos
            self._expert_pos = (self._expert_pos + 1) % self.args.expert_mem_cap
            cur_cap = self.args.expert_mem_cap
            idx = self.capacity + end_pos - 1
        else:
            cur_mem = self.memory
            end_pos = self._end_pos
            cur_cap = self.capacity - self.args.expert_mem_cap
            self._end_pos = (self._end_pos + 1) % cur_cap
            idx = self.capacity + (end_pos + len(self.expert_memory)) - 1
        
        # Add experience to memory
        if len(cur_mem) < cur_cap:
            cur_mem.append(exp)
        else:
            cur_mem[end_pos] = exp
    
        # Update sum tree
        self.update(idx, priority)



    def update(self, idx, priority):
        """Update priority of element and propagate through tree."""
        # Compute priority difference
        diff = priority - self.tree[idx]

        # Propagate update through tree
        while idx >= 0:
            self.tree[idx] += diff
            # Update to parent idx
            idx = (idx - 1) // 2

    def total(self):
        return self.tree[0]

    def get(self, val):
        """Sample from sum tree based on the sampled value."""
        tree_idx = self._retrieve(val)
        data_idx = tree_idx - self.capacity + 1

        #data = self.memory[data_idx]
        try:

            if data_idx < len(self.expert_memory):
                data = self.expert_memory[data_idx]
            else:
                data = self.memory[data_idx - len(self.expert_memory)]
        except Exception:
            print("len(self._expert_memory)", len(self.expert_memory))
            print("self.tree[tree_idx]", self.tree[tree_idx], "len(self.tree)", len(self.tree))
            print("data_idx", data_idx, "tree_idx", tree_idx, self.capacity, val, len(self.memory))
            print("self.tree", self.tree)
            import sys
            sys.exit()
        return self.tree[tree_idx], data, tree_idx

    def _retrieve(self, val):
        idx = 0
        # The left and right children
        left = 2 * idx + 1
        right = 2 * idx + 2

        # Keep going down the tree until leaf node with correct priority reached
        while left < len(self.tree):
            if val <= self.tree[left] or self.tree[left].isclose(val, atol=1e-2) or not self.tree[right].is_nonzero():
                idx = left
            else:
                idx = right
                val -= self.tree[left]

            left = 2 * idx + 1
            right= 2 * idx + 2

        return idx

    def cur_cap(self):
        return len(self.memory) + len(self.expert_memory)