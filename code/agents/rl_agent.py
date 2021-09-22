import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json

from model import SparRLNet
from conf import *
from agents.agent import Agent
from agents.storage import State
from agents.expert_agent import ExpertAgent


class RewardScaler:
    def __init__(self, args):
        self.args = args
        self._reward_file = os.path.join(self.args.save_dir, "reward_stats.pt")
        self._max_reward = torch.tensor(1, device=device)

        # Load saved rewards
        if self.args.load and not self.args.eval:
            print("Loading rewards")
            self.load()
        else:
            self._rewards = torch.zeros(self.args.reward_scaler_window, device=device)
            self._end = 0
    
    def add_reward(self, reward):
        """Add a reward to compute statistics over."""
        if self._end < self.args.reward_scaler_window:
            self._rewards[self._end] = reward
            self._end += 1
        
    def scale_reward(self, reward):
        """Scale a given reward."""
        if self._end >= 2: 
            reward = (reward - self._rewards[:self._end].mean()) / self._rewards[:self._end].std()
            # reward = reward * 0.1
            self._max_reward = max(self._max_reward, reward.abs())

            #reward = torch.clip(reward, -1, 1)
            #reward = reward / self._max_reward
        
        return reward

    def save(self):
        reward_dict = {
            "rewards" : self._rewards,
            "end" : self._end,
            "max_reward" : self._max_reward
        }
        print("self._max_reward", self._max_reward)
        torch.save(reward_dict, self._reward_file)

    def load(self):
        reward_dict = torch.load(self._reward_file, map_location=device)
        self._rewards = reward_dict["rewards"]
        self._end = reward_dict["end"]
        self._max_reward = reward_dict["max_reward"]


class RLAgent(Agent):
    def __init__(self, args, memory, num_nodes: int, expert_agent: ExpertAgent = None):
        super().__init__(args)
        self._memory = memory
        self._expert_agent = expert_agent
        self._reward_scaler = RewardScaler(self.args)

        # Number of elapsed parameter update steps
        self._train_dict = {
            "update_step" : 0,
            "episodes" : 0,
            "avg_rewards" : [],
            "mse_losses" : [],
            # "sigma_mean_abs_1": [],
            # "sigma_mean_abs_2": []
        }
        
        # Create SparRL networks
        self._sparrl_net = SparRLNet(self.args, num_nodes).to(device)
        
        self._sparrl_net_tgt = SparRLNet(self.args, num_nodes).to(device)
        self._sparrl_net_tgt.eval()

        # Create optimizer and LR scheduler to decay LR
        self._optim = optim.Adam(self._sparrl_net.parameters(), self.args.lr)
        
        self._gam_arr = self._create_gamma_arr()
        self._model_file = os.path.join(self.args.save_dir, "sparrl_net.pt")
        self._train_dict_file = os.path.join(self.args.save_dir, "train_dict.json")

        if self.args.load:
            self.load()

    def reset_noise(self):
        self._sparrl_net.reset_noise()
        self._sparrl_net_tgt.reset_noise()
        


    @property
    def epsilon_threshold(self):
        """Return the current epsilon value used for epsilon-greedy exploration."""
        # Adjust for number of expert episodes that have elapsed
        #return self.args.min_epsilon#self.args.epsilon
        eps = min(((self.args.decay_episodes - self._train_dict["episodes"]) / (self.args.decay_episodes)) * self.args.epsilon, self.args.epsilon) 
        return max(eps, self.args.min_epsilon)

        # eps = min(((self.args.episodes - self._train_dict["episodes"]) / (self.args.episodes)) * self.args.epsilon, self.args.epsilon) 
        # return max(eps, self.args.min_epsilon)
        # cur_step = max(self._train_dict["update_step"] - self.args.expert_episodes, 0)
        # return self.args.min_epsilon + (self.args.epsilon - self.args.min_epsilon) * \
        #     math.exp(-1. * cur_step / self.args.epsilon_decay)
    
    @property
    def is_ready_to_train(self) -> bool:
        """Check for if the model is ready to start training."""
        return self._memory.cur_cap() >= self.args.batch_size and self._train_dict["episodes"] >= self.args.min_ep

    def reset(self, avg_reward: float = None):
        if not self.args.eval:
            # Update number of elapsed episodes
            self._train_dict["episodes"] += 1

            # Add the last reward of the episode
            if avg_reward:
                self._train_dict["avg_rewards"].append(avg_reward)

            # Add episode experiences
            self._add_stored_exs()

    def save(self):
        """Save the models."""
        model_dict = {
            "sparrl_net" : self._sparrl_net.state_dict(),
            "sparrl_net_tgt" : self._sparrl_net_tgt.state_dict(),
            "optimizer" : self._optim.state_dict()
        }

        torch.save(model_dict, self._model_file)

        with open(self._train_dict_file, "w") as f:
            json.dump(self._train_dict, f)

        # Save the experience replay
        self._memory.save()
        self._reward_scaler.save()

    def load(self):
        """Load the models."""
        model_dict = torch.load(self._model_file, map_location=device)
        
        self._sparrl_net.load_state_dict(model_dict["sparrl_net"])
        self._sparrl_net_tgt.load_state_dict(model_dict["sparrl_net_tgt"])
        self._optim.load_state_dict(model_dict["optimizer"])

        with open(self._train_dict_file) as f:
            self._train_dict = json.load(f)

    def add_ex(self, ex):
        """Add a time step of experience."""
        if not self.args.eval and self._memory:
            ex.is_expert = self._should_add_expert_ex
            self._ex_buffer.append(ex)
 
    def _update_target(self):
        """Perform soft update of the target policy."""
        for tgt_sparrl_param, sparrl_param in zip(self._sparrl_net_tgt.parameters(), self._sparrl_net.parameters()):
            tgt_sparrl_param.data.copy_(
                self.args.tgt_tau * sparrl_param.data + (1.0-self.args.tgt_tau) * tgt_sparrl_param.data)
        # self._sparrl_net_tgt.load_state_dict(self._sparrl_net.state_dict())


    def _create_gamma_arr(self):
        """Create a gamma tensor for multi-step DQN."""
        gam_arr = torch.ones(self.args.dqn_steps)
        for i in range(1, self.args.dqn_steps):
            gam_arr[i] = self.args.gamma * gam_arr[i-1] 
        return gam_arr

    def _sample_action(self, q_vals: torch.Tensor, argmax=False) -> int:
        """Sample an action from the given Q-values."""
        if not argmax and self.epsilon_threshold >= np.random.rand():
            # Sample a random action
            action = np.random.randint(q_vals.shape[0])
        else:
            with torch.no_grad():
            # Get action with maximum Q-value
                action = q_vals.argmax()

        return int(action)

    def _add_stored_exs(self):
        """Add experiences stored in temporary buffer into replay memory.
        
        This method makes the assumption that self._ex_buffer only contains experiences
        from the same episode.
        """
        self._sparrl_net.eval()
        rewards = torch.zeros(self.args.dqn_steps)
        for i in reversed(range(len(self._ex_buffer))):
            rewards[0] = self._ex_buffer[i].reward
            self._reward_scaler.add_reward(self._ex_buffer[i].reward)
            cur_gamma = self.args.gamma

            # Update the experience reward to be the n-step return
            if i + self.args.dqn_steps < len(self._ex_buffer):
                self._ex_buffer[i].reward = rewards.dot(self._gam_arr)
                self._ex_buffer[i].next_state = self._ex_buffer[i + self.args.dqn_steps].state
                cur_gamma = cur_gamma ** self.args.dqn_steps

            # Update gamma based on n-step return
            self._ex_buffer[i].gamma = cur_gamma

            with torch.no_grad():
                # Get the Q-value for the state, action pair
                q_val = self._sparrl_net(self._ex_buffer[i].state)[self._ex_buffer[i].action]
                
                if self._ex_buffer[i].next_state is not None:
                    # Get the valid action for next state that maximizes the q-value
                    valid_actions = self._get_valid_edges(self._ex_buffer[i].next_state.subgraph[0])
                    q_next = self._sparrl_net(self._ex_buffer[i].next_state)
                    #print("q_next", q_next)
                    next_action = self._sample_action(q_next[valid_actions], argmax=True)
                    next_action = valid_actions[next_action] 

                    # Compute TD target based on target function q-value for next state
                    q_next_target = self._sparrl_net_tgt(self._ex_buffer[i].next_state)[next_action]
                    td_target = self._ex_buffer[i].reward + self._ex_buffer[i].gamma *  q_next_target
                    # print("q_next_target",  self._sparrl_net_tgt(self._ex_buffer[i].next_state))
                    # print("td_target", td_target, "\n")
                else:
                    td_target = self._ex_buffer[i].reward

            td_error = td_target - q_val
            self._memory.add(self._ex_buffer[i], td_error)      

            # Shift the rewards down
            rewards = rewards.roll(1)

        # Clear the experiences from the experince buffer
        self._ex_buffer.clear()
        self._sparrl_net.train()


    def _unwrap_exs(self, exs: list):
        """Extract the states, actions and rewards from the experiences."""
        subgraphs = torch.zeros(self.args.batch_size, self.args.subgraph_len*2, device=device, dtype=torch.int32)
        global_stats = torch.zeros(self.args.batch_size, 1, NUM_GLOBAL_STATS, device=device)
        local_stats = torch.zeros(self.args.batch_size, self.args.subgraph_len*2, NUM_LOCAL_STATS, device=device)
        masks = torch.zeros(self.args.batch_size, 1, 1, self.args.subgraph_len, device=device)

        actions = []
        rewards = torch.zeros(self.args.batch_size, device=device)
        next_subgraphs = torch.zeros(self.args.batch_size, self.args.subgraph_len*2, device=device, dtype=torch.int32)
        next_global_stats = torch.zeros(self.args.batch_size, 1, NUM_GLOBAL_STATS, device=device)
        next_local_stats = torch.zeros(self.args.batch_size, self.args.subgraph_len*2, NUM_LOCAL_STATS, device=device)
        next_masks = torch.zeros(self.args.batch_size, 1, 1, self.args.subgraph_len, device=device)
        
        next_state_mask = torch.zeros(self.args.batch_size, device=device, dtype=torch.bool)
        is_experts = torch.zeros(self.args.batch_size, dtype=torch.bool, device=device)
        gammas = torch.zeros(self.args.batch_size, device=device)

        # Unwrap the experiences
        for i, ex in enumerate(exs):
            # Create subgraph mask if edges less than subgraph length
            #if ex.state.subgraph.shape[1]//2 < self.args.subgraph_len:
                # Set edges that are null to 1 to mask out
            masks[i] = ex.state.mask
            # print("masks[i]", masks[i])
            # print("ex.state.mask", ex.state.mask)

            local_stats[i, :ex.state.local_stats.shape[1]] = ex.state.local_stats
            subgraphs[i, :ex.state.subgraph.shape[1]], global_stats[i], local_stats[i, :ex.state.local_stats.shape[1]] = ex.state.subgraph, ex.state.global_stats, ex.state.local_stats
            
            actions.append(ex.action)
            rewards[i] = self._reward_scaler.scale_reward(ex.reward)
            is_experts[i] = bool(ex.is_expert)
            gammas[i] = ex.gamma
            if ex.next_state is not None:
                next_subgraphs[i, :ex.next_state.subgraph.shape[1]] = ex.next_state.subgraph
                next_global_stats[i] = ex.next_state.global_stats
                next_local_stats[i, :ex.next_state.local_stats.shape[1]] = ex.next_state.local_stats
                next_state_mask[i] = True

                # Create subgraph mask if edges less than subgraph length
                # if ex.next_state.subgraph.shape[1]//2 < self.args.subgraph_len:
                    # Set edges that are null to 1 to mask out
                next_masks[i] = ex.next_state.mask

        states = State(subgraphs, global_stats, local_stats, masks)
        

        # Get nonempty next states
        next_states = State(
            next_subgraphs[next_state_mask],
            next_global_stats[next_state_mask],
            next_local_stats[next_state_mask],
            next_masks[next_state_mask])

        next_masks = next_masks[next_state_mask]
        return states, actions, rewards, next_states, next_state_mask, is_experts, gammas

    def train(self) -> float:
        """Train the model over a sampled batch of experiences.
        
        Returns:
            the loss for the batch
        """
        is_ws, exs, indices = self._memory.sample(self.args.batch_size, self._train_dict["episodes"])
        td_targets = torch.zeros(self.args.batch_size, device=device)
        
        states, actions, rewards, next_states, next_state_mask, is_experts, gammas = self._unwrap_exs(exs)

        # Select the q-value for every state
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        q_vals_matrix = self._sparrl_net(states)

        q_vals = q_vals_matrix.gather(1, actions.unsqueeze(1)).squeeze(1)

        # print("actions", actions, actions.shape)
        # print("states", states, states.subgraph.shape)
        # print("q_vals_matrix", q_vals_matrix, q_vals_matrix.shape)
        # print("q_vals", q_vals, q_vals.shape)
        # print("next_states", next_states, next_states.subgraph.shape, "\n\n")
        # print("\n\n")

        # Run policy on next states
        with torch.no_grad():
            q_next = self._sparrl_net(
                next_states)

            q_next_target = self._sparrl_net_tgt(
                next_states)
            

        # index used for getting the next nonempty next state
        q_next_idx = 0
        
        expert_margin_loss = 0

        # Number of terminated states elapsed thus far
        num_term_states = 0

        for i in range(self.args.batch_size):
            # Compute expert margin classification loss (i.e., imitation loss)
            if is_experts[i]:
                margin_mask = torch.ones(self.args.subgraph_len, device=device)
                # Mask out the expert action    
                margin_mask[actions[i]] = 0
                
                # Set to margin value
                margin_mask = margin_mask * self.args.expert_margin
                
                # Compute the expert imitation loss
                expert_margin_loss += torch.max(q_vals_matrix[i] + margin_mask) - q_vals[i]
            
            if not next_state_mask[i]:
                td_targets[i] = rewards[i]
                num_term_states += 1
            else:
                # Get the argmax next action for DQN
                valid_actions = self._get_valid_edges(next_states.subgraph[i - num_term_states])


                action = self._sample_action(
                    q_next_target[q_next_idx][valid_actions], True)
                action = int(valid_actions[action])

                # Set TD Target using the q-value of the target network
                # This is the Double-DQN target
                # print("q_next_target[q_next_idx, action]", q_next_target[q_next_idx, action])
                # print("gammas[i] * q_next_target[q_next_idx, action]", gammas[i] * q_next_target[q_next_idx, action])
                # print("rewards[i] + gammas[i] * q_next_target[q_next_idx, action]", rewards[i] + gammas[i] * q_next_target[q_next_idx, action],"\n")
                td_targets[i] = rewards[i] + gammas[i] * q_next_target[q_next_idx, action]
                q_next_idx += 1
 
        self._optim.zero_grad()

        # Compute L1 loss
        td_errors = td_targets  - q_vals
        #loss = torch.mean(td_errors.abs()  *  is_ws)
        #print("loss", loss)
        loss = torch.mean(td_errors ** 2  *  is_ws)

        self._memory.update_priorities(indices, td_errors.detach().abs(), is_experts)
        
        #print("L2 LOSS", torch.mean(td_errors ** 2  *  is_ws))
       
        
        loss += expert_margin_loss * self.args.expert_lam
        loss.backward()
        
        # Clip gradient
        nn.utils.clip_grad.clip_grad_norm_(
            self._sparrl_net.parameters(),
            self.args.max_grad_norm)

        #print("node_enc.node_embs.weight.grad", self._sparrl_net.node_enc.node_embs.weight.grad)
        # print("node_enc.node_embs.weight.grad", self._sparrl_net.node_enc.node_embs.weight.grad)
        # print("self._sparrl_net.q_fc_1.weight_mu.grad", self._sparrl_net.q_fc_1.weight.grad)
        # print("self._sparrl_net.q_fc_2.weight_mu.grad", self._sparrl_net.q_fc_2.weight.grad)

        # print("self._sparrl_net.q_fc_1.weight_sigma.grad", self._sparrl_net.q_fc_1.weight_sigma.grad, self._sparrl_net.q_fc_1.weight_mu.is_leaf)
        # print("self._sparrl_net.q_fc_2.weight_mu.grad", self._sparrl_net.q_fc_2.weight_mu.grad)
        # print("self._sparrl_net.q_fc_2.weight_sigma.grad", self._sparrl_net.q_fc_2.weight_sigma.grad)

        #print("self._sparrl_net.q_fc_2.bias_mu.grad", self._sparrl_net.q_fc_2.bias_mu.grad, self._sparrl_net.q_fc_2.bias_mu.is_leaf)
        # print("self._sparrl_net.q_fc_2.bias_sigma.grad", self._sparrl_net.q_fc_2.bias_sigma.grad, self._sparrl_net.q_fc_2.bias_sigma.is_leaf)
        # print("self._sparrl_net.q_fc_2.weight_epsilon.grad", self._sparrl_net.q_fc_2.weight_epsilon.grad, self._sparrl_net.q_fc_2.weight_epsilon.is_leaf)

        # print("self._sparrl_net.training", self._sparrl_net.training)
        # Train model
        self._optim.step()

        # Check if using decay and min lr not reached        
        # if self._train_dict["update_step"] < self.args.lr_warmup_steps:
        #     self._optim.param_groups[0]["lr"] = self.args.lr * (self._train_dict["update_step"] + 1) / self.args.lr_warmup_steps 
        # elif not self.args.no_lr_decay and self._optim.param_groups[0]["lr"] > self.args.min_lr:
        #     # If so, decay learning rate
        #     self._lr_scheduler.step()
        # else:
        #     self._optim.param_groups[0]["lr"] = self.args.min_lr
        #print("LR", self._optim.param_groups[0]["lr"])
        
        # Update train info
        self._train_dict["update_step"] += 1
        self._train_dict["mse_losses"].append(float(loss.detach()))
        # self._train_dict["sigma_mean_abs_1"].append(
        #     float(self._sparrl_net.v_fc_1.sigma_mean_abs()))
        # self._train_dict["sigma_mean_abs_2"].append(
        #     float(self._sparrl_net.v_fc_2.sigma_mean_abs()))

        # Update the DQN target parameters
        # if (self._train_dict["update_step"] + 1) % 16 == 0:
        self._update_target()
        
        # Print out q_values and td_targets for debugging/progress updates
        if (self._train_dict["update_step"] + 1) % 16 == 0:
            print("self.epsilon_threshold:", self.epsilon_threshold)
            print("q_next", q_next)
            print("q_next_target", q_next_target)

            print("loss", loss)
            print("expert_margin_loss", expert_margin_loss* self.args.expert_lam)
            print("q_values", q_vals)
            print("td_targets", td_targets)
            print("rewards", rewards)

        return float(loss.detach())

    def predict(self, state, argmax=False):
        batch_size = state.subgraph.shape[0]
        valid_actions = self._get_valid_edges(state.subgraph[0])
        q_vals = self._sparrl_net(state)
        return q_vals, valid_actions



    def __call__(self, state, argmax=False) -> int:
        """Make a sparsification decision based on the state.

        Returns:
            an edge index.
        """
        batch_size = state.subgraph.shape[0] 

        # Set for when experience is added
        self._should_add_expert_ex = self._train_dict["episodes"] < self.args.expert_episodes

        if self._should_add_expert_ex and not self.args.eval:
            # Run through expert policy
            action = self._expert_agent(state)
        else:
            # Get the q-values for the state
            q_vals = self._sparrl_net(state)
            # Sample an action (i.e., edge to prune)
            if batch_size > 1:
                action = []
                for i in range(batch_size):
                    valid_actions = self._get_valid_edges(state.subgraph[i])
                    cur_action = int(self._sample_action(q_vals[i][valid_actions], argmax))
                    action.append(int(valid_actions[cur_action])) 
            else:
                valid_actions = self._get_valid_edges(state.subgraph[0])
                action = self._sample_action(q_vals[valid_actions], argmax)
                action = int(valid_actions[action])

        return action
        