import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformer import Encoder
from conf import *


class NoisyLinear(nn.Module):
    def __init__(self, args, in_features, out_features):
        super().__init__()
        self.args = args
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        if not self.args.load:
            self.reset_parameters()

        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.args.noise_std / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.args.noise_std / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    
    def sigma_mean_abs(self):
        return self.weight_sigma.abs().mean()

class NodeEncoder(nn.Module):
    """Create node embedding using local statistics."""
    def __init__(self, args, num_nodes: int):
        super().__init__()
        self.args = args
        self.num_nodes = num_nodes
        
        self.node_embs = nn.Embedding(self.num_nodes+1, self.args.hidden_size) 
        #self.norm_1 = nn.utils.weight_norm(self.node_embs)
        self.norm_1 = nn.BatchNorm1d(self.args.hidden_size)
        # if not self.args.load and self.args.node_embs:
        #     self.load_pretrained_embs()

        if self.args.node_embs:
            self.load_pretrained_embs()
        
        #self.node_fc = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        # self.node_fc = nn.BatchNorm1d(self.args.hidden_size)
        self.fc_1 = nn.Linear(self.args.hidden_size + NUM_LOCAL_STATS + 1, self.args.hidden_size)
        self.fc_2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        #self.norm_1 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)
        # self.norm_2 = nn.LayerNorm(self.args.hidden_size, eps=1e-10)
        self.dropout_1 = nn.Dropout(self.args.drop_rate)
        # self.dropout_2 = nn.Dropout(self.args.drop_rate)
        
    def load_pretrained_embs(self):
        weights_dict = torch.load(self.args.node_embs)
        # Add pad embedding
        pretrained_node_embs = torch.cat((self.node_embs(torch.tensor([0])), weights_dict["node_embs"])) 
        self.node_embs = self.node_embs.from_pretrained(pretrained_node_embs, freeze=True)

    def forward(self, subgraph: torch.Tensor, local_stats: torch.Tensor):
        # Get initial node embeddings
        batch_size = subgraph.shape[0]
        node_embs = self.node_embs(subgraph)
        
        node_embs = self.norm_1(node_embs.reshape(-1, self.args.hidden_size))
        node_embs = node_embs.reshape(batch_size, -1, self.args.hidden_size)
        #node_embs = self.norm_1(node_embs)
        #node_embs = self.dropout_1(node_embs)
        # node_embs = self.norm_1(node_embs)
        #node_embs = (node_embs)
        
        # Combine node embeddings with current local statistics
        node_embs = F.relu(self.fc_1(
            torch.cat((node_embs, local_stats), -1)))
        #node_embs = self.norm_1(node_embs)
        #node_embs = F.relu(node_embs)

        # Create final node embeddings
        node_embs = F.relu(self.fc_2(node_embs))
        # node_embs = self.norm_1(node_embs)
        #node_embs = self.dropout_1(node_embs)
        #node_embs = F.relu(node_embs)
        
        return node_embs 

class EdgeEncoder(nn.Module):
    """Map node embedding to edge embedding."""
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Used to combine the embeddings
        # self.edge_conv1d = nn.Conv1d(
        #     self.args.hidden_size,
        #     self.args.hidden_size,
        #     kernel_size=2,
        #     stride=2)
        self.edge_fc_1 = nn.Linear(self.args.hidden_size * 2, self.args.hidden_size * 2)
        self.edge_fc_2 = nn.Linear(self.args.hidden_size * 2, self.args.hidden_size)
        #self.norm_1 = nn.LayerNorm(self.args.hidden_size, eps=1e-10)
        #self.dropout_1 = nn.Dropout(self.args.drop_rate)

        # self.norm_1 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)
        # # self.norm_2 = nn.LayerNorm(self.args.hidden_size, eps=1e-10)
        # self.dropout_1 = nn.Dropout(self.args.drop_rate)
        # self.dropout_2 = nn.Dropout(self.args.drop_rate)

        # Used produce to produce the final edge embedding
        self.edge_fc_3 = nn.Linear(self.args.hidden_size, self.args.hidden_size)

    def forward(self, node_embs: torch.Tensor):
        # Combine the node embeddings to create edge embeddings
        #print("node_embs", node_embs)
        node_embs = node_embs.reshape(node_embs.shape[0], -1, self.args.hidden_size * 2)
        #print("node_embs", node_embs)
        #print("node_embs.count_nonzero()", node_embs.count_nonzero(), node_embs.shape)
        edge_embs = F.relu(self.edge_fc_1(node_embs))
        edge_embs = F.relu(self.edge_fc_2(edge_embs))

        #print("edge_embs.count_nonzero()", edge_embs.count_nonzero())
        # edge_embs = self.norm_1(edge_embs.transpose(1,2))
        # edge_embs = self.dropout_1(edge_embs)
        
        # Create the final edge embeddings
        edge_embs = F.relu(self.edge_fc_3(edge_embs))
        # edge_embs = self.norm_1(edge_embs)
        #edge_embs = self.dropout_1(edge_embs)
        # edge_embs = self.edge_fc(edge_embs)
        # edge_embs = self.norm_2(edge_embs)
        # edge_embs = self.dropout_2(edge_embs)

        return edge_embs

        
class SparRLNet(nn.Module):
    def __init__(self, args, num_nodes: int):
        super().__init__()
        self.args = args
        self.num_nodes = num_nodes
        
        
        self.node_enc = NodeEncoder(self.args, self.num_nodes)
        self.edge_enc = EdgeEncoder(self.args)
        # self.global_stats_enc = GlobalStatisticsEncoder(self.args)
        # self.norm_1 = nn.LayerNorm(self.args.hidden_size)
        # self.edge_mha_enc = Encoder(self.args)

        

        # # self.share_gate = nn.Linear(self.args.hidden_size, 1)
        # # self.share_gate_act = nn.Sigmoid()
        
        # Mapping to q-values for pruning edges        
        #self.q_fc_1 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        #self.v_fc_2 = nn.Linear(self.args.hidden_size, 1)
        
        # self.v_fc_1 = NoisyLinear(self.args, self.args.hidden_size, self.args.hidden_size)
        # self.v_fc_2 = NoisyLinear(self.args, self.args.hidden_size, 1)

        # self.adv_fc_1 = NoisyLinear(self.args, self.args.hidden_size, self.args.hidden_size)
        # self.adv_fc_2 = NoisyLinear(self.args, self.args.hidden_size, 1)


        # self.v_fc_1 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        # self.v_fc_2 = nn.Linear(self.args.hidden_size, 1)

        # self.adv_fc_1 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        # self.adv_fc_2 = nn.Linear(self.args.hidden_size, 1)


        self.q_fc_1 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.q_fc_2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.q_fc_3 = nn.Linear(self.args.hidden_size, 1)
        # self.q_fc_3 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        # self.q_fc_4 = nn.Linear(self.args.hidden_size, 1)
        #self.q_fc_1 = NoisyLinear(self.args, self.args.hidden_size, self.args.hidden_size)
        #self.q_fc_2 = NoisyLinear(self.args, self.args.hidden_size, 1)

    def reset_noise(self):
        pass
        # self.v_fc_1.reset_noise()
        # self.v_fc_2.reset_noise()

        # self.adv_fc_1.reset_noise()
        # self.adv_fc_2.reset_noise()
        #self.q_fc_1.reset_noise()
        #self.q_fc_2.reset_noise()

    def forward(self, state) -> torch.Tensor:
        batch_size = state.subgraph.shape[0]

        # Create node embedding
        node_embs = self.node_enc(state.subgraph, torch.cat((state.local_stats, state.global_stats.repeat(1, state.local_stats.shape[1], 1)), -1))

        # Create edge embeddings
        embs = self.edge_enc(node_embs)

        # Create global statistics embedding
        # global_stats_emb = self.global_stats_enc(state.global_stats)
        
        # Perform MHA over edge embeddings (batch size, # edges, hidden size)
        # embs = self.norm_1(embs)
        # embs = self.edge_mha_enc(
        #     embs, state.mask) 
        
        # # # Pass the edge embeddings through FC to get Q-values
        # v_vals = F.relu(self.v_fc_1(embs))
        # v_vals = self.v_fc_2(v_vals)
        
        # # # # Pass the edge embeddings through FC to get advantage values
        # adv_vals = F.relu(self.adv_fc_1(embs))
        # adv_vals = self.adv_fc_2(adv_vals)

        # # # Derive the Q-values
        # q_vals = v_vals + adv_vals - adv_vals.mean(dim=-1, keepdim=True)
        
        # share_vals = self.share_gate_act(self.share_gate(embs))
        # # print("share_vals", share_vals)
        # embs = (1-share_vals) * edge_embs + share_vals * embs 

        q_vals = F.relu(self.q_fc_1(embs))
        q_vals = F.relu(self.q_fc_2(q_vals))

        # print("q_vals", q_vals)
        q_vals = self.q_fc_3(q_vals)
        
        if batch_size == 1:
            return q_vals.view(-1)
        else:
            # print(state)
            # print("q_vals", q_vals)
            # print("q_vals.shape", q_vals.shape)
            # print("embs.shape", embs.shape, "\n\n")
            #print("q_vals.view(q_vals.shape[0], q_vals.shape[1])", q_vals.view(q_vals.shape[0], q_vals.shape[1]))
            #print("q_vals", q_vals)
            return q_vals.view(q_vals.shape[0], q_vals.shape[1])