import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder
from conf import *

class NodeEncoder(nn.Module):
    """Create node embedding using local statistics."""
    def __init__(self, args, num_nodes: int):
        super().__init__()
        self.args = args
        self.num_nodes = num_nodes + 1
        
        self.node_embs = nn.Embedding(self.num_nodes+1, self.args.emb_size) 
        
        self.fc_1 = nn.Linear(self.args.hidden_size + NUM_LOCAL_STATS, self.args.hidden_size)
        self.fc_2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        self.norm_1 = nn.LayerNorm(self.args.emb_size, eps=1e-7)
        self.dropout_1 = nn.Dropout(self.args.drop_rate)
        
        
    def forward(self, subgraph: torch.Tensor, local_stats: torch.Tensor):
        # Get initial node embeddings
        node_embs = self.node_embs(subgraph)

        # Combine node embeddings with current local statistics
        node_embs = self.fc_1(
            torch.cat((node_embs, local_stats), -1))
        node_embs = self.norm_1(node_embs)
        node_embs = F.relu(node_embs)
        node_embs = self.dropout_1(node_embs)

        # Create final node embeddings
        node_embs = self.fc_2(node_embs)
        
        return node_embs 

class EdgeEncoder(nn.Module):
    """Map node embedding to edge embedding."""
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Used to combine the embeddings
        self.edge_conv1d = nn.Conv1d(
            self.args.emb_size,
            self.args.emb_size,
            kernel_size=2,
            stride=2)
        self.norm_1 = nn.LayerNorm(self.args.emb_size, eps=1e-7)
        self.dropout_1 = nn.Dropout(self.args.drop_rate)

        # Used produce to produce the final edge embedding
        self.edge_fc = nn.Linear(self.args.emb_size, self.args.emb_size)

    def forward(self, node_embs: torch.Tensor):
        # Combine the node embeddings to create edge embeddings
        edge_embs = self.edge_conv1d(node_embs.transpose(1,2))
        edge_embs = self.norm_1(edge_embs.transpose(1,2))
        edge_embs = F.relu(edge_embs)
        edge_embs = self.dropout_1(edge_embs)
        
        # Create the final edge embeddings
        edge_embs = self.edge_fc(edge_embs)
        
        return edge_embs

class GlobalStatisticsEncoder(nn.Module):
    """Create an embedding from global statistics about the graph."""
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.fc_1 = nn.Linear(NUM_GLOBAL_STATS, self.args.hidden_size)
        self.fc_2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        self.norm_1 = nn.LayerNorm(self.args.emb_size, eps=1e-7)
        self.dropout_1 = nn.Dropout(self.args.drop_rate)

    def forward(self, global_stats):
        global_stats_emb = self.fc_1(global_stats)
        global_stats_emb = self.norm_1(global_stats_emb)
        global_stats_emb = F.relu(global_stats_emb)
        global_stats_emb = self.dropout_1(global_stats_emb)

        # Create final statistics embedding
        global_stats_emb = self.fc_2(global_stats_emb)
        
        return global_stats_emb
        
class SparRLNet(nn.Module):
    def __init__(self, args, num_nodes: int):
        super().__init__()
        self.args = args
        self.num_nodes = num_nodes
        
        
        self.node_enc = NodeEncoder(self.args, self.num_nodes)

        self.global_stats_enc = GlobalStatisticsEncoder(self.args)
        self.edge_mha_enc = Encoder(self.args)

        self.edge_enc = EdgeEncoder(self.args)

        # Mapping to q-values for pruning edges
        self.q_fc_1 = nn.Linear(self.args.hidden_size, 1)

    def forward(self, state, subgraph_mask=None):
        batch_size = state.subgraph.shape[0]

        # Create node embedding
        node_embs = self.node_enc(state.subgraph, state.local_stats)

        # Create edge embeddings
        edge_embs = self.edge_enc(node_embs)

        # Create global statistics embedding
        global_stats_emb = self.global_stats_enc(state.global_stats)

        # Extend mask to add place for global stats
        if subgraph_mask is not None:
            mask = torch.zeros(batch_size, 1, 1, subgraph_mask.shape[1] + 1, device=device)
            mask[:, 0, 0, :subgraph_mask.shape[1]] = subgraph_mask
        else:
            mask = None
        
        # Perform MHA over edge embeddings (batch size, # edges, hidden size)
        embs = self.edge_mha_enc(
            torch.cat((edge_embs, global_stats_emb), 1), mask)

        # Pass the edge embeddings through FC to get Q-values
        q_vals = self.q_fc_1(embs[:, :-1])

        if batch_size == 1:
            return q_vals.view(-1)
        else:
            return q_vals.view(q_vals.shape[0], q_vals.shape[1])