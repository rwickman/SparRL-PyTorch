import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from typing import List

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Size of each hidden vector of a head
        self.depth = self.args.hidden_size // self.args.num_heads

        self.W_q = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.W_k = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.W_v = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        # W for multi-head attention
        self.mha_W = nn.Linear(self.args.hidden_size, self.args.hidden_size)

    def split_heads(self, x):
        # (batch size, num tokens,  num heads, depth)
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, -1, self.args.num_heads, self.depth))
        
        # (batch size, num heads, num tokens, depth)
        return x.transpose(2,1)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))
        
        qk = torch.matmul(q, k.transpose(3,2))
        dk = torch.tensor(q.shape[-1], dtype=torch.float32)
        
        scaled_att_logits = qk / torch.sqrt(dk)
        if mask is not None:
            scaled_att_logits += (mask * -1e9)
            
        att_weights = F.softmax(scaled_att_logits, dim=-1)
        # print("mask", mask)
        # print("scaled_att_logits", scaled_att_logits)
        #print("att_weights", att_weights)
        scaled_att = torch.matmul(att_weights, v)

        # (batch_size, fbs+hidden+enc_state, num_heads, depth)
        scaled_att = scaled_att.transpose(2,1)

        # Squeeze MHA together for each embedding
        scaled_att = torch.reshape(scaled_att, (batch_size, -1, self.args.hidden_size))

        # Combine the MHA for each embedding
        
        out = self.mha_W(scaled_att)
        return out, att_weights

class PointWiseFFN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(self.args.hidden_size, self.args.dff)
        self.fc2 = nn.Linear(self.args.dff, self.args.hidden_size)
       
    def forward(self, x):
        pw_out = F.relu(self.fc1(x))
        pw_out = self.fc2(pw_out)
       
        return pw_out


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.mha = MultiHeadAttention(self.args)

        self.pw_ffn = PointWiseFFN(self.args)

        self.dropout1 = nn.Dropout(self.args.drop_rate)
        self.dropout2 = nn.Dropout(self.args.drop_rate)

        self.norm1 = nn.LayerNorm(self.args.hidden_size, eps=1e-10)
        self.norm2 = nn.LayerNorm(self.args.hidden_size, eps=1e-10)

    def forward(self, x, mask=None):
        # Perform MHA attention
        att_out, _ = self.mha(x, x, x, mask)
        att_out = self.dropout1(att_out)
        mha_out = self.norm1(x + att_out)

        # Plug through Point-wise FFN
        pw_ffn_out = self.pw_ffn(mha_out)
        pw_ffn_out = self.dropout2(pw_ffn_out)
        out = self.norm2(mha_out + pw_ffn_out)
        
        return out

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.enc_layers = nn.ModuleList([EncoderLayer(self.args) for _ in range(self.args.num_enc_layers)])
        
        self.dropout = nn.Dropout(self.args.drop_rate)

    def forward(self, x, mask=None):
        """
            x: task emb, example emb, used fbs embs
        """
        #x = x * torch.sqrt(torch.tensor(self.args.hidden_size, dtype=torch.float32)).cuda()
        #print("PREV X: ", x)
        x = self.dropout(x)
        # print("NOW X: ", x)


        # Run through all encoder fb
        #i = 0
        for enc_layer in self.enc_layers:
            #print("layer", i)
            x = enc_layer(x, mask)
            #i += 1

        # Return output of last encoder fb
        return x