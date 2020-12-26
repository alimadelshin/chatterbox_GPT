import torch
from torch.nn import functional
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
import copy

class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps


    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        
        return norm




class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 255):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
        

    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], \
        requires_grad=False)

        return x



def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.gelu = GELU()
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(self.gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
# calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, rate=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, look_ahead_mask = None):

        x2 = self.norm_1(x)
        x =  x + self.dropout1(self.mha1(x2, x2, x2,look_ahead_mask))

        x2 = self.norm_2(x)
        x =  x  + self.dropout2(self.ffn(x2)) 

        return x

class LLayer(nn.Module):
    def __init__(self, d_model, rate=0.1):
        super().__init__()

        self.linear = nn.Linear(d_model, d_model)
        self.ffn = FeedForward(d_model) 

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        self.dropout_1 = nn.Dropout(rate)
        self.dropout_2 = nn.Dropout(rate)

    def forward(self, x):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.linear(x2))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ffn(x2))

        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class L(nn.Module):
    def __init__(self, d_model, num_layers = 1):
        super().__init__()
        self.num_layers = num_layers
        self.llayers = get_clones(LLayer(d_model), num_layers)
    
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.llayers[i](x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, vocab_size, rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dec_layers = get_clones(DecoderLayer(d_model, num_heads,  rate), num_layers)

        

    def forward(self, x, look_ahead_mask=None):
        
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, look_ahead_mask)            
        
        return x

class GTP2(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, vocab_size, rate=0.1):
        super().__init__() 
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.dropout = nn.Dropout(rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, vocab_size, rate)
        self.llf = L(d_model)
        self.final_layernorm = Norm(d_model)
        self.final_layer = nn.Linear(d_model, vocab_size)

        
    
    def forward(self, x, look_ahead_mask = None):
        
        x = self.embed(x)
        x = self.pe(x)
        
        x = self.dropout(x)
        
        dec_output = self.decoder(x, look_ahead_mask)
        ll_output = self.llf(dec_output)
        dec_output_norm = self.final_layernorm(ll_output)

        final_output = self.final_layer(dec_output_norm)

        return final_output


