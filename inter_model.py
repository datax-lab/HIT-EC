import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from tqdm import tqdm
import math

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class TransformerEmbedding(nn.Module): 

    def __init__(self, vocab_size, embed_size, dropout):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=embed_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)
    
class MultiHeadedAttention(nn.Module):

    def __init__(self, h=8, d_model=256, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        
        self.d_k = d_model // h
        self.h = h
        self.dim = d_model
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model*h)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def transpose_for_scores_value(self, x):
        new_x_shape = x.size()[:-1] + (self.h, self.dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        
        x = self.norm(x)
        
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores_value(value)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) 
        scores = scores / math.sqrt(query.size(-1))

        p_attn = self.softmax(scores)
        self.save_attn(p_attn)
        p_attn.register_hook(self.save_attn_gradients)

        x = torch.matmul(p_attn, value)
        batch_size = query.size(0)
        x = x.view(-1, batch_size, self.dim, self.h)
        x = reduce(torch.add,[x[:,:,:,i] for i in range(x.size(3))])
        x = x.view(batch_size, -1, self.dim)
        
        return x
    
    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
class SublayerConnection(nn.Module):
    
    def __init__(self, size, feed_forward_hidden, dropout):
        super(SublayerConnection, self).__init__()
        self.feed_forward = PositionwiseFeedForward(d_model=size, d_ff=feed_forward_hidden, dropout=dropout)
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        return x1 + self.dropout(self.feed_forward(self.norm(x2)))
    
class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(attn_heads, hidden, dropout)
        self.output_sublayer = SublayerConnection(hidden, feed_forward_hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x1 = self.attention(x, mask)
        x = x + x1
        x = self.output_sublayer(x, x)
        return self.dropout(x)
    
class Transformer(nn.Module) :
    
    def __init__(self, vocab_size, dimension, attn_heads, output_dims, dropout, beta) :
        super().__init__()
        self.embeddings = TransformerEmbedding(vocab_size, dimension, dropout)
        self.enc_1 = TransformerBlock(dimension, attn_heads, dimension*4, dropout)
        self.enc_2 = TransformerBlock(dimension, attn_heads, dimension*4, dropout)
        self.enc_3 = TransformerBlock(dimension, attn_heads, dimension*4, dropout)
        self.enc_4 = TransformerBlock(dimension, attn_heads, dimension*4, dropout)
        self.linear1 = nn.Linear(dimension, output_dims[0])
        self.linear2 = nn.Linear(dimension, output_dims[1])
        self.linear3 = nn.Linear(dimension, output_dims[2])
        self.linear4 = nn.Linear(dimension, output_dims[3])
        self.linear5 = nn.Linear(dimension, np.sum(output_dims))
        self.beta = beta
        
    def forward(self, x):
        local = []
        mask = None
        emb = self.embeddings(x)
        x = self.enc_1(emb, mask)
        local.append(self.linear1(x[:,0,:]))
        x = self.enc_2(x + emb, mask)
        local.append(self.linear2(x[:,0,:]))
        x = self.enc_3(x + emb, mask)
        local.append(self.linear3(x[:,0,:]))
        x = self.enc_4(x + emb, mask)
        local.append(self.linear4(x[:,0,:]))
        global_p = torch.sigmoid(self.linear5(x[:,0,:]))
        final = self.beta * global_p + (1 - self.beta) * torch.sigmoid(torch.cat(local, dim=1))
        return final
    
class InterpretationModel(pl.LightningModule):

    def __init__(self, config, dimension=1024, vocab_size=23):
        
        super().__init__()
        self.model = Transformer(vocab_size, dimension, config["ah"], config["output_dims"], config["dr"], config["beta"])
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        pass
            

    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass