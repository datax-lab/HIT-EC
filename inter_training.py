import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from tqdm import tqdm

        
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

        x = torch.matmul(p_attn, value)
        batch_size = query.size(0)
        x = x.view(-1, batch_size, self.dim, self.h)
        x = reduce(torch.add,[x[:,:,:,i] for i in range(x.size(3))])
        x = x.view(batch_size, -1, self.dim)
        
        return x
        
class InterAttention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    
class InterMultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = InterAttention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

    
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
        self.inter_attention = InterMultiHeadedAttention(attn_heads, hidden, dropout)
        self.output_sublayer = SublayerConnection(hidden, feed_forward_hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, mode):
        if mode == 'infer' : 
            x1_infer = self.attention(x, mask)
            x_infer = x + x1_infer
            x_infer = self.output_sublayer(x_infer, x_infer)
            x_infer = self.dropout(x_infer)
            return x_infer
        
        elif mode == 'inter' : 
            x1_inter = self.inter_attention(x, x, x, mask)
            x_inter = x + x1_inter
            x_inter = self.output_sublayer(x_inter, x_inter)
            x_inter = self.dropout(x_inter)
            return x_inter
    
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
        
    def forward(self, x, mode):
        local = []
        mask = None
        
        if mode == 'infer': 
            emb = self.embeddings(x)
            x = self.enc_1(emb, mask, mode='infer')
            local.append(self.linear1(x[:,0,:]))
            x = self.enc_2(x + emb, mask, mode='infer')
            local.append(self.linear2(x[:,0,:]))
            x = self.enc_3(x + emb, mask, mode='infer')
            local.append(self.linear3(x[:,0,:]))
            x = self.enc_4(x + emb, mask, mode='infer')
            local.append(self.linear4(x[:,0,:]))
            global_p = self.linear5(x[:,0,:])
            final = self.beta * global_p + (1 - self.beta) * torch.cat(local, dim=1)
            return final
        
        if mode == 'inter': 
            emb = self.embeddings(x)
            x = self.enc_1(emb, mask, mode='inter')
            local.append(self.linear1(x[:,0,:]))
            x = self.enc_2(x + emb, mask, mode='inter')
            local.append(self.linear2(x[:,0,:]))
            x = self.enc_3(x + emb, mask, mode='inter')
            local.append(self.linear3(x[:,0,:]))
            x = self.enc_4(x + emb, mask, mode='inter')
            local.append(self.linear4(x[:,0,:]))
            global_p = self.linear5(x[:,0,:])
            final = self.beta * global_p + (1 - self.beta) * torch.cat(local, dim=1)
            return final
    
class Model(pl.LightningModule):

    def __init__(self, config, dimension=1024, output_dims=[7, 71, 256, 3567], vocab_size=23):
        
        super().__init__()
        self.model = Transformer(vocab_size, dimension, config["ah"], output_dims, config["dr"], config["beta"])
        self.output_dims = output_dims
        self.lr = config["lr"]
        self.alpha = config["alpha"]
        self.observed_loss = config["observed_loss"]
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.automatic_optimization = False

        self.optimizer = None 
        self.inter_scheduler = None
        self.inter_optimizer = None 
    
    def forward(self, x, mode="infer"):
        return self.model(x, mode=mode)
    
    @staticmethod
    def _nan_to_zero_grads(params):
        for p in params:
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _any_nonfinite(params):
        for p in params:
            if (p is not None) and (
                (p.data is not None and not torch.isfinite(p.data).all()) or
                (p.grad is not None and not torch.isfinite(p.grad).all())
            ):
                return True
        return False


    def _inter_attention_params(self):
        params = []
        for k in [1, 2, 3, 4]:
            blk = getattr(self.model, f"enc_{k}")
            params += list(blk.inter_attention.parameters())
        return params

    def configure_optimizers(self):
        inter_params = self._inter_attention_params()
        self.inter_optimizer = torch.optim.AdamW(
            inter_params, lr=self.lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-6
        )
        self.inter_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            self.inter_optimizer, lr_lambda=lambda epoch: 0.95
        )
        return self.inter_optimizer

            

    def training_step(self, batch, batch_idx):
        x, label = batch 

        opt_inter = self.optimizers()
        if isinstance(opt_inter, (list, tuple)):  
            opt_inter = opt_inter[0]

        logits_inter = self.forward(x, mode="inter")
        loss_inter = self.criterion(logits_inter, label.float())

        self.toggle_optimizer(opt_inter)
        opt_inter.zero_grad(set_to_none=True)
        self.manual_backward(loss_inter)
        
        self._nan_to_zero_grads(self._inter_attention_params())

        torch.nn.utils.clip_grad_norm_(self._inter_attention_params(), max_norm=1.0)

        if self._any_nonfinite(self._inter_attention_params()):
            opt_inter.zero_grad(set_to_none=True)
        else:
            opt_inter.step()
    
        opt_inter.step()
        self.untoggle_optimizer(opt_inter)

        return loss_inter
    
    def on_train_epoch_end(self):
        if self.inter_scheduler is not None:
            self.inter_scheduler.step()
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass
    
def _finite(x): 
    return torch.isfinite(x).all()

def _any_inf_or_nan(params):
    for p in params:
        if p is not None:
            if (p.data is not None and not _finite(p.data)) or \
               (p.grad is not None and not _finite(p.grad)):
                return True
    return False
    
    
config = {
    'ah': 2,
    'dr': 0.1,
    'beta': 0.59,
    'lr': 8.75e-5,
    'observed_loss': 0.96,
    'alpha': 0.93
}

model = Model(config, output_dims=output_dims) 

CKPT_PATH = './GitHub_HIT-EC/model.ckpt'
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state = ckpt.get("state_dict", ckpt)
ik = model.load_state_dict(state, strict=False)

print(f"[InterOnly] Loaded from {CKPT_PATH}")
if ik.missing_keys:
    print(f"[InterOnly] Missing keys: {len(ik.missing_keys)} (expected if some heads are absent)")
if ik.unexpected_keys:
    print(f"[InterOnly] Unexpected keys: {len(ik.unexpected_keys)}")

for n, p in model.model.named_parameters():
    train_inter = ("inter_attention" in n)
    p.requires_grad = bool(train_inter)

callbacks = [
    ModelCheckpoint(
        dirpath='./inter_model/',
        filename='{epoch}',
        every_n_epochs=1,
        auto_insert_metric_name=True,
        save_top_k=-1
    ),
]

trainer = Trainer(
    max_epochs=80,
    accelerator="gpu",
    devices=4,
    enable_progress_bar=False,
    callbacks=callbacks,
    num_sanity_val_steps=0,
    strategy=DDPStrategy(find_unused_parameters=False),
    precision="16-mixed",
)

data_module = ECDataModule(16, output_dims)
trainer.fit(model, data_module)
