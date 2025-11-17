import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from tqdm import tqdm

from ..model.model import Transformer     

class Model(pl.LightningModule):

    def __init__(self, config, dimension, output_dims, vocab_size=23):
        
        super().__init__()
        self.model = Transformer(vocab_size, config['dimension'], config['ah'], config['output_dims'], config['dr'], config['beta'])
        self.output_dims = output_dims
        self.lr = config['lr']
        self.alpha = config['alpha']
        self.observed_loss = config['observed_loss']
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt_parameters = []
        named_parameters = list(self.named_parameters())
        
        init_lr = self.lr
        linear_lr = self.lr * 1.05
        no_decay = ['norm', 'bias']
        
        for i, (name, params) in enumerate(named_parameters):
            weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01
            
            if name.startswith('model.embeddings') or name.startswith('model.enc_1'):
                lr = init_lr * (0.9 ** 3)
                
                opt_parameters.append({"params": params,
                                       "weight_decay": weight_decay,
                                       "lr": lr}) 
                
            elif name.startswith('model.enc_2') :
                lr = init_lr * (0.9 ** 2)
                
                opt_parameters.append({"params": params,
                                       "weight_decay": weight_decay,
                                       "lr": lr})
                
            elif name.startswith('model.enc_3') :
                lr = init_lr * 0.9
                
                opt_parameters.append({"params": params,
                                       "weight_decay": weight_decay,
                                       "lr": lr})
                
            elif name.startswith('model.enc_4') :
                lr = init_lr
                
                opt_parameters.append({"params": params,
                                       "weight_decay": weight_decay,
                                       "lr": lr})
            
            elif 'linear' in name :
                lr = linear_lr
                
                opt_parameters.append({"params": params,
                                       "weight_decay": weight_decay,
                                       "lr": lr})
                
        self.optimizer = torch.optim.AdamW(opt_parameters, lr=init_lr)
        lr_lambda = lambda epoch : 0.95
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda)
        return {
            'optimizer': self.optimizer,
            'scheduler' : self.scheduler
       }
            

    def training_step(self, batch, batch_idx):
        x, label = batch
        
        final = self.model(x, mode='infer')
        
        observed_values = torch.where(observed, label, torch.zeros_like(label))
        observed_pred = torch.where(observed, final, torch.zeros_like(final))
        
        loss = nn.BCEWithLogitsLoss(reduction='sum')(observed_pred, observed_values) / torch.sum(observed)
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass
    
    
config = {
    'ah': 2,
    'dr': 0.1,
    'beta': 0.59,
    'lr': 8.75e-6,
    'observed_loss': 0.96,
    'alpha': 0.93,
    'dimension': 1024,
    'output_dims': [7, 72, 268, 4255]
}

model = Model(config, dimension=dimension, output_dims=output_dims)
callbacks = [StochasticWeightAveraging(swa_lrs=3e-7, swa_epoch_start=49, annealing_epochs=30), ModelCheckpoint(dirpath='./model/', filename='{epoch}', every_n_epochs=1, auto_insert_metric_name=True, save_top_k=-1)]
trainer = Trainer(max_epochs=80, gpus=4, enable_progress_bar=False, callbacks=callbacks, num_sanity_val_steps=0, strategy="ddp_find_unused_parameters_false", precision=16)

# Use your training data such as: data_module = YourDataModule(...)
trainer.fit(model, data_module)