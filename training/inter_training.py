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

# output_dims = ... Define the number of classes like [# of classes for level 1, # of classes for level 2, # of classes for level 3, # of classes for level 4]

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

# Use your training data such as: data_module = YourDataModule(...)
trainer.fit(model, data_module)
