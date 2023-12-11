from argparse import Namespace

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import RNA_Model
from utils import sort_weight_decay_params
from apex.optimizers import FusedAdam as Adam


class RNA_Lightning(pl.LightningModule):
    def __init__(self, hp: Namespace):
        super(RNA_Lightning, self).__init__()
        self.save_hyperparameters(hp)
        self.hp = self.hparams
        self.model = RNA_Model(**vars(hp))

    def configure_optimizers(self):
        if self.hp.wt_decay:
            decay, no_decay = sort_weight_decay_params(self.model)
            opt_groups = [
                {"params": decay, "weight_decay": self.hp.wt_decay},
                {"params": no_decay, "weight_decay": 0},
            ]
            opt = Adam(opt_groups)
        else:
            opt = Adam(self.model.parameters(), weight_decay=0)
        return opt

    def on_train_start(self):
        self.n_steps = self.trainer.estimated_stepping_batches
        self.n_warmup_steps = self.n_steps * self.hp.lr_warmup
        self.hp_metric, self.val_cache = 0.2, []
        self.logger.log_hyperparams(self.hp, {"hp/metric": self.hp_metric})

    def on_train_epoch_start(self):
        t = self.trainer
        self.logger.log_metrics({"hp/epoch": t.current_epoch}, t.global_step)

    def loss_L1(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        l = F.l1_loss(x, y, reduction="none")
        return l[~torch.isnan(l)].mean()

    def loss_CE(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ls = self.hp.aux_smooth if self.training else 0
        x = x.transpose(1, 2)
        l = F.cross_entropy(x, y, reduction="none", label_smoothing=ls, ignore_index=0)
        return l[~torch.isnan(l)].mean()

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    def fit_forward(self, x: dict, batch: dict):
        log_prefix = "loss/T" if self.training else "loss/V"
        x["react"] = x["react"] if self.training else x["react"].clip(0, 1)
        react = self.loss_L1(x["react"], batch["react"])
        loop = (
            self.loss_CE(x["loop"], batch["loop"])
            if self.hp.aux_loop is not None
            else 0
        )
        log_d = {log_prefix: react}
        log_d = log_d | {f"{log_prefix}/loop": loop} if self.hp.aux_loop else log_d
        self.log_dict(log_d, on_step=False, on_epoch=True, add_dataloader_idx=False)
        return react + loop * self.hp.aux_scale

    def update_LR(self):
        lr_m = inv_sqrt_sched(self.trainer.global_step, self.n_warmup_steps)
        self.log("hp/lr", (lr := self.hp.lr * lr_m))
        for p in self.optimizers().optimizer.param_groups:
            p["lr"] = lr

    def training_step(self, batch, batch_idx):
        self.update_LR()
        return self.fit_forward(self(batch), batch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x = self(batch)
        if dataloader_idx:
            mask, lmax = batch["mask"].sum(1), batch["seq"].size(1)
            for k, v in self.val_cache.pop(0).items():
                v = [v[i][: mask[i]].flip(0) for i in range(v.size(0))]
                v = [F.pad(s, [0] * 3 + [lmax - len(s)]) for s in v]
                x[k] = (x[k] + torch.stack(v)) / 2
        elif self.hp.val_flip:
            return self.val_cache.append(x)
        self.fit_forward(x, batch)

    def on_validation_end(self):
        l = self.trainer.logged_metrics["loss/V"].item()
        if l < self.hp_metric:
            self.hp_metric = l
            self.logger.log_metrics({"hp/metric": l}, self.trainer.global_step)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, mask, loss = self(batch), batch["mask"], None
        x["react"] = x["react"].clip(0, 1)
        loss = None
        if "loop" in x:
            loss = self.loss_CE(x["loop"], batch["loop"])
            loss = loss.unsqueeze(0)
        if dataloader_idx:
            mask = mask.flip(1)
            x = {k: v.flip(1) for k, v in x.items()}
        x = {k: v[mask] for k, v in x.items()}
        return x if loss is None else x | {"loss": loss}


def inv_sqrt_sched(current_step: int, num_warmup_steps: int, timescale=None) -> float:
    timescale = num_warmup_steps if timescale is None else timescale
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / (((current_step + shift) / timescale) ** 0.5)
    return decay
