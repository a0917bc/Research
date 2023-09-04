import sys, math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.loss import LabelSmoothingCrossEntropy
from timm.data import Mixup
from timm import create_model

from qat.export.utils import replace_module_by_name, fetch_module_by_name
from operations.amm_linear import AMMLinear, PQLinear, TrivenLinear
from losses import DistillationLoss

def create_deit(start_replaced_layer_idx, end_replaced_layer_idx, model_name='deit3_small_patch16_224.fb_in1k', pretrained=True, subvec_len=32, k=16):
    model = create_model(model_name, pretrained=pretrained)
    ncodebooks = {
        "attn.qkv": 384 // subvec_len, # PQ配合 # of heads, In ViT, num_heads=12
        # "attn.proj": 384 // subvec_len,
        "mlp.fc1": 384 // subvec_len
        # "mlp.fc2": 1536 // subvec_len
    }
    
    for i in range(start_replaced_layer_idx, end_replaced_layer_idx): 
        for name in ncodebooks:
            layer = model.blocks[i]
            module = fetch_module_by_name(layer, name)
            amm_linear = AMMLinear(
                ncodebooks[name],
                module.in_features,
                module.out_features,
                module.bias is not None,
                k=k
            )
            amm_linear.inverse_temperature_logit.data.copy_(
                torch.tensor(10)
            )
            # print(amm_linear.weight.data.shape)
            # print(module.weight.data.shape)
            # weight = rearrange(module.weight.data, 'o i -> i o')
            # weight = rearrange(weight, '(c v) o -> c v o', c=ncodebooks[name], v=subvec_len)
            # amm_linear.weight.data.copy_(weight.data)
            # amm_linear.bias.data.copy_(module.bias.data)
            replace_module_by_name(layer, name, amm_linear)
            
    return model

class LUT_DeiT(L.LightningModule):
    def __init__(self, 
                 kmeans_init=True, 
                 start_replaced_layer_idx=0, 
                 end_replaced_layer_idx=12, 
                 num=1024,
                 model_name="deit3_small_patch16_224.fb_in1k",
                 distillation_type="hard",
                 tau=1/2,
                 alpha=1/2,
                 smoothing=0.1,
                 adam_epsilon: float = 1e-8,
                 lr=5e-4, 
                 weight_decay=0.1
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_deit(start_replaced_layer_idx, end_replaced_layer_idx, model_name=model_name)
        if kmeans_init:
            from pathlib import Path   
            save_path = Path('/home/u1887834/Research/base_model')
            self.model.load_state_dict(torch.load(save_path / f"{num}_base_{start_replaced_layer_idx}_{end_replaced_layer_idx}.pt"))
        
        # float_model
        float_model = create_model(model_name=model_name, pretrained=True)
        float_model.eval()
        for param in float_model.parameters():
            param.requires_grad = False

        float_model = torch.compile(float_model)
        # float_model
        
        self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.distil_loss = DistillationLoss(base_criterion=self.criterion, 
                                            teacher_model=float_model, 
                                            distillation_type=distillation_type,
                                            alpha=alpha,
                                            tau=tau)
    # @torchsnooper.snoop()
    def forward(self, x):
        return self.model(x)

    def common_step_v2(self, x, y, stage):
        logits = self(x)
        base_loss = self.criterion(logits, y)
        loss = self.distil_loss(x, logits, y)
        
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log(f"{stage}/loss", loss, 
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/base_loss", base_loss, 
                 on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, 
                 on_step=True, on_epoch=True, sync_dist=True)
        # self.current_epoch
        return loss
    def common_step(self, x, y, stage):
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log(f"{stage}/loss", loss, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, sync_dist=True)
        return loss
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step_v2(x, y, "train")
        # if not math.isfinite(loss.item()):
            # print("Loss is {}, stopping training".format(loss.item()))
            # sys.exit(1)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step(x, y, "val")
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        # Initialize lists to hold parameters for each group
        default_params = []
        special_linear_params = []
        no_decay_params = []

        # Define the names or substrings that should not have weight decay
        no_decay = ["bias", "LayerNorm.weight"]

        # Iterate through all named parameters
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
                # Parameters that should not have weight decay
                no_decay_params.append(param)
            elif 'inverse_temperature_logit' in name:  # Replace with the actual name or substring
                # Parameters in nn.Linear that should have a special learning rate
                special_linear_params.append(param)
            else:
                # All other parameters
                default_params.append(param)

        # Create the list of parameter groups
        optimizer_grouped_parameters = [
            {'params': default_params, 'weight_decay': self.hparams.weight_decay},  # Default learning rate and weight decay
            {'params': no_decay_params, 'weight_decay': 0.0},  # No weight decay
            {'params': special_linear_params, 'lr': 0.1}  # Special learning rate for specific nn.Linear parameters
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)

        
        return optimizer # TODO scheduler...


