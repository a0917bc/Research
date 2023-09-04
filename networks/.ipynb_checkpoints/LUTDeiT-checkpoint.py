from timm import create_model

import lightning as L
from qat.export.utils import replace_module_by_name, fetch_module_by_name
from operations.amm_linear import AMMLinear, PQLinear, TrivenLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

def create_deit(replacedLayer, stop, model_name='deit3_small_patch16_224.fb_in1k', pretrained=True, subvec_len=32, k=16):
    model = create_model(model_name, pretrained=pretrained)
    # print(replacedLayer)
    # exit()
    ncodebooks = {
        "attn.qkv": 384 // subvec_len, # PQ配合 # of heads, In ViT, num_heads=12
        # "attn.proj": 384 // subvec_len,
        "mlp.fc1": 384 // subvec_len
        # "mlp.fc2": 1536 // subvec_len
    }
    
    for i in range(replacedLayer, stop): # 
        for name in ncodebooks:
            # print(i, name)
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

# 使用方式
# custom_model = create_custom_model()
class LUT_DeiT(L.LightningModule):
    def __init__(self, 
                 pretrained=True, 
                 start_replaced_layer_idx=0, 
                 end_replaced_layer_idx=12, 
                 lr=0.001, 
                 num=1024):
        super().__init__()
        self.lr = lr
        self.model = create_deit(start_replaced_layer_idx, end_replaced_layer_idx)
        from pathlib import Path
         
        save_path = Path('/home/u1887834/Research/base_model')
        if pretrained:
            print("here")
            self.model.load_state_dict(torch.load(save_path / f"{num}_base_{start_replaced_layer_idx}_{end_replaced_layer_idx}.pt"))
        # self.model = torch.compile(self.model)
        self.linear_params_except_inverse_temp = [p for name, p in self.model.named_parameters() if 'inverse_temperature_logit' not in name]
        self.inverse_temp = [p for name, p in self.model.named_parameters() if 'inverse_temperature_logit' in name]
        print(self.linear_params_except_inverse_temp)
        print(self.inverse_temp)
        self.criterion = nn.CrossEntropyLoss()
    # @torchsnooper.snoop()
    def forward(self, x):
        return self.model(x)

    def common_step(self, x, y, stage):
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log(f"{stage}/loss", loss, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step(x, y, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step(x, y, "val")
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.linear_params_except_inverse_temp},  # 使用預設學習率
            {'params': self.inverse_temp, 'lr': 0.01}  # 使用特定的學習率
        ], lr=0.001)
        return optimizer


