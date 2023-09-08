from timm import create_model

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightningWrapper(L.LightningModule):
    def __init__(self, 
                 model,
                 lr=0.001
                 ):
        super().__init__()
        # self.pretrained = pretrained
        self.lr = lr
        
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
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
            # outputs = model(samples)
            # if not args.cosub:
            #     loss = criterion(samples, outputs, targets)
            # else:
            #     outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
            #     loss = 0.25 * criterion(outputs[0], targets) 
            #     loss = loss + 0.25 * criterion(outputs[1], targets) 
            #     loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
            #     loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())
        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step(x, y, "val")
    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step(x, y, "test")
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
