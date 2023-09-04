import torch
# from torchvision.models import resnet18 as resnet
from time import perf_counter
import torchsnooper
from networks.LUTDeiT import * # create_deit
import lightning as L
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as M
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
def benchmark_inference(model, inputs, trials=10):
    # warmup
    model(inputs)

    t0 = perf_counter()
    for i in range(trials):
        model(inputs)
    t1 = perf_counter()
    return (t1 - t0) / trials


def benchmark_trainer(model, **kwargs):
    trainer = L.Trainer(
        max_epochs=1,
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    t0 = perf_counter()
    trainer.fit(model, **kwargs)
    t1 = perf_counter()
    return t1 - t0



import timm

def load_data():
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )

    batch_size = 64

    # train_set = torchvision.datasets.CIFAR10(
    #     root="~/.cache/data", train=True, download=True, transform=transform
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_set, batch_size=batch_size, shuffle=True, num_workers=4
    # )
    traindir = os.path.join("/work/u1887834/imagenet/", 'train')
    valdir = os.path.join("/work/u1887834/imagenet/", 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        Subset(train_dataset, range(4096)), batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, sampler=None)
    return train_loader

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = create_deit()
        self.criterion = nn.CrossEntropyLoss()
    # @torchsnooper.snoop()
    def forward(self, x):
        return self.model(x)

    def common_step(self, x, y, stage):
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log(f"{stage}/loss", loss, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step(x, y, "train")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)


def main():
    
    train_loader = load_data()

    model = LitModel()

    unoptimized_t = benchmark_trainer(
        model,
        train_dataloaders=train_loader,
    )
    print(f"time taken to train unoptimized model: {unoptimized_t}")
    unoptimized_t = benchmark_trainer(
        model,
        train_dataloaders=train_loader,
    )
    print(f"time taken to train unoptimized model: {unoptimized_t}")

    compiled_model = torch.compile(LitModel())
    optimized_t = benchmark_trainer(
        compiled_model,
        train_dataloaders=train_loader,
    )
    print(f"time taken to train optimized model: {optimized_t}")
    optimized_t = benchmark_trainer(
        compiled_model,
        train_dataloaders=train_loader,
    )
    print(f"time taken to train optimized model: {optimized_t}")
if __name__ == "__main__":
    main()