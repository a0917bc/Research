import os
from argparse import ArgumentParser

import timm
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader
from lightning.pytorch import seed_everything

from networks.LUTDeiT import create_target, Attention2
from transforms.transfer import transfer, AMM_PASS


def load_data(batchSize, numWorkers):
    batch_size = batchSize
    traindir = os.path.join("/work/u1887834/imagenet/", 'train')
    valdir = os.path.join("/work/u1887834/imagenet/", 'val')
    # traindir = os.path.join("/dev/shm/imagenet/", 'train')
    # valdir = os.path.join("/dev/shm/imagenet/", 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
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
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=numWorkers, pin_memory=True, sampler=None)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=numWorkers, pin_memory=True, sampler=None)
    return train_loader

if __name__ == "__main__":
    parser = ArgumentParser()
    # Trainer arguments
    parser.add_argument("--devices", type=int, default=4)
    # Hyperparameters for the model
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batchSize", type=int, default=192)
    parser.add_argument("--numWorkers", type=int, default=8)
    parser.add_argument("--num", type=int, default=1024, 
                    help="Specify the number of dataset to initialize base LUT model. "
                    )
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--layer", type=int, default=0, 
                    help="Specify the number of layer to be product-quantized. "
                    )
    parser.add_argument("--stop", type=int, default=12, 
                    help="Specify stopping layer. "
                    )
    args = parser.parse_args()
    seed_everything(7)

    train_loader = load_data(args.batchSize, args.numWorkers)
    # model_name = 'deit3_small_patch16_224.fb_in22k_ft_in1k' # 很久以前蒐集的Centroids，我記得是用 deit3_small_patch16_224.fb_in1k
    model_name = "resmlp_12_224.fb_in1k"
    float_model = timm.create_model(model_name, pretrained=True)
    # float_model = torch.load(f"/home/u1887834/Research/notebook/{model_name}.pth") 
    target_model = create_target(args.layer, args.stop, model_name)
    pass_type = AMM_PASS
    transfer(
        float_model, target_model,
        Subset(train_loader.dataset, 
               range(min(args.num, len(train_loader.dataset))
                     )
        ), 
        pass_type
    )
    from pathlib import Path
         
    save_path = Path('/home/u1887834/Research/base_model'+"_qk")
    save_path.mkdir(parents=True, exist_ok=True)
    # torch.save(target_model.state_dict(), save_path / f"{args.num}_base_{args.layer}_{args.stop}.pt")
    torch.save(target_model, save_path / f"{args.num}_base_{args.layer}_{args.stop}.pt")
