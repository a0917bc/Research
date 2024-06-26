# Standard library imports
from argparse import ArgumentParser
import os

# Third-party imports
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import lightning as L
from lightning.pytorch.callbacks import StochasticWeightAveraging, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from timm import create_model
from timm.data import Mixup, resolve_model_data_config, create_transform
# Custom imports
from networks.LUTDeiT import LUT_DeiT, LUT_Distilled_DeiT, Attention2, create_target
from ema import EMA

def get_args_parser():
    parser = ArgumentParser()
    # Trainer arguments
    parser.add_argument("--devices", type=int, default=2)
    
    # My settings
    parser.add_argument('--model_name', type=str, default='deit3_small_patch16_224.fb_in22k_ft_in1k')
    
    parser.add_argument("--numWorkers", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--layer", type=int, default=9, 
                    help="Specify the number of layer to be product-quantized. "
                    )
    parser.add_argument("--stop", type=int, default=12, 
                    help="Specify stopping layer. "
                    )
    parser.add_argument("--num", type=int, default=120000, 
                    help="Specify the number of dataset to initialize base LUT model. "
                    )
    parser.add_argument('--resume', type=str, default='/home/yllab/JiaXing/Research/home/u1887834/Research/BeyondLUTNN/yefgcf9q/checkpoints/epoch=49-step=41750.ckpt')
    parser.add_argument('--ckpt', type=str, default=None)
    
    # Knowledge distillation
    parser.add_argument('--kd', type=str, default="soft", 
                        help='kd type (default: soft)') 
    parser.add_argument('--alpha', default=0.8, type=float) # 0.8*teacher_loss
    parser.add_argument('--tau', type=float, default=1, 
                        help='kd type (default: hard)') 
    
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--batchSize", type=int, default=16) # change to 128
    
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='weight decay (default: 0.05)')
    
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    return parser.parse_args()

def load_data(batchSize, 
              num_workers,
              float_model
              ):
    batch_size = batchSize
    # traindir = os.path.join("/work/u1887834/imagenet/", 'train')
    valdir = os.path.join("/dataset/imagenet/", 'val')

    data_config = resolve_model_data_config(float_model)
    val_transform = create_transform(**data_config, is_training=False)
    train_transform = create_transform(**data_config, is_training=True)
    train_dataset = datasets.ImageFolder(
        valdir,
        train_transform
        )

    val_dataset = datasets.ImageFolder(
        valdir,
        val_transform
        )
   
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, sampler=None)
    # train_loader = None
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, sampler=None)
    return val_loader, val_loader

if __name__ == "__main__":
    L.seed_everything(7)
    args = get_args_parser()
   
    args.numWorkers = args.devices * 4
    float_model = create_model(args.model_name, pretrained=True)
    train_loader, val_loader = load_data(
        args.batchSize, 
        args.numWorkers,
        float_model
        )
    
    pl_model = LUT_DeiT().load_from_checkpoint(args.resume) 
    # print(pl_model)
    
    trainer = L.Trainer(
        max_epochs=args.epoch,
        # precision='16-mixed',
        # devices=args.devices,
        accelerator="cpu",
        accumulate_grad_batches=4,
        log_every_n_steps=10,
        # profiler="simple", # Once the .fit() function has completed, you’ll see an output.
        callbacks = [
            EMA(decay=0.999),
            # StochasticWeightAveraging(swa_lrs=1e-2),
            # EarlyStopping(monitor="val_acc", mode="max", patience=5), 
            ModelCheckpoint(monitor='val_loss', save_top_k=1),
            LearningRateMonitor(logging_interval="epoch")
            ],
        strategy='ddp_find_unused_parameters_true',
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    trainer.validate(pl_model, val_loader)
    