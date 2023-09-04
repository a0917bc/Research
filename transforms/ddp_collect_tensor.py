from typing import Callable, OrderedDict


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import os

# Initialize the distributed environment


def collect_input_tensors_v2(
    model: nn.Module, dataset: Dataset,
    func: Callable[[nn.Module], bool],
    batch_size=8,
    is_input_tensor=True,
):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    if len(dataset) == 0:
        return {}
    sampler = DistributedSampler(dataset)
    train_data_loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=32, sampler=sampler)
    
    model = model.to(device=f'cuda:{local_rank}')
    model.eval()
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Hook to collect input tensors
    input_tensors = []

    def collect_input(module, input_tensor, output_tensor):
        input_tensors.append(input_tensor[0].cpu())

    handles = []
    
    for _, module in model.named_modules():
        if func(module):
            handle = module.register_forward_hook(collect_input)
            handles.append(handle)
            
    if len(handles) == 0:
        return input_tensors

    
    for data_batch in tqdm(train_data_loader, desc=f"collecting {'input' if is_input_tensor else 'output'} tensors"):
        x = data_batch[0].cuda(local_rank)
        model(x)
    for handle in handles:
        handle.remove()
    print(input_tensors)
    exit()
    return input_tensors
