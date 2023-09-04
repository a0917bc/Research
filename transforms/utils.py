from typing import Callable, OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



def collect_input_tensors(
    model: nn.Module, 
    module_name: str,
    dataset: Dataset,
    func: Callable[[nn.Module], bool],
    batch_size=8,
    is_input_tensor=True,
):
    if len(dataset) == 0:
        return {}

    input_tensors = []
    def hook(module, input_tensor, output_tensor):
        if is_input_tensor:
            assert len(input_tensor) == 1
            tensor = input_tensor[0].cpu()
        else:
            tensor = output_tensor.cpu()
        input_tensors.append(tensor)
    handles = []
    
    for name, module in model.named_modules(): 
        if func(module) and name == module_name: # double check module_name
            handles.append(module.register_forward_hook(hook))
            
    if len(handles) == 0:
        return input_tensors
    
    train_data_loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=16)
    model.eval()
    model = model.to("cuda:0")
    model = nn.DataParallel(model)
    for data_batch in tqdm(train_data_loader, desc=f"collecting {'input' if is_input_tensor else 'output'} tensors"):
        x = data_batch[0].to("cuda:0")
        with torch.no_grad():
            model(x)
    for handle in handles:
        handle.remove()
    
    input_tensors = torch.concat(input_tensors)
    
    return input_tensors

