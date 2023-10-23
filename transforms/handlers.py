
from collections import OrderedDict
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import faiss
from tqdm import tqdm
import gc
from qat.ops import QuantizedConv2dBatchNorm2dReLU
from qat.export.utils import fetch_module_by_name
from operations.im2col import im2col
from operations.maddness.maddness_conv2d import MaddnessConv2d
from operations.maddness.maddness_linear import MaddnessLinear
from operations.amm_conv2d import AMMConv2d
from operations.amm_linear import AMMLinear
from .utils import collect_input_tensors
from .ddp_collect_tensor import collect_input_tensors_v2

from concurrent.futures import ProcessPoolExecutor
from libKMCUDA import kmeans_cuda

import torchsnooper


def run_kmeans_on_device(data, clusters=16, seed=3, device=1):
    kmcuda_centroids, _ = kmeans_cuda(data, clusters, verbosity=0, seed=seed, device=device) 
    return kmcuda_centroids

def train_pq_4GPU(d, ncodebooks, num_centroids, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert d % ncodebooks == 0 # d means token's dimension
    assert len(data.shape) == 2 and data.shape[1] == d
    devices = [8, 4, 2, 1]  # 這些是 2 的冪次方值，分別對應於第 1, 2, 3, 4 張 GPU
    subvec_len = d // ncodebooks
    centroids = np.empty((ncodebooks, num_centroids, subvec_len), dtype=np.float32)
    N,D = data.shape
    data = np.reshape(data, (12, N, D//12))
    for j in range(3):  # 每張 GPU 計算三組數據
        with ProcessPoolExecutor(max_workers=4) as executor: # better memory efficient
            futures = []
            for i in range(4):  # 因為有四張 GPU
                idx = i * 3 + j  # 數據集的索引
                tmp_data = data[idx] 
                # print(tmp_data.shape) # (batch * patch, subvec_len)
                future = executor.submit(run_kmeans_on_device, tmp_data, device=devices[i])
                futures.append((idx, future))
            
            for idx, future in futures:
                centroids[idx] = future.result() # future.result() -> kmcuda_centroids
    return centroids, 0

def train_pq(d, ncodebooks, num_centroids, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # data: (n, d)
    assert d % ncodebooks == 0
    assert len(data.shape) == 2 and data.shape[1] == d
    
    subvec_len = d // ncodebooks
    centroids = np.empty((ncodebooks, num_centroids, subvec_len), dtype=np.float32)
    
    for i in tqdm(range(ncodebooks), desc="Processing codebooks"):
        train_data = data[:, i * subvec_len: (i + 1) * subvec_len]
        train_data = np.ascontiguousarray(train_data)
        kmcuda_centroids, _ = kmeans_cuda(train_data, num_centroids, verbosity=1, seed=3) # (16,32)
        centroids[i] = kmcuda_centroids

    return centroids, _


def _sync_centroids(modules, centroids):
    if dist.is_initialized():
        dist.barrier()
        for module in modules:
            if dist.get_rank() == 0:
                rank = torch.tensor(len(centroids[module].shape))
            else:
                rank = torch.tensor(0)
            dist.broadcast(rank, 0)
            if dist.get_rank() == 0:
                shape = torch.tensor(centroids[module].shape)
            else:
                shape = torch.empty(rank.item()).to(torch.long)
            dist.broadcast(shape, 0)
            if dist.get_rank() > 0:
                centroids[module] = torch.empty(tuple(shape.numpy())) \
                    .to(torch.float32)
            dist.broadcast(centroids[module], 0)


class TransferHandler:
    def __init__(self, model, target_model, calibrate_dataset) -> None:
        ...

    def transfer_conv2d(self, conv2d: nn.Conv2d, target: nn.Conv2d):
        target.weight.data.copy_(conv2d.weight.data)
        if target.bias is not None:
            target.bias.data.copy_(conv2d.bias.data)

    def transfer_bn2d(self, bn2d: nn.BatchNorm2d, target: nn.BatchNorm2d):
        target.load_state_dict(bn2d.state_dict())

    def transfer_linear(self, linear: nn.Linear, target: nn.Linear):
        target.weight.data.copy_(linear.weight.data)
        if target.bias is not None:
            target.bias.data.copy_(linear.bias.data)

    def transfer(self, module, target):
        target.load_state_dict(module.state_dict())

    def transfer_conv2d_bn2d(
        self, conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d,
        target: QuantizedConv2dBatchNorm2dReLU
    ):
        target.load_state_dict(conv2d.state_dict())


class AMMConv2dTransferHandler(TransferHandler):
    def __init__(self, model, target_model, calibrate_dataset):
        super().__init__(model, target_model, calibrate_dataset)
        self.centroids = self._learn_centroids(
            model, target_model, calibrate_dataset
        )

    @staticmethod
    def _learn_centroids(model, target_model, calibrate_dataset):
        module_names = OrderedDict([
            (module, name) for name, module in model.named_modules()
        ])

        centroids = {}

        def is_amm_conv2d(m):
            if not isinstance(m, nn.Conv2d):
                return False
            target_module = fetch_module_by_name(target_model, module_names[m])
            return isinstance(target_module, AMMConv2d)

        def ncodebooks(m):
            return getattr(fetch_module_by_name(target_model, module_names[m]), "ncodebooks")

        def num_centroids(m):
            return getattr(fetch_module_by_name(target_model, module_names[m]), "k")

        if not dist.is_initialized() or dist.get_rank() == 0:
            input_tensors = collect_input_tensors(
                model, calibrate_dataset,
                is_amm_conv2d
            )
            for conv, input_tensor in tqdm(list(input_tensors.items()), desc="computing centroids"):
                cols = im2col(
                    input_tensor.data.cpu().numpy(),
                    conv.kernel_size, conv.stride, conv.padding
                )
                # pq_centroids, _ = _train_pq(
                #     cols.shape[1],
                #     ncodebooks(conv),
                #     num_centroids(conv),
                #     np.transpose(cols, (0, 2, 1)).reshape((-1, cols.shape[1]))
                # )
                # centroids[conv] = torch.tensor(pq_centroids)

        modules = list(filter(is_amm_conv2d, module_names.keys()))
        _sync_centroids(modules, centroids)

        return centroids

    def transfer_conv2d_bn2d(
        self, conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d,
        target: AMMConv2d
    ):
        self.transfer(conv2d, target)

    def transfer(self, conv2d: nn.Conv2d, target: AMMConv2d):
        target.centroids.data.copy_(self.centroids[conv2d].data)
        target.weight.data.copy_(
            conv2d.weight.data.reshape(conv2d.out_channels, -1).permute(1, 0).reshape_as(target.weight))
        if target.bias is not None:
            target.bias.data.copy_(conv2d.bias.data)


class AMMLinearTransferHandler(TransferHandler):
    def __init__(self, model, target_model, calibrate_dataset):
        super().__init__(model, target_model, calibrate_dataset)
        self.centroids = self._learn_centroids(
            model, target_model, calibrate_dataset
        )
    
    @staticmethod
    # @torchsnooper.snoop()
    def _learn_centroids(model, target_model, calibrate_dataset):
        module_names = OrderedDict([
            (module, name) for name, module in model.named_modules()
        ])
        centroids = {}
        from pathlib import Path
        tensor_storage_path = Path('/work/u1887834/tensor_storage')  # Define a path to store the tensors
        tensor_storage_path.mkdir(parents=True, exist_ok=True)
        
        def is_amm_linear(m):# 他去檢查target_model中，對應名字的module是否為AMMLinear
            if not isinstance(m, nn.Linear):
                return False
            target_module = fetch_module_by_name(target_model, module_names[m]) 
            return isinstance(target_module, AMMLinear)

        def ncodebooks(m):
            return getattr(fetch_module_by_name(target_model, module_names[m]), "ncodebooks")

        def num_centroids(m):
            return getattr(fetch_module_by_name(target_model, module_names[m]), "k")
        collected_modules = list(filter(is_amm_linear, module_names.keys()))
        collected_modules.reverse()
        if not dist.is_initialized() or dist.get_rank() == 0:
            for linear in collected_modules:
                module_name = module_names[linear]  # Get the name of the current module
                print(f"module: {linear}") # Linear(in_features=384, out_features=1152, bias=True) blocks.11.attn.qkv
                print(f"Currently processing module: {module_name}")  # Print the name
                
                input_tensors = collect_input_tensors(
                    model, 
                    module_name, 
                    calibrate_dataset,
                    lambda m: m == linear,  # Collect for this specific module
                    batch_size=128
                )
                
                input_tensors = input_tensors.flatten(0, -2).cpu().numpy()
                np.save(tensor_storage_path / f"{module_name}.npy", input_tensors)
                
                # pq_centroids, _ = train_pq_4GPU( # train_pq
                #     linear.in_features,
                #     ncodebooks(linear),
                #     num_centroids(linear),
                #     input_tensors.flatten(0, -2).cpu().numpy()
                    
                # )
                #### Here build base LUT model
                # print(f"Current dataset number: {len(calibrate_dataset)}")
                
                # module_name = module_name.split(".")
                # Currently processing module: blocks.5.attn.q_linear
                # if "q_linear" in module_name or "k_linear" in module_name:
                #     print(f"{module_name[0]}.{module_name[1]}.{module_name[2]}.qkv_centroids.npy")
                #     # exit()
                #     pq_centroids = np.load(tensor_storage_path / f"{module_name[0]}.{module_name[1]}.{module_name[2]}.qkv_centroids.npy")
                    # hand load......                              blocks.0.attn.qkv_centroids

                # pq_centroids = np.load(tensor_storage_path / f"{len(calibrate_dataset)}_{module_name}_centroids.npy")
                # centroids[linear] = torch.tensor(pq_centroids) # also for one run ################
                ####
                ### Here save centroids
                # np.save(tensor_storage_path / f"{len(calibrate_dataset)}_{module_name}_centroids.npy", pq_centroids)
                
                del input_tensors
                # del pq_centroids
                gc.collect()
                ###
            exit()
            
        _sync_centroids(collected_modules, centroids)

        return centroids

    def transfer(self, linear: nn.Linear, target: AMMLinear):
        target.centroids.data.copy_(self.centroids[linear].data)
        target.weight.data.copy_(
            linear.weight.data.permute(1, 0).reshape_as(target.weight))
        if target.bias is not None:
            target.bias.data.copy_(linear.bias.data)


class MaddnessConv2dTransferHandler(TransferHandler):
    def __init__(self, model, target_model, calibrate_dataset) -> None:
        super().__init__(model, target_model, calibrate_dataset)

        self.dic = self._learn_dt(model, target_model, calibrate_dataset)

    @staticmethod
    def _learn_dt(model, target_model, calibrate_dataset):
        from blink_mm.ops.maddness.maddness import _learn_codebooks, _optimize_prototypes, _create_lookup_tables

        module_names = OrderedDict([
            (module, name) for name, module in model.named_modules()
        ])

        def is_maddness_conv2d(m):
            if not isinstance(m, nn.Conv2d):
                return False
            target_module = fetch_module_by_name(target_model, module_names[m])
            return isinstance(target_module, MaddnessConv2d)

        def ncodebooks(m):
            return getattr(fetch_module_by_name(target_model, module_names[m]), "ncodebooks")

        dic = {
            "split_idxs": {},
            "split_vals": {},
            "lookup_tables": {},
        }

        if not dist.is_initialized() or dist.get_rank() == 0:
            input_tensors = collect_input_tensors(
                model, calibrate_dataset,
                is_maddness_conv2d
            )

            for conv, input_tensor in tqdm(list(input_tensors.items()), desc="computing decision trees"):
                x = im2col(
                    input_tensor.data.cpu().numpy(),
                    conv.kernel_size, conv.stride, conv.padding
                )
                x = np.transpose(x, (0, 2, 1)).reshape((-1, x.shape[1]))
                x = x + np.random.rand(*x.shape) * 1e-6

                codebooks = _learn_codebooks(ncodebooks(conv), x)
                codebooks = _optimize_prototypes(x, codebooks)
                lookup_tables = _create_lookup_tables(
                    conv.weight.reshape(conv.out_channels, -1)
                        .transpose(1, 0).detach().cpu().numpy(),
                    codebooks
                )
                split_idxs = torch.tensor([
                    codebook.split_idxs
                    for codebook in codebooks
                ])
                split_vals = torch.empty(
                    ncodebooks(conv), 4, 8, dtype=torch.float32)
                for i in range(ncodebooks(conv)):
                    for j in range(4):
                        split_vals[i][j][:1 << j] = \
                            torch.tensor(codebooks[i].split_vals[j])

                dic["split_idxs"][conv] = split_idxs.detach().cpu()
                dic["split_vals"][conv] = split_vals.detach().cpu()
                dic["lookup_tables"][conv] = \
                    torch.tensor(np.array(lookup_tables))

        modules = list(filter(is_maddness_conv2d, module_names.keys()))
        _sync_centroids(modules, dic["split_idxs"])
        _sync_centroids(modules, dic["split_vals"])
        _sync_centroids(modules, dic["lookup_tables"])

        return dic

    def transfer_conv2d_bn2d(
        self, conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d,
        target: MaddnessConv2d
    ):
        self.transfer(conv2d, target)

    def transfer(self, conv2d: nn.Conv2d, target: MaddnessConv2d):
        target.split_idxs.data.copy_(self.dic["split_idxs"][conv2d])
        target.split_vals.data.copy_(self.dic["split_vals"][conv2d])
        target.lookup_tables.data.copy_(self.dic["lookup_tables"][conv2d])
        if target.bias is not None:
            target.bias.data.copy_(conv2d.bias.data)


class MaddnessLinearTransferHandler(TransferHandler):
    def __init__(self, model, target_model, calibrate_dataset) -> None:
        super().__init__(model, target_model, calibrate_dataset)

        self.dic = self._learn_dt(model, target_model, calibrate_dataset)

    @staticmethod
    def _learn_dt(model, target_model, calibrate_dataset):
        from blink_mm.ops.maddness.maddness import _learn_codebooks, _optimize_prototypes, _create_lookup_tables

        module_names = OrderedDict([
            (module, name) for name, module in model.named_modules()
        ])

        def is_maddness_linear(m):
            if not isinstance(m, nn.Linear):
                return False
            target_module = fetch_module_by_name(target_model, module_names[m])
            return isinstance(target_module, MaddnessLinear)

        def ncodebooks(m):
            return getattr(fetch_module_by_name(target_model, module_names[m]), "ncodebooks")

        dic = {
            "split_idxs": {},
            "split_vals": {},
            "lookup_tables": {},
        }

        if not dist.is_initialized() or dist.get_rank() == 0:
            input_tensors = collect_input_tensors(
                model, calibrate_dataset,
                is_maddness_linear
            )

            for linear, input_tensor in tqdm(list(input_tensors.items()), desc="computing decision trees"):
                x = input_tensor.detach().cpu().numpy()
                x = x + np.random.rand(*x.shape) * 1e-6

                codebooks = _learn_codebooks(ncodebooks(linear), x)
                codebooks = _optimize_prototypes(x, codebooks)
                lookup_tables = _create_lookup_tables(
                    linear.weight.transpose(1, 0).detach().cpu().numpy(),
                    codebooks
                )
                split_idxs = torch.tensor([
                    codebook.split_idxs
                    for codebook in codebooks
                ])
                split_vals = torch.empty(
                    ncodebooks(linear), 4, 8, dtype=torch.float32)
                for i in range(ncodebooks(linear)):
                    for j in range(4):
                        split_vals[i][j][:1 << j] = \
                            torch.tensor(codebooks[i].split_vals[j])

                dic["split_idxs"][linear] = split_idxs.detach().cpu()
                dic["split_vals"][linear] = split_vals.detach().cpu()
                dic["lookup_tables"][linear] = \
                    torch.tensor(np.array(lookup_tables))

        modules = list(filter(is_maddness_linear, module_names.keys()))
        _sync_centroids(modules, dic["split_idxs"])
        _sync_centroids(modules, dic["split_vals"])
        _sync_centroids(modules, dic["lookup_tables"])

        return dic

    def transfer(self, linear: nn.Linear, target: MaddnessLinear):
        target.split_idxs.data.copy_(self.dic["split_idxs"][linear])
        target.split_vals.data.copy_(self.dic["split_vals"][linear])
        target.lookup_tables.data.copy_(self.dic["lookup_tables"][linear])
        if target.bias is not None:
            target.bias.data.copy_(linear.bias.data)
