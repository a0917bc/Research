from networks.LUTDeiT import LUT_DeiT, LUT_Distilled_DeiT, Attention2, create_target
from timm import create_model
from operations.amm_linear import LUT_Linear
from qat.export.utils import replace_module_by_name, fetch_module_by_name
from torchinfo import summary
import time
import torch
import torch_tensorrt
import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def benchmark(model, input_shape=(1024, 3, 512, 512), dtype='fp32', nwarmup=50, nruns=1000, cuda=False):
    input_data = torch.randn(input_shape)
    if cuda:
        input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            pred_loc  = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print('Average throughput: %.2f images/second'%(input_shape[0]/np.mean(timings)))


def build(model_compressed):
    torch._dynamo.reset()
    sample_inputs_half = [
        torch.rand((1, 3, 224, 224)).cuda(),
        # torch.rand((1, 3, 224, 224)).half().cuda(),
    ]
    backend_kwargs = {
        "enabled_precisions": {torch.half, torch.float},
        "debug": True,
        "min_block_size": 2,
        "torch_executed_ops": {"torch.ops.aten.sub.Tensor"},
        "optimization_level": 4,
        "use_python_runtime": False,
    }
    optimized_model = torch.compile(
        model_compressed.cuda(), 
        options=backend_kwargs,
        backend="torch_tensorrt")
    # optimized_model(*sample_inputs_half)
    
    torch._dynamo.reset()
    benchmark(optimized_model, input_shape=(1, 3, 224, 224), nruns=100, cuda=True)

if __name__ == '__main__':
    model = create_model(model_name="deit3_small_patch16_224.fb_in1k", pretrained=False)
    model.eval()
    subvec_len = 32
    ncodebooks = {
        "attn.qkv": 384 // subvec_len,
        # "attn.q_linear": 384 // subvec_len, 
        # "attn.k_linear": 384 // subvec_len, 
        "mlp.fc1": 384 // subvec_len,
        # "mlp.fc2": 1536 // subvec_len
    }
    for i in range(0, 12): 
        for name in ncodebooks:
            layer = model.blocks[i]
            module = fetch_module_by_name(layer, name)
            amm_linear = LUT_Linear(
            # amm_linear = PQLinear(
                ncodebooks[name],
                module.in_features,
                module.out_features,
                module.bias is not None,
                k=16
            )
            # print(amm_linear.weight.data.shape)
            # print(module.weight.data.shape)
            # weight = rearrange(module.weight.data, 'o i -> i o')
            # weight = rearrange(weight, '(c v) o -> c v o', c=ncodebooks[name], v=subvec_len)
            # amm_linear.weight.data.copy_(weight.data)
            # amm_linear.bias.data.copy_(module.bias.data)
            replace_module_by_name(layer, name, amm_linear)
    model_compressed = model
    model_compressed.eval()
    model_compressed = model_compressed
    # build(model_compressed)
    build(model)
