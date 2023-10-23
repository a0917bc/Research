from concurrent.futures import ProcessPoolExecutor
import numpy as np  
from libKMCUDA import kmeans_cuda
from pathlib import Path
import os
tensor_storage_path = Path('/work/u1887834/tensor_storage')  # Define a path to store the tensors
def run_kmeans_on_device(data, clusters=16, seed=3, device=1):
    centroids, _ = kmeans_cuda(data, clusters, verbosity=1, seed=seed, device=device) 
    # kmeans_cuda(arr, 16, verbosity=1, seed=3, device=3)
    # return np.array(centroids, assignments)
    return centroids

devices = [1, 2, 4, 8]  # 這些是 2 的冪次方值，分別對應於第 1, 2, 3, 4 張 GPU
'''ncodebooks = {
        # "attn.qkv": 384 // subvec_len,
        # "attn.q_linear": 384 // subvec_len, 
        # "attn.k_linear": 384 // subvec_len, 
        # "mlp.fc1": 384 // subvec_len
        # "mlp.fc2": 1536 // subvec_len
        
        "linear_tokens": 196 // 14, 
        "mlp_channels.fc1": 384 // subvec_len,
        "mlp_channels.fc2":1536 // subvec_len
    }'''
module_name_list = [
    "blocks.11.mlp_channels.fc1.npy",
    "blocks.11.linear_tokens.npy",
    "blocks.10.mlp_channels.fc2.npy",
    "blocks.10.mlp_channels.fc1.npy",
    "blocks.10.linear_tokens.npy",
    "blocks.9.mlp_channels.fc2.npy",
    "blocks.9.mlp_channels.fc1.npy",
    "blocks.9.linear_tokens.npy",
    "blocks.8.mlp_channels.fc2.npy",
    "blocks.8.mlp_channels.fc1.npy",
    "blocks.8.linear_tokens.npy",
    "blocks.7.mlp_channels.fc2.npy",
    "blocks.7.mlp_channels.fc1.npy",
    "blocks.7.linear_tokens.npy"
]
    
# path = "/work/u1887834/tensor_storage/blocks.11.mlp_channels.fc2.npy"
# module_name = os.path.splitext(os.path.basename(path))[0]
# print(module_name)

for p in module_name_list:
    
    data_set = np.load(tensor_storage_path / p)
    module_name = os.path.splitext(os.path.basename(tensor_storage_path / p))[0]
    
    N, D = data_set.shape
    if "mlp_channels" in p:
        print(module_name)
        data_set = np.reshape(data_set, (12, N, D//12))
        centroids = np.empty((12, 16, D//12), dtype=np.float32) # centroids = np.empty((ncodebooks, num_centroids, subvec_len), dtype=np.float32)
        for j in range(3):  # 每張 GPU 計算三組數據
            with ProcessPoolExecutor(max_workers=4) as executor: # better memory efficient
                futures = []
                for i in range(4):  # 因為有四張 GPU
                    idx = i * 3 + j  # 數據集的索引
                    tmp_data = data_set[idx] 
                    # print(tmp_data.shape) # (batch * patch, subvec_len)
                    future = executor.submit(run_kmeans_on_device, tmp_data, device=devices[i])
                    futures.append((idx, future))
                
                for idx, future in futures:
                    centroids[idx] = future.result() # future.result() -> kmcuda_centroids
    else:
        print(module_name)
        data_set = np.reshape(data_set, (14, N, D//14))
        centroids = np.empty((14, 16, D//14), dtype=np.float32) # centroids = np.empty((ncodebooks, num_centroids, subvec_len), dtype=np.float32)
        for i in range(14):
            tmp_data = data_set[i]
            centroids, _ = kmeans_cuda(tmp_data, 16, verbosity=1) 
    np.save(tensor_storage_path / f"{module_name}_centroids.npy", centroids)          
# np.save(tensor_storage_path / f"{module_name}_centroids.npy", centroids)
# print(centroids.shape)
#%%
# import os
# import numpy as np  
# from pathlib import Path
# tensor_storage_path = Path('/work/u1887834/tensor_storage')  # Define a path to store the tensors

# path = "/work/u1887834/tensor_storage/blocks.11.mlp_channels.fc2.npy"
# module_name = os.path.splitext(os.path.basename(path))[0]
# centroids = np.load(tensor_storage_path / f"{module_name}_centroids.npy")
