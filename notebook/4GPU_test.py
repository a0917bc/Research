from concurrent.futures import ProcessPoolExecutor
import numpy as np  
from libKMCUDA import kmeans_cuda
# from tqdm import tqdm

def run_kmeans_on_device(data, clusters=16, seed=3, device=1):
    centroids, _ = kmeans_cuda(data, clusters, verbosity=1, seed=seed, device=device) 
    # kmeans_cuda(arr, 16, verbosity=1, seed=3, device=3)
    # return np.array(centroids, assignments)
    return centroids

devices = [1, 2, 4, 8]  # 這些是 2 的冪次方值，分別對應於第 1, 2, 3, 4 張 GPU
centroids = np.empty((12, 16, 32), dtype=np.float32) # centroids = np.empty((ncodebooks, num_centroids, subvec_len), dtype=np.float32)
data_set = np.load("/work/u1887834/tensor_storage/blocks.0.attn.qkv.npy")
N,D = data_set.shape
data_set = np.reshape(data_set, (12, N, D//12))
# with ProcessPoolExecutor(max_workers=12) as executor:
#     futures = []
#     for i in range(4):  # 因為有四張 GPU
        
#             idx = i * 3 + j  # 數據集的索引
#             tmp_data = data_set[idx] 
#             # print(tmp_data.shape) # (batch * patch, subvec_len)
#             future = executor.submit(run_kmeans_on_device, tmp_data, device=devices[i])
#             futures.append((idx, future))
    
#     for idx, future in futures:
#         centroids[idx] = future.result() # future.result() -> kmcuda_centroids
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
np.save("./centroids.npy", centroids)
print(centroids.shape)