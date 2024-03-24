# Differentiable Lookup-Based Matrix Multiplication for Compressing Transformer Network

## Overview
<img width="787" alt="image" src="https://github.com/a0917bc/Research/assets/22569133/970aa7c0-1bee-442b-83cd-0158cd2f14ad">
https://github.com/huawei-noah/AdderNet

In recent years, there has been research on replacing multiplication operations. A graph has been gathered to compare the energy cost of multiplication and addition.

As a result, approaches like AdderNet replace multiplication in convolutions with addition, while ShiftCNN represents weights as powers of two, allowing multiplication to be replaced with bit-shift operations.

Furthermore, in recent research, some have replaced the Multiply-Accumulate (MAC) operations in matrix multiplication with table lookup and addition.

## Usage
This is a research-oriented project, without a complete usage guide yet, but I can explain what these files intend to do.

demo.py: Compiles the model using TVM to find the optimal parameters (block size) for hardware and runs inference.
prototype_learning.py: Initializes prototypes using KMCUDA.
tensorrt_op.py: Attempts to compile the model using torch_tensorrt and runs it on the GPU after compilation.
train.py and other files containing "train": Retrains the model after replacing the lookup-based matrix multiplication.
OpCounter.ipynb: Measures the GFLOPs and model size after replacing the lookup-based matrix multiplication using thop.

Short Flow:
prototype_learning.py -> train.py -> demo.py

## Contributions
• Developed a comprehensive training pipeline, particularly effective in handling ImageNet.
• Achieved a significant accuracy improvement of up to 10% by surpassing LUT-NN at MobiCom 2023.

## License
Specify the license for your research code or data, so others know how they can use your project. Common options include MIT License, GNU General Public License (GNU GPL), etc.

## Contact
Feel free to contact me(a0917bc(at)gmail(dot)com) if you have any questions. 

## References
https://github.com/dblalock/bolt/tree/master
https://github.com/lutnn/blink-mm
https://github.com/src-d/kmcuda
