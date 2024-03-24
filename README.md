# Differentiable Lookup-Based Matrix Multiplication for Compressing Transformer Network

## Overview

<img width="696" alt="image" src="https://github.com/a0917bc/Research/assets/22569133/45ddd6a9-f319-438e-9e2b-97eb21fbbe8e">

In recent years, there has been research on replacing multiplication operations. The below graph has been gathered to compare the energy cost of multiplication and addition.
<img src="https://github.com/a0917bc/Research/assets/22569133/970aa7c0-1bee-442b-83cd-0158cd2f14ad" alt="energy_cost_comparison_graph" width="200">

As a result, approaches like [AdderNet](https://arxiv.org/abs/1706.02393) replace multiplication in convolutions with addition, while [ShiftCNN](https://github.com/huawei-noah/AdderNet) represents weights as powers of two, allowing multiplication to be replaced with bit-shift operations.

Furthermore, in recent research, some have replaced the Multiply-Accumulate (MAC) operations in matrix multiplication with table lookup and addition ([source](https://github.com/dblalock/bolt/tree/master)).

This research is a little complicated. If you find it interesting, further details are available [here](https://drive.google.com/file/d/1MWdCc87fbf3tu652l5MjYaCSlNwwaN5k/view?usp=sharing).

## Usage

This is a research-oriented project, without a complete usage guide yet, but I can explain what these files intend to do.

- demo.py: Compiles the model using TVM to find the optimal parameters (block size) for hardware and runs inference.
- prototype_learning.py: Initializes prototypes using KMCUDA.
- tensorrt_op.py: Attempts to compile the model using torch_tensorrt and runs it on the GPU after compilation.
- train.py and other files containing "train": Retrains the model after replacing the lookup-based matrix multiplication.
- OpCounter.ipynb: Measures the GFLOPs and model size after replacing the lookup-based matrix multiplication using thop.

Short Flow:
prototype_learning.py -> train.py -> demo.py

## Contributions

- Developed a comprehensive training pipeline, particularly effective in handling ImageNet.
- Achieved a significant accuracy improvement of up to 10% by surpassing LUT-NN at MobiCom 2023.
  
## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the [LICENSE](LICENSE) file for details.

## Contact

Feel free to contact me(a0917bc(at)gmail(dot)com) if you have any questions.

## References

- [https://github.com/dblalock/bolt/tree/master](https://github.com/dblalock/bolt/tree/master)
- [https://github.com/lutnn/blink-mm](https://github.com/lutnn/blink-mm)
- [https://github.com/src-d/kmcuda](https://github.com/src-d/kmcuda)
