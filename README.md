# Fast-GPR-FWI
This repository gives the codes for "Fast ground penetrating radar dual-parameter full waveform inversion method accelerated by hybrid compilation of CUDA kernel function and PyTorch". This work has been submitted to Computers and Geosciences.

# Overview
This study proposes a high-performance dual-parameter full waveform inversion framework (FWI) for ground-penetrating radar (GPR), accelerated through the hybrid compilation of CUDA kernel functions and PyTorch. The method leverages the computational efficiency of GPU programming while preserving the flexibility and usability of Python-based deep learning frameworks. By integrating customized CUDA kernels into PyTorch’s automatic differentiation mechanism, the developed framework enables accurate and efficient inversion of both dielectric permittivity and electrical conductivity.
![gpr_figre](https://github.com/user-attachments/assets/28e18a25-4e25-4ecb-9698-f9fdbae2fa02)

Cross-hole dual-parameter GPR FWI. (a) True relative permittivity model; (b) initial relative permittivity model; (c) inverted relative permittivity model; (d) true conductivity model; (e) initial conductivity model; (f) inverted conductivity model.

## Usage Instructions

1. **CUDA Must Be Installed on the Runtime Device**

   Ensure that your machine has the NVIDIA CUDA drivers and toolkit properly installed.  
   You can verify the installation by running:

   ```bash
   nvcc -V
2. **Create or Activate a Conda Environment with CUDA-Enabled PyTorch**

   Create or activate a Conda virtual environment, and make sure it includes a version of PyTorch with CUDA support.

   - You can find suitable CUDA-enabled PyTorch versions at:：
     [https://download.pytorch.org/whl/torch/](https://download.pytorch.org/whl/torch/)
   - Example: to install PyTorch for CUDA 11.7:
     ```bash
     pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
     ```

3. **Navigate to the FastGPRFWI Project Directory and Compile**

   Change to the `FastGPRFWI/src` directory and compile using the provided Makefile: 

   ```bash
   cd FastGPRFWI/src
   make
   ```
   This will compile the necessary .cu source files into .so shared objects.
4. **Run the Test File `example.ipynb`**
   
   Open and execute the dm.ipynb notebook using Jupyter to verify everything is working correctly. 

