# Fast-GPR-FWI
This repository gives the codes for "Fast ground penetrating radar dual parameters full waveform inversion method accelerated by hybrid compilation of CUDA kernel function and PyTorch". This work has been submitted to Computers and Geosciences.

# Overview
This study proposes a high-performance dual-parameter full waveform inversion framework (FWI) for ground-penetrating radar (GPR), accelerated through the hybrid compilation of CUDA kernel functions and PyTorch. The method leverages the computational efficiency of GPU programming while preserving the flexibility and usability of Python-based deep learning frameworks. By integrating customized CUDA kernels into PyTorch’s automatic differentiation mechanism.

# Usage Instructions
1. CUDA Must Be installed on the Runtime Device
Ensure that your machine has the NVIDlA CUDA drivers and toolkit properly installed.You can verify the installation by running: nvcc -V
2. Create or Activate a Conda Environment with CUDA-Enabled PyTorch
Create or activate a Conda virtual environment, and make sure it includes a version of PyTorch with CUDA support. You can find suitable CuDA-enabled PyTorch versions at: https://download.pytorch.org/whl/torch/。 Example: to install PyTorch for CUDA 11.7:
pip install torch==l.13.1+cu117 torchvision==0.14.1+cu117 -fhttps://download.pytorch.org/whl/torch stable.html
3. Navigate to the DeepGPR Project Directory and Compile
Change to the DeepGPR/deepgpr directory and compile using the provided Makefile:
cd DeepGPR/deepgpr
make
