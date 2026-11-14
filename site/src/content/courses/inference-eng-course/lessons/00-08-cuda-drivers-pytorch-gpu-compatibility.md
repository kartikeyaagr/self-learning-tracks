---
id: "00.08"
course: "inference-eng-course"
module: "00"
slug: "cuda-drivers-pytorch-gpu-compatibility"
title: "CUDA, Drivers, PyTorch & GPU Compatibility"
keyIdeas: ["Stack: GPU hardware, kernel driver, CUDA runtime/toolkit, libraries (cuBLAS, cuDNN, NCCL), framework (PyTorch), engine (vLLM).", "The driver must be new enough for the CUDA runtime your framework was built with.", "nvidia-smi's \"CUDA Version\" is the max the driver supports, not what is installed.", "Each GPU generation has a compute capability (e.g. 8.0 A100, 9.0 H100). Some kernels and dtypes (FP8) need a minimum.", "Pin versions. Containers make the whole stack reproducible."]
question: "nvidia-smi reports \"CUDA Version: 12.4\" but your PyTorch wheel was built for CUDA 12.8 and fails to see the GPU. Explain why and how to fix it."
answer: "nvidia-smi shows the highest CUDA version the installed driver supports, not an installed toolkit. A PyTorch build for CUDA 12.8 needs a driver new enough for 12.8. Either upgrade the NVIDIA driver or install a PyTorch build for a CUDA version the driver supports, such as 12.4 or earlier."
sources: [{"title": "CUDA Compatibility guide", "url": "https://docs.nvidia.com/deploy/cuda-compatibility/index.html", "note": "Driver vs runtime version rules."}, {"title": "PyTorch: Get Started locally", "url": "https://pytorch.org/get-started/locally/", "note": "Picking the right wheel for your CUDA version."}]
tags: ["cuda", "setup"]
---

## The compatibility chain

Running anything on an NVIDIA GPU depends on a chain of versions that must agree:

~~~dg-layers From your code down to the hardware.
Your code and inference engine | vLLM, SGLang, Transformers
*PyTorch build | compiled against a specific CUDA version, e.g. cu124
CUDA runtime and libraries | cuBLAS, cuDNN, NCCL, shipped inside the PyTorch wheel
NVIDIA driver | installed on the host, sets the maximum CUDA version supported
GPU hardware | compute capability, e.g. 8.0 for A100, 9.0 for H100
~~~

## The rules that matter

1. **The driver must be new enough for the CUDA runtime.** A driver supports its own CUDA version and all older ones. `nvidia-smi` shows "CUDA Version: 12.4" meaning the driver supports up to 12.4. A PyTorch built for CUDA 12.6 will not run on it (minor-version compatibility helps in some cases, but do not rely on it).
2. **PyTorch wheels bundle their own CUDA runtime.** You usually do not need the CUDA toolkit installed system-wide to run PyTorch. You do need it (nvcc) to compile custom kernels.
3. **The GPU's compute capability must be supported by the build.** FP8 tensor cores need compute capability 8.9 (Ada) or 9.0 (Hopper). An A100 (8.0) cannot run native FP8 GEMMs; engines fall back or refuse.
4. **Engines pin PyTorch versions.** vLLM releases are built against a specific PyTorch. Installing a different PyTorch afterwards can break its compiled kernels.

## Checking what you have

```
nvidia-smi                       # driver version, max CUDA, GPUs
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

<div class="ex">
<div class="lab">Worked example</div>

**decoding a typical failure**

Error: "CUDA driver version is insufficient for CUDA runtime version".

nvidia-smi shows driver 535 (supports CUDA 12.2). pip installed torch 2.5 with cu124.

Fixes: install a torch wheel built for cu121, or upgrade the driver to 550 or newer. On a cloud VM, picking a newer base image is usually faster than upgrading drivers by hand.

</div>

## Compute capability and features

| GPU | Compute capability | Notable inference features |
| --- | --- | --- |
| T4 | 7.5 | FP16 tensor cores, INT8 |
| A100 | 8.0 | BF16, TF32, sparsity |
| L4, L40S, RTX 4090 | 8.9 | FP8 tensor cores |
| H100, H200 | 9.0 | FP8, TMA, larger shared memory, FlashAttention-3 |
| B200 | 10.0 | FP4, FP6 |

Kernels are often written for specific architectures. FlashAttention-3 targets Hopper; on an A100 the engine uses FlashAttention-2. Expect performance features to depend on the GPU generation, not just its size.

## Containers solve most of this

NVIDIA's container toolkit lets a container use the host GPU driver while bringing its own CUDA libraries and PyTorch. Official vLLM and SGLang images pin a known-good stack. In production almost everyone uses containers for exactly this reason: the host only needs a recent driver.

## NCCL for multi-GPU

Multi-GPU inference uses NCCL for collective communication. Version mismatches or wrong topology detection show up as hangs, not errors. `NCCL_DEBUG=INFO` prints what NCCL detected (NVLink, PCIe, InfiniBand) and is the first thing to set when multi-GPU runs stall.

## Deep dive: reading the compatibility chain as a set of inequalities

Every pair of adjacent layers has a rule. Writing them as inequalities makes failures easier to read:

⟦M169⟧

⟦M170⟧

⟦M171⟧

~~~dg-raw
%%grid Which features need which GPU architecture (compute capability).
cols = Turing 7.5, Ampere 8.0, Ada 8.9, Hopper 9.0, Blackwell 10.0
key Y = supported | go
key N = not supported | c1
key P = partial or emulated | c2
BF16 matmul | N Y Y Y Y
FP8 matmul | N N Y Y Y
FlashAttention-2 | N Y Y Y Y
FlashAttention-3 | N N N Y P
FP4 matmul | N N N N Y
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**reading an error with the chain**

Error: "no kernel image is available for execution on the device". The third inequality failed: the installed PyTorch or engine wheel was not compiled for your GPU's compute capability.

Error: "CUDA driver version is insufficient for CUDA runtime version". The first inequality failed: upgrade the driver or use a build for an older CUDA.

Error at FP8 start-up on an A100: the feature check failed. FP8 matmul needs capability 8.9 or higher; an A100 is 8.0.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

driver ≥ CUDA runtime ≥ what PyTorch was built with, and the GPU's compute capability decides which fast paths (FP8, FlashAttention-3) exist. Containers pin the whole chain.

</div>
