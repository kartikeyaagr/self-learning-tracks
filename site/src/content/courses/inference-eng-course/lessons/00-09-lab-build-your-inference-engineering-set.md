---
id: "00.09"
course: "inference-eng-course"
module: "00"
slug: "lab-build-your-inference-engineering-setup"
title: "Lab: Build Your Inference Engineering Setup"
keyIdeas: ["Goal: a repeatable GPU environment you can rebuild in minutes.", "Use a container or a pinned virtual environment; record versions automatically.", "Cache weights on local NVMe and set HF_HOME.", "Write an env-report script and run it before every experiment.", "Finish with a smoke test: load a small model and generate tokens."]
question: "Write the checklist your setup must pass before you trust any benchmark from it."
answer: "GPU visible to PyTorch with the expected name and memory; driver and CUDA versions recorded; model weights cached on fast local disk; a pinned environment (container or lockfile); a tmux session; a script that prints environment info into every result file; and a smoke test that generates tokens from a small model."
sources: [{"title": "NVIDIA Container Toolkit install guide", "url": "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html", "note": "Running GPU containers with Docker."}, {"title": "Hugging Face Hub documentation", "url": "https://huggingface.co/docs/hub/index", "note": "Downloading and caching models."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal of this lab

Build a GPU environment you can rebuild from scratch in under 15 minutes and that you trust for benchmarking. Every later lab assumes it.

## Step 1: get a machine

Rent one GPU with at least 24 GB of memory (an L4, A10G, RTX 4090, A100 or H100). An 8B model in BF16 needs about 16 GB for weights plus room for the KV cache. Attach a disk of at least 200 GB for model caches.

## Step 2: verify the stack

```
nvidia-smi
nvidia-smi topo -m
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

Record the driver version, CUDA version, GPU model and compute capability in a `ENVIRONMENT.md` file.

## Step 3: create a pinned environment

```
uv venv .venv && source .venv/bin/activate
uv pip install vllm  # pulls a matching torch
uv pip freeze > requirements.lock
export HF_HOME=/mnt/data/hf
```

Commit `requirements.lock` and `ENVIRONMENT.md` to a repository. If you prefer containers, pin an image tag instead, for example `vllm/vllm-openai` at a specific version, never `latest`.

## Step 4: measure the hardware you got

Two quick microbenchmarks tell you whether the GPU performs as specified:

```
import torch, time
x = torch.empty(4 * 1024**3, dtype=torch.uint8, device="cuda")   # 4 GiB
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(10): y = x.clone()
torch.cuda.synchronize(); dt = time.perf_counter() - t
print("copy bandwidth GB/s:", 10 * 2 * x.numel() / dt / 1e9)       # read + write
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): c = a @ a
torch.cuda.synchronize(); dt = time.perf_counter() - t
print("BF16 TFLOP/s:", 20 * 2 * 8192**3 / dt / 1e12)
```

<div class="ex">
<div class="lab">Worked example</div>

**what good numbers look like**

H100 SXM: copy bandwidth about 2,800 to 3,000 GB/s (spec 3,350), BF16 matmul about 650 to 750 TFLOP/s (spec about 990 dense).

A100 80GB: about 1,600 to 1,800 GB/s and 250 to 280 TFLOP/s.

If you get half of these, check for a power cap (`nvidia-smi -q -d POWER`), thermal throttling, or another process on the GPU.

</div>

## Step 5: a smoke test

Download a small model and generate a few tokens:

```
python -c "from vllm import LLM; llm = LLM('Qwen/Qwen2.5-0.5B-Instruct'); print(llm.generate('The capital of France is')[0].outputs[0].text)"
```

If this works, the driver, CUDA, PyTorch and engine all agree.

## Step 6: write it down

Your `ENVIRONMENT.md` should contain: GPU model and count, driver, CUDA, PyTorch and engine versions, measured bandwidth and TFLOP/s, disk paths, and the exact commands to rebuild. When a benchmark looks strange in three weeks, this file is how you rule out the environment.

## Deliverables

- A repository with `ENVIRONMENT.md`, `requirements.lock` (or a pinned image) and a `bench_hw.py` script.
- Measured copy bandwidth and matmul TFLOP/s with a one-line comparison to the spec sheet.
- A screenshot or log of the smoke test.

## Deep dive: turning your measurements into a machine profile

Two microbenchmarks give you the two numbers that drive almost every later estimate.

Memory bandwidth from a large copy: ⟦m173⟧, as in 00.05.

Compute from a large square matmul of size ⟦m174⟧:

⟦M172⟧

The ratio of the two is your machine's measured ridge point, which is usually lower than the spec-sheet ratio.

~~~dg-raw
%%chart Typical achieved BF16 matmul throughput on an H100 as the matrix grows (illustrative shape).
x = 256, 512, 1024, 2048, 4096, 8192, 16384
logx = 1
xlab = Matrix size n (n × n × n)
ylab = TFLOP/s
Achieved: 8, 55, 260, 560, 690, 720, 730
Spec peak*: 989, 989, 989, 989, 989, 989, 989
%%
~~~

Small matmuls cannot fill 132 SMs, so throughput rises with size and then flattens around 70 to 75% of peak. Decode matmuls at batch 1 to 32 live at the far left of this curve, which is why they are memory bound.

<div class="ex">
<div class="lab">Worked example</div>

**writing the profile**

Measured: copy bandwidth 2.95 TB/s, large-matmul 715 TFLOP/s.

Measured ridge: ⟦m175⟧ FLOP/byte, versus 295 from the spec.

Predicted decode step for an 8B model at batch 1: ⟦m176⟧. If your smoke test shows 7 ms, the 1.6 ms difference is overhead to chase later with CUDA graphs.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a reproducible, measured environment is the foundation of every benchmark you will run. Pin versions, measure the hardware, and write it all down.

</div>
