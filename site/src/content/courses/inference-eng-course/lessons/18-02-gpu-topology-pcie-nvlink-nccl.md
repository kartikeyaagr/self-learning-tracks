---
id: "18.02"
course: "inference-eng-course"
module: "18"
slug: "gpu-topology-pcie-nvlink-nccl"
title: "GPU Topology, PCIe, NVLink & NCCL"
keyIdeas: ["GPUs talk over NVLink (fast, inside a node), PCIe (slower) or network (between nodes).", "nvidia-smi topo -m shows the link type between each GPU pair.", "NCCL implements collectives (all-reduce, all-gather, all-to-all) over the best available path.", "NVLink bandwidth is hundreds of GB/s per GPU; PCIe Gen5 x16 is about 64 GB/s per direction.", "Between nodes: InfiniBand or RoCE, often 400 Gb/s per GPU."]
question: "How would you check how the GPUs in a node are connected, and why does it matter for tensor parallelism?"
answer: "Run nvidia-smi topo -m, which shows for each GPU pair whether they connect through NVLink (NV#), a PCIe switch (PIX/PXB) or across CPU sockets (SYS). Tensor parallelism does all-reduces every layer, so it needs high bandwidth and low latency; over NVLink it scales well, over PCIe or across sockets the communication can dominate step time."
sources: [{"title": "NCCL User Guide", "url": "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html", "note": "Collectives, environment variables and debugging."}, {"title": "NVIDIA NVLink and NVSwitch", "url": "https://www.nvidia.com/en-us/data-center/nvlink/", "note": "Interconnect specs."}]
tags: ["interconnect", "nccl"]
---

## Link types

GPUs talk to each other over several kinds of links, with bandwidths that differ by more than ten times.

| Link | Typical bandwidth per GPU | Where you find it |
| --- | --- | --- |
| NVLink 4 through NVSwitch (H100) | about 900 GB/s total, 450 GB/s each way | inside an HGX or DGX server |
| NVLink 5 (Blackwell) | about 1.8 TB/s total | inside Blackwell servers and NVL72 racks |
| PCIe Gen5 x16 | about 64 GB/s each way in theory, 50 to 55 in practice | PCIe servers and workstations |
| PCIe Gen4 x16 | about 32 GB/s each way in theory | older servers, consumer GPUs |
| InfiniBand or RoCE at 400 Gb/s | about 50 GB/s each way | between servers |

Inside an 8-GPU HGX H100 server, every GPU connects to NVSwitch chips, so any pair of GPUs can talk at full NVLink speed. In a PCIe server, GPUs may share PCIe switches, and traffic between GPUs on different CPU sockets crosses the inter-socket link, which is slower still.

## Checking topology

```
nvidia-smi topo -m
```

The matrix shows how each pair of GPUs is connected:

- **NV#**: connected by # NVLink links (NV18 on H100 HGX).
- **PIX**: through a single PCIe switch.
- **PXB**: through several PCIe switches.
- **PHB**: through the PCIe host bridge on the CPU.
- **SYS**: across CPU sockets, the slowest path.

Check this before choosing a parallelism layout. Tensor parallelism across GPUs connected by SYS links will be slow.

## NCCL

NCCL (NVIDIA Collective Communications Library) implements the collective operations that multi-GPU inference uses, such as all-reduce, all-gather and all-to-all. It detects the topology and picks algorithms such as rings or trees over the available links.

Useful settings while investigating:

- `NCCL_DEBUG=INFO` logs the transports and topology NCCL chose. Check that it uses NVLink (P2P) inside a server and InfiniBand between servers, not sockets.
- `NCCL_IB_HCA` selects which network adapters to use.
- `NCCL_P2P_DISABLE=1` forces traffic away from direct GPU-to-GPU paths. Use it only for debugging.

## Test the links before blaming the engine

NVIDIA's `nccl-tests` measure the raw speed of each collective:

```
./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8
```

This runs all-reduce on 8 GPUs with message sizes from 8 bytes to 1 GB. Compare the measured bus bandwidth at large sizes with what the links should deliver. A result far below expectations points to a configuration or hardware problem, such as a missing NVLink, a misconfigured network adapter or a degraded cable.

<div class="ex">
<div class="lab">Worked example</div>

**why the link decides**

Tensor parallelism of 8 on Llama 3 70B. At batch 64, each all-reduce moves 64 × 8,192 × 2 bytes = 1 MB. There are 2 per layer and 80 layers, so 160 per step.

Over NVLink at an effective 300 GB/s plus about 10 µs of latency: about 13 µs each, about 2 ms per step.

Over PCIe Gen4 at an effective 20 GB/s plus latency: about 60 µs each, about 10 ms per step, more than the step's compute.

The same layout is a good choice on an HGX server and a poor one on a PCIe box.

</div>

## Deep dive: the cost of an all-reduce

A ring all-reduce over ⟦m899⟧ GPUs of a message of ⟦m900⟧ bytes does a reduce-scatter then an all-gather. Each GPU sends and receives ⟦m901⟧ bytes, in ⟦m902⟧ steps:

⟦M898⟧

⟦m903⟧ is the per-step latency (a few microseconds on NVLink, more over PCIe or a network). NCCL reports "bus bandwidth", which is the algorithm bandwidth multiplied by ⟦m904⟧, so that it can be compared with the link speed directly.

~~~dg-raw
%%chart Predicted all-reduce time on 8 GPUs against message size (α = 2 µs per step).
x = 8, 64, 512, 4096, 32768, 262144
logx = 1
logy = 1
xlab = Message size (KB)
ylab = µs
NVLink, 450 GB/s per direction: 28, 28, 30, 44, 158, 1072
PCIe, 25 GB/s effective: 29, 33, 65, 322, 2377, 18819
%%
~~~

Below about 100 KB, both lines are flat: the latency term dominates, and a faster link does not help. Decode all-reduces at small batch are 8 KB to 1 MB, so decode TP is latency-bound. Prefill all-reduces are tens to hundreds of MB, where link bandwidth decides.

<div class="take">
<div class="lab">Key takeaway</div>

know your topology before choosing parallelism. NVLink makes tensor parallelism cheap, while PCIe and network links make it expensive. Test the raw collective speed with nccl-tests before blaming the engine.

</div>
