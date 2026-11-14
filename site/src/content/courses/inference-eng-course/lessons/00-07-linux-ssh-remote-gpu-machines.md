---
id: "00.07"
course: "inference-eng-course"
module: "00"
slug: "linux-ssh-remote-gpu-machines"
title: "Linux, SSH & Remote GPU Machines"
keyIdeas: ["Most inference work happens on remote Linux GPU machines over SSH.", "Use SSH keys and an ~/.ssh/config entry; forward ports with -L to reach a server running on the box.", "Run long jobs inside tmux or screen so a dropped connection does not kill them.", "Check disk early: model weights are tens to hundreds of GB.", "nvidia-smi is the first health check: GPU type, memory, driver, running processes."]
question: "You rent a cloud GPU box. List the first five commands you run and what each tells you."
answer: "nvidia-smi (GPU model, driver, CUDA version, memory, processes); nproc and free -h (CPU cores and RAM); df -h (disk space for model weights); python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\" (PyTorch sees the GPU); and tmux new -s work so long jobs survive an SSH drop."
sources: [{"title": "nvidia-smi documentation", "url": "https://docs.nvidia.com/deploy/nvidia-smi/index.html", "note": "Every field and query flag."}, {"title": "tmux wiki", "url": "https://github.com/tmux/tmux/wiki", "note": "Keep sessions alive across disconnects."}]
tags: ["linux", "tooling"]
---

## Why you need a remote GPU

Most inference work happens on rented machines: a cloud VM with one to eight GPUs that you reach over SSH. Being comfortable there saves hours every week.

## SSH essentials

- Use key-based login. Generate a key with `ssh-keygen -t ed25519` and add the public key to the provider.
- Put hosts in `~/.ssh/config` so you can type `ssh gpu1` instead of a long command:

```
Host gpu1
  HostName 203.0.113.42
  User ubuntu
  IdentityFile ~/.ssh/id_ed25519
  LocalForward 8000 localhost:8000
```

- `LocalForward` makes the server's port 8000 appear on your laptop's port 8000, so you can call the inference API locally without opening it to the internet.

## Keep work alive: tmux

A dropped SSH connection kills processes started in that shell. Run long jobs inside tmux: `tmux new -s serve`, start the server, detach with Ctrl-b d, reattach later with `tmux attach -t serve`. Use one window for the server, one for the benchmark and one for `nvidia-smi`.

## Disks and model weights

Model checkpoints are large: 16 GB for an 8B model, 140 GB for 70B. Check where the big disk is mounted (`df -h`) and point the Hugging Face cache there with `HF_HOME=/mnt/data/hf`. Downloading to a small root disk is a common way to fill it and crash the machine.

<div class="ex">
<div class="lab">Worked example</div>

**estimating download and load time**

A 70B BF16 checkpoint is 140 GB. At 500 MB/s download, that is 280 s, almost 5 minutes.

Loading from a local NVMe SSD at 3 GB/s takes about 47 s; from a network volume at 300 MB/s, nearly 8 minutes.

This is why cold starts matter in production, and why you should download once to fast local disk.

</div>

## The commands you will use daily

| Command | What it tells you |
| --- | --- |
| nvidia-smi | GPUs, memory use, utilization, processes |
| nvidia-smi topo -m | How GPUs connect (NVLink, PCIe), affects multi-GPU |
| nvtop | Live GPU usage graphs |
| htop | CPU and RAM usage; tokenization and data loading can be CPU-bound |
| df -h, du -sh | Disk space and cache sizes |
| free -g | System RAM, matters for CPU offload |
| journalctl, dmesg | Driver errors such as Xid faults |

## Environments that stay reproducible

Record exact versions: driver, CUDA, PyTorch, engine. Use a virtual environment (`uv venv` or conda) or a container. When a result changes, the first question is always "what changed in the environment", and you can only answer it if you wrote it down.

## Cost hygiene

Cloud GPUs bill by the second or hour whether you use them or not. Stop instances when you are done, keep data on a persistent volume so you can tear the VM down freely, and set a billing alert. An idle 8×H100 node left on over a weekend costs over \$1,000.

## Deep dive: will the transfer or the disk be your bottleneck?

Loading a model is a pipeline of stages. Its time is set by the slowest stage:

⟦M168⟧

~~~dg-raw
%%bars Time to move 140 GB (a 70B model in BF16) through each stage, in seconds.
unit = s
Download at 1 Gb/s | 1120 | c1
Download at 10 Gb/s | 112 | flame
Network disk at 250 MB/s | 560 | c1
Local NVMe at 3 GB/s | 47 | c3
PCIe Gen5 at 50 GB/s | 2.8 | go
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**choosing where to keep weights**

A cloud VM has a 10 Gb/s network, a network-attached disk at 250 MB/s and a local NVMe drive at 3 GB/s.

Downloading to the network disk: limited by the disk, about 9 minutes for 140 GB. Each restart re-reads from that disk: another 9 minutes.

Downloading to local NVMe: about 2 minutes, limited by the network. Each restart reads at 3 GB/s: under a minute.

The local NVMe is erased when the VM stops, so keep a copy in object storage and re-download on a new machine.

</div>

## Deep dive: the cost of forgetting a machine

GPU cost accrues per hour whether or not it does work. A forgotten 8×H100 node at about \$25 per hour costs \$600 a day. A one-line idle shutdown, triggered when `nvidia-smi` reports 0% utilization for an hour, pays for itself the first time it fires.

<div class="take">
<div class="lab">Key takeaway</div>

SSH config with port forwarding, tmux, a big disk for weights, and a short list of monitoring commands make remote GPU work fast and safe.

</div>
