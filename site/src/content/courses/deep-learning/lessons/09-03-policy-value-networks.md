---
id: "09.03"
course: "deep-learning"
module: "09"
slug: "policy-value-networks"
title: "Policy-value networks"
keyIdeas: ["AlphaGo Zero uses one residual network with two heads: a policy head giving 362 move logits and a value head giving $v\\in[-1,1]$ for the player to move.", "The policy target is the MCTS visit distribution $\\pi$, not the move played; the value target is the final game result $z=\\pm1$ from the same player's perspective.", "The loss is $(z-v)^2-\\pi^\\top\\log p+c\\|\\theta\\|^2$ with $c=10^{-4}$: value regression, policy cross-entropy to the search, and L2 regularisation.", "Sharing one body regularises both heads: the value head alone overfits to game outcomes, while the policy target gives dense supervision on every move.", "Illegal moves are masked out of the policy before it becomes MCTS priors, and the 8 board symmetries multiply the training data and average out evaluation noise."]
question: "Why is the policy target the MCTS visit distribution $\\pi$ rather than a one-hot vector on the move actually played? Give two concrete reasons, considering how the move is chosen in self-play and what the network learns from each target."
answer: "First, the played move is sampled from $\\pi$ (with temperature, and for the first 30 moves stochastically), so a one-hot target would add sampling noise: an exploratory move with 3 percent of the visits would be taught as if it were the search's choice. The full distribution is the less noisy, expected version of that target. Second, $\\pi$ carries much more information: it ranks all moves the search considered, says how close the alternatives were and spreads probability across equally good moves, so the network learns calibrated priors rather than overconfident ones. That matters because those priors steer future searches; a peaked, wrong prior starves good moves of visits under PUCT. More generally, the visit distribution is the output of a policy improvement operator (the search), and matching it with cross-entropy is distillation of that improved policy, the same role the greedy improvement step plays in policy iteration."
sources: [{"title": "Silver et al.: Mastering the game of Go without human knowledge", "url": "https://www.nature.com/articles/nature24270", "note": "Methods: network architecture, the loss and the separate-versus-dual network comparison."}, {"title": "He et al.: Deep Residual Learning for Image Recognition", "url": "https://arxiv.org/abs/1512.03385", "note": "The residual block that the tower is built from."}, {"title": "Wu: Accelerating Self-Play Learning in Go (KataGo)", "url": "https://arxiv.org/abs/1902.10565", "note": "Auxiliary ownership and score targets, global pooling and other upgrades to the AlphaGo Zero network."}]
practice: ["policy-and-value-heads", "alphago-policy-value-loss", "apply-alphago-residual-block", "assemble-compact-dual-head-network", "apply-one-policy-value-head-training-step"]
tags: ["policy-value network", "distillation"]
tools: ["PyTorch"]
---

## From two networks to one

The original AlphaGo (2016) used several networks: a policy network trained by supervised learning on about 30 million positions from human games (predicting the expert move about 57 percent of the time), a second policy network improved by policy-gradient self-play, a value network trained on self-play outcomes, and a fast rollout policy for playouts. AlphaGo Zero replaced all of them with a single network $f_\theta(s)=(p,v)$, trained from random initialisation with no human data. The paper's ablation found that the combined residual network both predicted game outcomes better and played markedly stronger than separate policy and value networks of the same kind.

## Architecture

~~~dg-layers AlphaGo Zero's network (20-block version); every convolution is followed by batch norm and ReLU.
Input | $19\times19\times17$ planes: 8 history planes per player, 1 colour plane
Stem | 3x3 convolution, 256 filters
*Residual tower | 19 residual blocks: conv 3x3, BN, ReLU, conv 3x3, BN, skip add, ReLU
Policy head | 1x1 conv to 2 filters, BN, ReLU, fully connected to 362 logits
Value head | 1x1 conv to 1 filter, BN, ReLU, FC to 256, ReLU, FC to 1, tanh
~~~

The residual tower is a plain ResNet (covered in this course's Convolutional Networks module) at constant $19\times19$ resolution: no pooling, because every point matters and moves are tied to exact locations. With 3x3 kernels, 39 convolutional layers give a receptive field that covers the whole board many times over. The stronger 40-block version doubles the depth. The heads are deliberately tiny: almost all computation and capacity is shared.

## Training targets

Each self-play position $s_t$ yields a training example $(s_t,\pi_t,z_t)$:

- $\pi_t$ is the **search policy**: root visit counts normalised, $\pi(a)\propto N(s_t,a)$. MCTS with 1,600 simulations is a much stronger player than the raw network, so $\pi_t$ is a better policy than $p$. Training $p$ towards $\pi$ distils the search into the network.
- $z_t\in\{-1,+1\}$ is the **final game result** from the perspective of the player to move at $s_t$. It is the Monte Carlo return of lesson 08.03, with no discounting and no bootstrapping.

The loss sums value regression, policy cross-entropy and L2 regularisation:

$$\ell=(z-v)^2-\pi^\top\log p+c\,\|\theta\|^2,\qquad c=10^{-4}$$

```python
def pv_loss(logits, v, pi, z, legal):            # logits: (B, 362), v: (B,), pi: (B, 362), z: (B,)
    logits = logits.masked_fill(~legal, -1e9)     # never put probability on illegal moves
    policy = -(pi * torch.log_softmax(logits, -1)).sum(-1).mean()
    value = ((z - v) ** 2).mean()
    return value + policy                         # L2 via optimizer weight decay
```

The two terms are weighted equally. The value term is in squared units of $[-1,1]$, while cross-entropy on 362 classes can start near $\ln362=5.9$, so the policy term dominates early; that was fine in AlphaGo Zero, and in other settings it is worth monitoring both. Later work (KataGo) adds auxiliary heads, such as final ownership of each point and score margin, which give denser supervision than a single $\pm1$ per game.

<div class="ex">
<div class="lab">Worked example</div>

**the loss on one position**

Three legal moves. Search policy $\pi=(0.7, 0.2, 0.1)$, network policy $p=(0.5, 0.3, 0.2)$, network value $v=0.6$, and the player to move went on to win, $z=+1$.

Value term: $(1-0.6)^2=0.16$.

Policy term: $-(0.7\ln0.5+0.2\ln0.3+0.1\ln0.2)=0.485+0.241+0.161=0.887$.

Total before L2: 1.047. The policy gradient on the logits is $p-\pi=(-0.2, 0.1, 0.1)$, moving probability towards the move the search preferred.

</div>

## Why the value head needs help

A single game of about 200 positions supplies only one bit of outcome information, shared by highly correlated consecutive positions. Trained alone, a value network memorises games: the original AlphaGo found its value network overfitting badly when trained on whole games, and had to sample one position per game. In AlphaGo Zero two things help. The shared tower is also trained by the policy head, which receives a full 362-way target at every position, so the features cannot specialise to memorising outcomes. And self-play produces a vast, constantly refreshed stream of games (lesson 09.04).

## Masking and symmetry

- **Masking.** The network outputs logits for all 362 actions, including occupied points and ko violations. Before the policy is used as MCTS priors, illegal moves are set to $-\infty$ (or their probabilities zeroed and the rest renormalised). Training targets $\pi$ are already zero on illegal moves, so masking the logits in the loss as well prevents wasted probability mass.
- **Symmetry.** The rules are invariant under the 8 rotations and reflections of the board. During training, each example is transformed by a random one of the 8 (the planes and the policy target transform together, while the pass move and value are unchanged). During search, each leaf is evaluated under a random symmetry, which averages out the network's orientation-specific errors.

<div class="take">
<div class="lab">Key takeaway</div>

AlphaGo Zero's network is one ResNet with a 362-way policy head and a tanh value head, trained to match the MCTS visit distribution and the final game result with $(z-v)^2-\pi^\top\log p+c\|\theta\|^2$; sharing the body, masking illegal moves and symmetry augmentation keep both heads honest.

</div>
