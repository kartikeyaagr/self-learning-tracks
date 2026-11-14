---
id: "09.03"
course: "deep-learning"
module: "09"
slug: "policy-value-networks"
title: "Policy-value networks"
keyIdeas: ["A **policy-value network** $f_\\theta(s)=(p,v)$ is one network with a shared residual body and two small heads: a policy head with one logit per move (362 in Go) and a tanh value head giving $v\\in[-1,1]$ for the player to move.", "The policy target is the MCTS visit distribution $\\pi$, not the move played; the value target is the final result $z=\\pm1$ from the same player's view. Copying the search into the network is called **distillation**.", "The loss is $(z-v)^2-\\pi^\\top\\log p+c\\|\\theta\\|^2$ with $c=10^{-4}$. The policy gradient on the logits is simply $p-\\pi$, and the policy loss can never go below the entropy of $\\pi$.", "The value head sees one outcome per game and overfits easily; sharing the body with the densely supervised policy head, more games and the 8 board symmetries all fight this.", "Illegal moves are masked to $-\\infty$ before the softmax, and training examples are rotated or reflected at random, with the policy target transformed together with the board."]
question: "You are training a policy-value network for 9x9 Go from self-play. After a day, the held-out policy cross-entropy has stalled at 1.9 nats, and the held-out value error is 1.05 and rising while the training value error is 0.35. A teammate proposes (1) switching the policy target to the one-hot move actually played, since 'the policy loss is stuck', and (2) training a separate value network so the value problem stops 'hurting' the policy. What is going on, and what would you do instead?"
answer: "Read the two numbers against their reference lines. The policy cross-entropy has a floor equal to the average entropy of the search targets $\\pi$: the search spreads its visits, so even $p=\\pi$ leaves a positive loss. Compute that entropy on the held-out set; if it is, say, 1.7 nats, the network is only 0.2 nats (the KL) away from the search and the policy is not stuck at all. Switching to one-hot played moves would make things worse: the played move is a sample from $\\pi$ (stochastic early in the game), so it adds noise, throws away the ranking of alternatives, and produces overconfident priors that starve good moves of visits in PUCT. The value numbers are the real problem: a held-out error of 1.05 against a training error of 0.35 is memorisation, and it is probably already worse than always predicting the average result (the variance of $z$, at most 1). Each game supplies only one outcome for many highly correlated positions. A separate value network would remove the regularising effect of the shared body, which receives a dense 81-plus-pass target at every position, and the original AlphaGo's separate value network overfitted for exactly this reason. Instead: keep one network; add random symmetry augmentation (8 rotations/reflections, transforming the planes, $\\pi$ and the legal mask together); generate more games and train from a larger, refreshed window of recent games; sample fewer positions per game or downweight the value loss if needed; check that $z$ is taken from the player to move; and monitor value error against the variance-of-$z$ baseline and policy loss as KL above the entropy floor."
sources: [{"title": "Silver et al.: Mastering the game of Go without human knowledge", "url": "https://www.nature.com/articles/nature24270", "note": "Methods: network architecture, the loss and the separate-versus-dual network comparison."}, {"title": "He et al.: Deep Residual Learning for Image Recognition", "url": "https://arxiv.org/abs/1512.03385", "note": "The residual block that the tower is built from."}, {"title": "Wu: Accelerating Self-Play Learning in Go (KataGo)", "url": "https://arxiv.org/abs/1902.10565", "note": "Auxiliary ownership and score targets, global pooling and other upgrades to the AlphaGo Zero network."}]
practice: ["policy-and-value-heads", "alphago-policy-value-loss", "apply-alphago-residual-block", "assemble-compact-dual-head-network", "apply-one-policy-value-head-training-step"]
tags: ["policy-value network", "distillation"]
tools: ["PyTorch"]
---

## What you will learn

- What a **policy-value network** is: one neural network that looks at a board and returns both "which moves look good" and "who is winning".
- How AlphaGo Zero's network is built: a residual tower shared by a 362-way policy head and a single-number value head.
- Where the training targets come from: the search's visit distribution $\pi$ for the policy, the final game result $z$ for the value.
- How to compute the loss $(z-v)^2-\pi^\top\log p+c\|\theta\|^2$ by hand and in PyTorch, including masking illegal moves.
- Why one shared body, full search distributions and board symmetries matter, shown with real training runs on tic-tac-toe.

*Before you start:* you need Monte Carlo tree search and the PUCT rule (09.02), softmax and cross-entropy for classification, and residual blocks from the Convolutional Networks module. The Monte Carlo return from lesson 08.03 explains the value target.

Lesson 09.02 left one question open. PUCT needs, at every new leaf, a **prior** $P(s,a)$ over moves and a **value** $v$ for the position. Something has to produce those numbers thousands of times per move, quickly and well. In AlphaGo Zero that something is a single network $f_\theta(s)=(p,v)$, and this lesson is about how it is built and what it is trained to predict.

## The idea in one picture

Take one position with three legal moves, A, B and C. The network is called once and returns two things:

- a **policy** $p$: a probability for each move, here $p=(0.5,0.3,0.2)$;
- a **value** $v$: a number in $[-1,1]$ saying how good the position is for the player to move, here $v=0.6$ ("probably winning").

Then MCTS runs from this position, using $p$ as priors and values at its leaves. After many simulations it has visited A 70% of the time, B 20% and C 10%. That **search policy** $\pi=(0.7,0.2,0.1)$ is better than $p$, because it looked ahead. The game is played to the end and the player who was to move here wins, so the **game result** is $z=+1$.

| | move A | move B | move C | value
| --- | --- | --- | --- | --- |
| network output | $p=0.5$ | $p=0.3$ | $p=0.2$ | $v=0.6$
| training target | $\pi=0.7$ | $\pi=0.2$ | $\pi=0.1$ | $z=+1$

~~~dg-fig 09.03-example.svg
Left: what the network said (blue) against what the search found (red). Right: the gradient of the policy loss on each logit is simply p − π, so training raises move A and lowers B and C.
~~~

Read the table as two small supervised problems sharing one input: a classification with a *soft* label ($\pi$ instead of one correct class) and a regression with target $+1$. Training nudges $p$ towards what the search preferred and $v$ towards what actually happened, so the next search starts from better priors. That loop is lesson 09.04; this lesson is about the network in the middle.

## Step 1: one body, two heads

Earlier systems used separate networks. The original AlphaGo (2016) had a policy network trained by supervised learning on about 30 million positions from human games (it predicted the expert move about 57% of the time), a second policy network improved by policy-gradient self-play, a value network trained on self-play outcomes, and a small fast rollout policy for playouts. AlphaGo Zero replaced all of them with one network, trained from random weights with no human data.

The design is a **shared body** (a deep stack of layers that turns the board into features) followed by two small **heads** (output layers that read those features for one task each).

~~~dg-layers AlphaGo Zero's network (20-block version); every convolution is followed by batch norm and ReLU.
Input | $19\times19\times17$ planes: 8 history planes per player, 1 colour plane
Stem | 3x3 convolution, 256 filters
*Residual tower | 19 residual blocks: conv 3x3, BN, ReLU, conv 3x3, BN, skip add, ReLU
Policy head | 1x1 conv to 2 filters, BN, ReLU, fully connected to 362 logits
Value head | 1x1 conv to 1 filter, BN, ReLU, FC to 256, ReLU, FC to 1, tanh
~~~

What each piece means:

- **Input planes.** A plane is a $19\times19$ grid of 0s and 1s. Eight planes mark the current player's stones in the last 8 positions, eight more mark the opponent's, and one constant plane says whose turn it is. History is needed because the ko rule depends on earlier positions.
- **Residual tower.** A plain ResNet kept at full $19\times19$ resolution: no pooling, because every point matters and moves are tied to exact locations. With 39 convolutions of size 3x3, each output point can see a $79\times79$ window, about four board widths, so information from any corner reaches every point.
- **Policy head.** 362 **logits** (unnormalised scores), one per point plus pass; a softmax gives $p$.
- **Value head.** One number squashed by $\tanh$ into $[-1,1]$; $+1$ means the player to move wins.

The heads are deliberately tiny. Counting the parameters of the PyTorch model in the code section below gives 22.8 million in total, and 98.3% of them sit in the tower.

~~~dg-fig 09.03-params.svg
Where AlphaGo Zero's parameters live (log scale). Almost everything is shared; the two heads together hold about 1.6%. The 40-block version doubles the tower.
~~~

Number check: one 3x3 convolution from 256 to 256 channels has $3\cdot3\cdot256\cdot256=589{,}824$ weights; the tower's 38 of them give 22.41 million, plus 19,456 batch-norm parameters.

## Step 2: the two targets

Every position $s_t$ played in self-play becomes one training example $(s_t,\pi_t,z_t)$.

**The policy target $\pi_t$** is the root visit distribution from the search:

$$\pi_t(a)=\frac{N(s_t,a)}{\sum_b N(s_t,b)},$$

where $N(s_t,a)$ is how many of the search's simulations went through move $a$. AlphaGo Zero used 1,600 simulations per move. MCTS with that budget plays far better than the raw network, so $\pi_t$ is a better policy than $p$. Training $p$ towards $\pi_t$ is **distillation**: compressing what an expensive procedure (search) knows into a cheap one (one forward pass).

**The value target $z_t$** is the final result of the game, $+1$ or $-1$, seen from the side of the player to move at $s_t$. It is the Monte Carlo return of lesson 08.03 with no discounting and no bootstrapping: nothing is estimated, the game simply finished. The sign flip matters. In one game the positions where Black is to move all get $z=+1$ if Black won, and the positions where White is to move all get $z=-1$.

## Step 3: the loss

The loss adds a value regression, a policy cross-entropy and weight decay:

$$\ell=(z-v)^2-\pi^\top\log p+c\,\|\theta\|^2,\qquad c=10^{-4}.$$

Symbol by symbol:

- $(z-v)^2$: squared error between the game result and the predicted value. It ranges from 0 to 4.
- $\pi^\top\log p=\sum_a\pi(a)\log p(a)$: the log-probability the network gives to the search's choices, weighted by how much the search liked each. The minus sign turns it into **cross-entropy**, which is small when $p$ puts mass where $\pi$ does.
- $c\|\theta\|^2$: L2 regularisation on all weights $\theta$. In code this is usually the optimiser's weight decay.

The two terms get equal weight but live on different scales: an untrained 362-way softmax has cross-entropy near $\ln362=5.9$, while the value error starts around 1, so the policy term dominates early. Log both separately.

<div class="ex">
<div class="lab">Worked example</div>

**the loss on one position**

Use the numbers from the picture: $\pi=(0.7,0.2,0.1)$, $p=(0.5,0.3,0.2)$, $v=0.6$, $z=+1$.

Value term: $z-v=1-0.6=0.4$, squared is $0.16$.

Policy term, one move at a time: $0.7\ln0.5=0.7\times(-0.693)=-0.485$; $0.2\ln0.3=0.2\times(-1.204)=-0.241$; $0.1\ln0.2=0.1\times(-1.609)=-0.161$. Sum $-0.887$, so the cross-entropy is $0.887$.

Total before L2: $0.16+0.887=1.047$.

Gradient of the policy term on the logits: $p-\pi=(0.5-0.7,\;0.3-0.2,\;0.2-0.1)=(-0.2,+0.1,+0.1)$. Gradient descent subtracts it, so A's logit goes up and B's and C's go down: probability moves towards the move the search preferred.

Lower bound: even a perfect network with $p=\pi$ cannot reach zero policy loss. The minimum is the entropy of $\pi$, $-(0.7\ln0.7+0.2\ln0.2+0.1\ln0.1)=0.250+0.322+0.230=0.802$. Only the gap above it, $0.887-0.802=0.085$, is the network's error (the KL divergence).

</div>

## Step 4: masking illegal moves

The policy head always produces 362 logits, including ones for occupied points and ko violations. Those moves must get zero probability before $p$ becomes priors in MCTS, otherwise the search would waste its prior on moves it can never play.

The fix is to set illegal logits to $-\infty$ (in practice $-10^9$) before the softmax. That is the same as zeroing their probabilities and renormalising the rest.

<div class="ex">
<div class="lab">Worked example</div>

**masking one illegal move**

Logits for four moves are $(2.0, 1.0, 0.5, 3.0)$, and the fourth point is occupied.

Without masking: $e^{2}=7.39$, $e^{1}=2.72$, $e^{0.5}=1.65$, $e^{3}=20.09$; total 31.84. Softmax $=(0.232, 0.085, 0.052, 0.631)$. The network spends 63% of its prior on a move that does not exist.

With masking: drop the fourth term; total $7.39+2.72+1.65=11.76$. Softmax $=(0.629, 0.231, 0.140)$, which is exactly the first three numbers divided by $1-0.631=0.369$.

</div>

Training targets $\pi$ are already zero on illegal moves; masking in the loss too just stops the network wasting effort on logits that will be masked anyway.

## Step 5: using the board's symmetries

Go's rules do not change if you rotate the board by 90 degrees or reflect it: there are 8 such **symmetries** (4 rotations, each with or without a mirror). AlphaGo Zero used them twice:

- **Training.** Each example is transformed by a random one of the 8. The input planes and the 361 board entries of $\pi$ are rotated together; the pass entry and $z$ stay as they are. This multiplies the effective data by up to 8 for free.
- **Search.** Each leaf is evaluated under a random symmetry, averaging out orientation-specific errors.

AlphaZero (chess, shogi and Go) dropped both, since chess and shogi are not rotation-symmetric.

## A policy-value network in code

The full-size network, the loss, and a check of the worked example (runs on a CPU in seconds).

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

def conv_bn(cin, cout, k):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout))

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__(); self.a, self.b = conv_bn(c, c, 3), conv_bn(c, c, 3)
    def forward(self, x):
        return F.relu(x + self.b(F.relu(self.a(x))))        # skip connection, then ReLU

class PolicyValueNet(nn.Module):
    def __init__(self, planes=17, c=256, blocks=19, size=19):
        super().__init__(); n = size * size
        self.stem = nn.Sequential(conv_bn(planes, c, 3), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(c) for _ in range(blocks)])
        self.policy = nn.Sequential(conv_bn(c, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(2 * n, n + 1))
        self.value = nn.Sequential(conv_bn(c, 1, 1), nn.ReLU(), nn.Flatten(),
                                   nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh())
    def forward(self, x):
        h = self.tower(self.stem(x))                          # shared body
        return self.policy(h), self.value(h).squeeze(-1)      # 362 logits, value in [-1, 1]

net = PolicyValueNet().eval()
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"{count(net):,}", f"{count(net.tower):,}")  # 22,827,877 22,432,768
with torch.no_grad():
    logits, v = net(torch.randn(2, 17, 19, 19))
print(logits.shape, v.shape)                       # torch.Size([2, 362]) torch.Size([2])

def pv_loss(logits, v, pi, z, legal):
    logits = logits.masked_fill(~legal, -1e9)                 # no mass on illegal moves
    policy = -(pi * F.log_softmax(logits, -1)).sum(-1).mean()  # cross-entropy to the search
    value = ((z - v) ** 2).mean()                              # squared error to the result
    return value, policy                                       # L2: weight_decay=1e-4 in the optimiser

# worked example; a 4th, illegal move gets a large logit and is masked away
logits = torch.log(torch.tensor([[0.5, 0.3, 0.2, 0.9]])).requires_grad_()
legal = torch.tensor([[True, True, True, False]])
value, policy = pv_loss(logits, torch.tensor([0.6]), torch.tensor([[0.7, 0.2, 0.1, 0.0]]),
                        torch.tensor([1.0]), legal)
policy.backward()
print(round(value.item(), 3), round(policy.item(), 3))   # 0.16 0.887
print(logits.grad)                                        # [[-0.2, 0.1, 0.1, 0.0]]
```

The illegal move's gradient is exactly zero: once masked, it no longer takes part in the softmax.

## Watching one train: a tic-tac-toe experiment

Go is too big for a laptop, so the figures use tic-tac-toe with the same recipe. I generated 3,000 self-play games with a PUCT search of 48 simulations per move (uniform priors and random playouts at the leaves, so no network yet) and sampled each move from the visit counts. The first player won 1,642 games, drew 652 and lost 706, so $z\in\{-1,0,+1\}$ here. A miniature AlphaGo Zero network (a 32-filter stem, 2 residual blocks, the same two heads, 3 input planes) was trained with Adam on 2,500 games (17,482 positions) and tested on the other 500 games (3,530 positions).

~~~dg-fig 09.03-curves.svg
Training the two-head network on 2,500 tic-tac-toe games. Left: the policy loss levels off at 1.45, above the 1.16 floor set by the entropy of the targets. Right: the value error drops from 0.71 (no better than guessing the average) to about 0.52. Train and held-out curves stay close: no overfitting at this data size.
~~~

How to read these curves:

- **The policy loss has a floor.** It cannot fall below the average entropy of the targets, 1.16 nats here, because the search itself spreads its visits. The meaningful number is the gap above that floor (about 0.29 nats here). Comparing raw cross-entropy across datasets with different search budgets is misleading.
- **The value loss has a baseline.** Always predicting the average result gives an error equal to the variance of $z$, 0.71 on the test games. A value head that does not beat that line has learned nothing.
- **The value error stays well above 0.** Moves were sampled, so one position can lead to a win or a loss; the best possible $v$ is the *expected* outcome.
- **Agreement with the search.** The network's top move matched the held-out search's top move 58% of the time; with only 48 noisy simulations, $\pi$ itself is far from perfect.

## Why one shared body: the value head overfits

One game gives one outcome, shared by many near-identical positions, so a value network trained alone tends to memorise games. The original AlphaGo hit exactly this: trained on whole human games, its value network overfit badly, and the fix was to build a new dataset with only one position sampled from each of about 30 million separate self-play games.

The policy target gives a full distribution at every position, and sharing the body lets that dense signal shape the features the value head reads.

To see the effect, I retrained on only 100 games (689 positions), three ways, averaging three seeds.

~~~dg-fig 09.03-shared.svg
With 100 games the value head alone reaches a training error of 0.30 but a held-out error of 0.90, worse than guessing the average (0.71). Sharing the body with the policy head helps a little (0.86). Training on random board symmetries helps much more (0.61). The policy loss barely differs between a policy-only network and the two-head one.
~~~

So the value-only network memorised (training error 0.30, held-out 0.90). The policy head helped only slightly on this tiny problem; the 8 symmetries helped far more (value 0.61, policy cross-entropy 1.47 instead of 1.62) because they genuinely add positions; and more data helps most: with 2,500 games the same network reached 0.52.

The AlphaGo Zero paper's own ablation agrees in direction. Stated from memory, combining policy and value into one network slightly lowered move-prediction accuracy but lowered value error and gained about 600 Elo in playing strength over separate networks. My policy panel shows the same small cost: 1.616 with two heads against 1.605 for policy alone.

## Why the full $\pi$ and not the move played

Why not use the move actually played as a one-hot target? I trained the same network both ways on 50, 200 and 800 games.

~~~dg-fig 09.03-targets.svg
Training on the search's full visit distribution beats training on the one-hot played move at every data size. Left: how often the top move matches the held-out search's favourite. Right: cross-entropy to the held-out search distribution; the one-hot target gives badly overconfident policies when data is scarce.
~~~

With 50 games the $\pi$-trained network's top move agreed with the search 37% of the time against 29% for one-hot; at 800 games, 56% against 48%. The cross-entropy gap is larger: 1.68 against 3.33 at 50 games. Two reasons:

- **Less noise.** The played move is a *sample* from $\pi$. A move that got 3% of the visits is sometimes played, and a one-hot target would teach it as if it were the search's choice. $\pi$ is the expected value of that one-hot target.
- **More information.** $\pi$ ranks every move and shows how close the alternatives were. The network learns calibrated priors, which matters because overconfident wrong priors starve good moves of visits under PUCT.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Value loss stuck at the variance of $z$ | value head not learning; $z$ sign from the wrong player's view | check $z$ is from the player to move; check the tanh is not saturated at init
| Held-out value error rises while training error falls | memorising few games | more games, symmetry augmentation, sample fewer positions per game
| Search plays illegal moves or wastes priors | no masking before the softmax | mask illegal logits to $-10^9$ before it becomes priors
| Policy loss "stuck" well above 0 | normal: entropy floor of $\pi$ | track cross-entropy minus target entropy (KL)
| Rotated position gives a different move | $\pi$ not rotated together with the planes | apply the same permutation to planes, $\pi$ and legal mask
| Strong offline numbers, weak play | batch norm left in training mode at inference | call `net.eval()` before search

## Going deeper

**Gradients.** For softmax cross-entropy, $\partial\ell/\partial\text{logit}_a=p_a-\pi_a$ whenever $\pi$ sums to 1, which is why the update is so simple. For the value, $\partial\ell/\partial v=-2(z-v)$, and through the tanh it is multiplied by $1-v^2$. When $|v|$ is near 1 that factor is near 0, so a confidently wrong value learns slowly.

**Distillation view.** Minimising $-\pi^\top\log p$ over $p$ is the same as minimising $\mathrm{KL}(\pi\,\|\,p)$, because the entropy of $\pi$ does not depend on $\theta$. KL in this direction is "mass-covering": $p$ is punished for giving low probability to any move the search visited, so it learns to keep plausible alternatives alive, which is what good priors need. Search is the policy improvement step; the network update projects it back into the space of networks (09.04).

**Why squared error for a $\pm1$ target?** With $z\in\{-1,+1\}$, the best $v$ under squared error is the expected result $2P(\text{win})-1$, so $v$ is a scaled win probability. KataGo adds auxiliary heads for each point's final ownership and the score margin, giving a target at every intersection instead of one bit per game.

## Key terms

| Term | Meaning
| --- | --- |
| Policy-value network | One network $f_\theta(s)=(p,v)$ that outputs move probabilities and a position value
| Shared body / residual tower | The layers both heads read from; holds almost all parameters
| Policy head | Small output layer producing one logit per move (362 in Go)
| Value head | Small output layer producing a number in $[-1,1]$ via tanh
| Search policy $\pi$ | Root visit counts of MCTS, normalised; the policy target
| Game result $z$ | Final outcome, $\pm1$, from the view of the player to move; the value target
| Distillation | Training a cheap model to copy the output of an expensive one (here, search)
| Masking | Setting illegal moves' logits to $-\infty$ before the softmax
| Symmetry augmentation | Training on random rotations and reflections of each position

<div class="take">
<div class="lab">Key takeaway</div>

a policy-value network is one shared ResNet with a tiny move head and a tiny tanh value head, trained with $(z-v)^2-\pi^\top\log p+c\|\theta\|^2$ to copy the search's visit distribution and predict the game result; mask illegal moves, use the full $\pi$ rather than the played move, and fight value-head overfitting with more games, symmetries and the shared body.

</div>
