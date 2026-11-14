---
id: "13.05"
course: "ai-research-track-web"
module: "13"
slug: "scaling-laws"
title: "Scaling laws"
keyIdeas: ["Pretraining loss falls as a power law toward a floor, $L(x) \\approx E + (x_0/x)^{\\alpha}$, as parameters, data or compute grow. Subtract the floor $E$ and the rest is a straight line on log-log axes, which is what lets cheap small runs predict an expensive big one.", "Kaplan et al. (2020) concluded that extra compute should go mostly into a bigger model ($N \\propto C^{0.73}$). Hoffmann et al. (2022, Chinchilla) found parameters and tokens should grow about equally ($N, D \\propto C^{0.5}$), roughly 20 tokens per parameter, so $C \\approx 120N^2$ at the optimum.", "An **IsoFLOP sweep** trains several model sizes at each fixed budget, fits a parabola in $\\log N$ to find the best size, and fits a line to those optima against $\\log C$. The slope is the allocation exponent; for $L = E + A/N^{\\alpha} + B/D^{\\beta}$ it equals $\\beta/(\\alpha+\\beta)$.", "The loss is flat near the bottom of an IsoFLOP curve, so a model several times smaller than compute-optimal loses very little. Because serving costs about $2N$ FLOPs per token, deployed models are deliberately smaller and over-trained.", "Extrapolate with care: fit the floor explicitly, match each run's learning-rate schedule to its token count, tune hyperparameters at every scale, hold out the largest runs to test the fit, and predict loss rather than benchmark accuracy."]
question: "You have a training budget of $2 \\times 10^{23}$ FLOPs and expect to serve $10^{13}$ tokens over the model's lifetime. Compute the Chinchilla-style compute-optimal size and token count at 20 tokens per parameter, then the tokens a 7B model would see with the same budget. Compare lifetime FLOPs (training plus about $2N$ per served token). What would you need to check before committing to the 7B option?"
answer: "Compute-optimal: at 20 tokens per parameter $C = 6N \\times 20N = 120N^2$, so $N = \\sqrt{2 \\times 10^{23}/120} = 40.8$B parameters and $D = 20N = 816$B tokens. A 7B model with the same budget sees $2 \\times 10^{23}/(6 \\times 7 \\times 10^9) = 4.76$T tokens, about 680 tokens per parameter. Serving cost: $2 \\times 40.8\\text{B} \\times 10^{13} = 8.2 \\times 10^{23}$ FLOPs versus $2 \\times 7\\text{B} \\times 10^{13} = 1.4 \\times 10^{23}$. Lifetime totals: about $1.02 \\times 10^{24}$ for the 41B model versus $3.4 \\times 10^{23}$ for the 7B, three times cheaper overall, because serving dominates. The catch is quality: the 7B model is off the compute-optimal point, so at equal training compute its loss is somewhat higher (the Chinchilla approach-3 fit predicts 1.988 versus 1.980, under half a percent, but do not trust the fit's third digit; that fit is known to disagree with the paper's other methods). Before committing, check that the extra tokens exist at the same quality (4.76T unique tokens, or an acceptable amount of repetition), validate with your own IsoFLOP or over-training points at smaller scale that the loss gap is as small as predicted (hold out the largest small runs to test the fit), make sure the learning-rate schedule and batch size are tuned for the long run, and check downstream metrics rather than loss alone. If the gap matters, spending part of the inference savings on more training tokens for the 7B model is usually the better trade."
sources: [{"title": "Hoffmann et al.: Training Compute-Optimal Large Language Models (2022)", "url": "https://arxiv.org/abs/2203.15556", "note": "Section 3: the three estimation approaches, the IsoFLOP figures and the Chinchilla versus Gopher comparison."}, {"title": "Kaplan et al.: Scaling Laws for Neural Language Models (2020)", "url": "https://arxiv.org/abs/2001.08361", "note": "The original power-law fits for N, D and C and the critical batch size discussion."}, {"title": "How to Scale Your Model (JAX scaling book)", "url": "https://jax-ml.github.io/scaling-book/", "note": "How compute budgets translate into hardware time and model shape choices."}]
practice: ["isoflop-sweep-exponent", "chinchilla-compute-optimal", "saturating-power-law-fit", "fixed-ratio-budget-split", "fit-learning-rate-and-batch-size-scaling"]
tags: ["scaling laws", "Chinchilla", "compute-optimal"]
tools: ["NumPy", "SciPy"]
---

## What you will learn

- What a scaling law is: why the loss of a language model falls along a smooth power law as you add parameters, data or compute, and what the irreducible floor means.
- How to read and fit the form $L(x) = E + (x_0/x)^{\alpha}$, and why the floor $E$ changes what you predict.
- The two landmark answers to "how should a compute budget be split between model size and tokens": Kaplan et al. (2020) and Chinchilla (Hoffmann et al., 2022), and why they disagreed.
- How to run an IsoFLOP sweep yourself and turn it into an allocation exponent, by hand and in NumPy.
- Why deployed models are deliberately smaller and "over-trained", and the traps that make extrapolations wrong.

*Before you start:* you need the compute rule $C \approx 6ND$ from lesson 13.01 (about 6 FLOPs per parameter per training token: 2 forward, 4 backward), logarithms, and least-squares fitting of a straight line. Lessons 13.02 to 13.04 showed how to spend a budget efficiently on hardware; this lesson asks what the budget should be spent *on*.

## Why this lesson comes next

Lessons 13.01 to 13.04 took the model size and token count as given and fitted the job onto hardware. But a frontier run is expensive and you get one shot, so first you must choose how many parameters, how many tokens, and what loss to expect. Scaling laws answer that by fitting simple curves to many cheap runs.

## The idea in one picture

Imagine you train nine models that are identical except for size, from 20 million to 500 billion parameters, each on plenty of fresh data. You record the final loss (average cross-entropy per token, in nats). The numbers look like this (simulated from the published Chinchilla fit, explained below):

| Parameters | 20M | 100M | 1B | 10B | 100B | 500B
| --- | --- | --- | --- | --- | --- | --- |
| Final loss | 3.03 | 2.46 | 2.04 | 1.85 | 1.76 | 1.73

Each tenfold increase in size helps less than the one before: 0.42, then 0.19, then 0.09. Plotted on log-log axes, the loss bends toward a level it never crosses.

~~~dg-fig 13.05-powerlaw.svg
Left: loss against model size on log-log axes bends toward the floor E = 1.69. Right: subtract the floor and what is left falls on a straight line; its slope is the exponent α. Points are simulated runs.
~~~

The right panel is the key observation. Once you subtract a constant, the remaining loss is a straight line in log-log space, which means it is a **power law**. That is a scaling law. Some terms:

- The **irreducible loss** $E$ is the floor. Even an infinite model cannot predict text perfectly because text is genuinely uncertain (which word comes next in a novel is not determined). Roughly, $E$ is the entropy of the text under your tokenizer.
- The **reducible loss** $L - E$ is the part that more scale can remove.
- The **scaling exponent** $\alpha$ is the slope of the straight line (with a minus sign). It says how fast the reducible loss shrinks.

A straight line fitted on runs costing a thousandth of your budget predicts where the big run will land before you spend the money.

## Step 1: the shape of a scaling law

In words: the loss equals a floor plus a term that shrinks like a power of scale. In symbols:

$$L(x) = E + \left(\frac{x_0}{x}\right)^{\alpha}.$$

- $x$ is the scale you vary: parameters $N$, training tokens $D$, or compute $C$.
- $E$ is the irreducible loss (same units as $L$, nats per token).
- $x_0$ is a scale constant: the size at which the reducible part equals 1 nat.
- $\alpha > 0$ is the exponent.

Take logs of the reducible part and you get $\log(L - E) = \alpha \log x_0 - \alpha \log x$: a line with slope $-\alpha$. If you ignore $E$ (set it to zero), the whole loss is a line in log-log space and you can fit it with ordinary least squares on $(\log x, \log L)$. With $E$ in the model the fit becomes nonlinear (use `scipy.optimize.curve_fit`), and $E$ is usually the parameter the data pins down worst, because it only shows itself in the bend at the largest scales.

A useful way to feel an exponent: multiplying scale by 10 divides the reducible loss by $10^{\alpha}$. Halving the reducible loss needs $2^{1/\alpha}$ times more scale.

<div class="ex">
<div class="lab">Worked example</div>

**reading an exponent off two runs**

Using the published size term $406.4 / N^{0.34}$: at $N = 10^8$ it is $406.4 / 10^{8 \times 0.34} = 406.4 / 10^{2.72} = 406.4 / 524.8 = 0.774$ nats.

At $N = 10^9$: $406.4 / 10^{3.06} = 406.4 / 1148 = 0.354$ nats.

Ratio: $0.774 / 0.354 = 2.19$. A tenfold bigger model cuts the reducible loss by a factor of 2.19, and $\log_{10} 2.19 = 0.34$, which recovers $\alpha$.

To halve the reducible loss you need $2^{1/0.34} = 7.7$ times the parameters. With Kaplan's compute exponent of about 0.05, halving would need $2^{20}$, a million times the compute.

</div>

## Step 2: two variables at once

Size alone is not the whole story. A huge model trained on a few tokens is starved of data, and a tiny model trained on endless tokens runs out of capacity. Chinchilla's third approach models both limits as separate power-law penalties added to the floor:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}.$$

- $N$ is the parameter count, $D$ the number of training tokens.
- $A / N^{\alpha}$ is the loss you pay for a finite model, $B / D^{\beta}$ the loss you pay for finite data.
- The published values are $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$.

Every figure in this lesson uses this fit as a simulator in place of real training runs. That is a convenient stand-in, not ground truth: a 2024 replication argued that the published numbers do not fit the paper's own data well and disagree with its other two methods (see Going deeper). Use it for the shape, not for the third decimal.

Now add the budget. Training compute is $C \approx 6ND$, so for a fixed $C$ choosing $N$ also fixes $D = C / (6N)$. A bigger model means fewer tokens. The question "what is the best model for this budget?" becomes a one-dimensional search along that trade-off.

## Step 3: the IsoFLOP sweep

An **IsoFLOP curve** ("iso" means equal) is the loss of several models trained with the same compute, plotted against model size. Small models on the left see many tokens but lack capacity; big models on the right have capacity but see too few tokens. In between sits a minimum.

~~~dg-fig 13.05-isoflop.svg
Five IsoFLOP curves, one per budget from 1e19 to 1e23 FLOPs. Seven simulated runs per budget, a parabola in log N through them, and a star at the parabola's minimum: the best size for that budget. The stars move right as the budget grows.
~~~

The recipe, which you can run at small scale:

~~~dg-flow An IsoFLOP sweep, from budgets to an allocation exponent.
Pick budgets C1..Ck -> Train 5 to 9 sizes per budget, D = C / 6N -> Fit a parabola in log N per budget -> *Minimum N_opt(C) -> Regress log N_opt on log C -(slope a)-> N_opt proportional to C^a
~~~

1. Choose 4 to 6 budgets spaced by a factor of about 10.
2. At each budget, train 5 to 9 sizes spread across roughly a factor of 100 around your guess of the optimum, each with $D = C / 6N$ tokens.
3. Set each run's learning-rate schedule to end exactly at its own token count. Otherwise a run that stops partway through its cosine decay looks worse than it is, and the comparison is biased.
4. Fit a parabola $L \approx c_2 (\log N)^2 + c_1 \log N + c_0$ to each budget's points. Its minimum is at $\log N_{\text{opt}} = -c_1 / (2 c_2)$.
5. Fit a straight line to $\log N_{\text{opt}}$ against $\log C$. Its slope $a$ is the **allocation exponent**: $N_{\text{opt}} \propto C^{a}$. Since $D = C / 6N$, tokens then grow as $D_{\text{opt}} \propto C^{1-a}$.

~~~dg-fig 13.05-frontier.svg
Left: the stars from the IsoFLOP plot sit on a line of slope 0.45, much closer to 0.5 than to Kaplan's 0.73. Right: tokens per parameter at the optimum under this fit, which drifts upward because the slope is a little below 0.5.
~~~

On our simulated runs the five minima fall on a line of slope 0.45. The two numbers that matter: an exponent of 0.73 says "spend most new compute on parameters", an exponent of 0.5 says "split it equally".

## Step 4: Kaplan versus Chinchilla

**Kaplan et al. (2020)** at OpenAI fitted loss against non-embedding parameters, dataset size and compute over several orders of magnitude. They found exponents of about 0.076 for parameters and 0.095 for data, and two findings that still hold: model shape (depth versus width) matters little at a fixed parameter count, and loss is a smooth, predictable function of scale. Their allocation advice did not hold up: they concluded $N_{\text{opt}} \propto C^{0.73}$, so most of a growing budget goes into a bigger model trained well short of convergence. GPT-3 (175B parameters on about 300B tokens) reflects that regime.

**Hoffmann et al. (2022)** at DeepMind re-ran the question with over 400 models and three independent methods (fixed-size training curves, IsoFLOP sweeps, and the parametric fit of step 2). All three put both exponents near 0.5: parameters and tokens should grow together. One cause of the disagreement, the authors argued, is that Kaplan's runs used one learning-rate schedule length for all runs, so smaller, shorter runs were judged partway through their decay. Later analyses add that counting only non-embedding parameters and tuning hyperparameters at a single small scale also tilted the result.

The proof was a model. **Chinchilla**, 70B parameters on 1.4T tokens, beat **Gopher**, 280B parameters on 300B tokens, at a similar budget ($6 \times 70 \times 10^9 \times 1.4 \times 10^{12} = 5.9 \times 10^{23}$ FLOPs versus $5.0 \times 10^{23}$). The rule of thumb that came out of it is about **20 tokens per parameter**. With $D = 20N$, the budget is $C = 6N \times 20N = 120N^2$, so $N_{\text{opt}} = \sqrt{C / 120}$.

<div class="ex">
<div class="lab">Worked example</div>

**sizing a 10^24 FLOP run**

At 20 tokens per parameter, $C = 120 N^2$, so $N = \sqrt{10^{24} / 120} = \sqrt{8.33 \times 10^{21}} = 9.1 \times 10^{10}$, about 91B parameters.

Tokens: $D = 20 N = 1.83 \times 10^{12}$, about 1.83T. Check: $6 \times 9.1 \times 10^{10} \times 1.83 \times 10^{12} = 1.0 \times 10^{24}$.

Now the flatness. At Chinchilla's budget of $5.88 \times 10^{23}$ FLOPs the parametric fit predicts a loss of 1.937 for 70B on 1.4T tokens.

An 8B model with the same budget gets $5.88 \times 10^{23} / (6 \times 8 \times 10^9) = 1.225 \times 10^{13}$, about 12.25T tokens, and the fit predicts 1.953: only $0.016 / 1.937 = 0.8$ percent worse.

Generating a token costs about $2N$ FLOPs: 16 GFLOPs for the 8B model against 140 GFLOPs for the 70B, so the small model is $140 / 16 = 8.75$ times cheaper to serve.

Taken literally, this fit's own optimum at that budget is near 32B (loss 1.930), not 70B: one visible symptom of the disagreement between the paper's methods.

</div>

## Step 5: flat valleys and over-training

The IsoFLOP curve is shallow near its bottom. Moving a factor of 2 to 4 away from the optimum in either direction costs very little loss.

~~~dg-fig 13.05-flat.svg
One IsoFLOP curve at Chinchilla's budget. The 8B, 32B and 70B models all land within 0.025 nats of each other; only going far too big (280B on 0.35T tokens) costs real loss.
~~~

"Compute-optimal" means optimal for training cost alone. But a model is trained once and then served for a very large number of tokens. Lifetime cost is roughly

$$C_{\text{life}} \approx 6 N D + 2 N D_{\text{serve}},$$

where $D_{\text{serve}}$ is the number of tokens generated or read in deployment. The serving term grows with $N$ and not with $D$, so when $D_{\text{serve}}$ is large the cheapest way to reach a target quality is a smaller model trained on many more tokens. This is called **over-training** (training past the compute-optimal token count).

~~~dg-fig 13.05-lifetime.svg
Lifetime FLOPs to reach the Chinchilla-70B loss of 1.937, as a function of model size. Training only: cheapest near 30B. Serving 1e13 tokens: near 16B on 6T tokens. Serving 1e14 tokens: near 9B on 20T tokens.
~~~

Reading the figure: with no serving the cheapest size is about 30B (on 2.7T tokens). At $10^{13}$ served tokens it moves to about 16B on 6T tokens, and at $10^{14}$ to about 9B on 20T tokens. Real releases follow this logic: Llama 3 8B was trained on over 15T tokens, roughly 1,900 tokens per parameter and almost 100 times the Chinchilla ratio.

## Scaling laws in code

Here `run` stands in for a training job by drawing from the published fit plus noise.

```python
import numpy as np
from scipy.optimize import curve_fit

# Stand-in for real training runs: the published Chinchilla fit plus a little noise.
E, A, B, alpha, beta = 1.69, 406.4, 410.7, 0.34, 0.28
def run(N, D, rng):                       # "train" a model, return final loss
    return E + A / N**alpha + B / D**beta + rng.normal(0, 0.004)

rng = np.random.default_rng(0)
budgets = [1e19, 1e20, 1e21, 1e22]
best = []
for C in budgets:
    Ns = 0.3 * np.sqrt(C / 6) * np.logspace(-1.2, 1.2, 7)   # 7 sizes around a guess
    losses = [run(N, C / (6 * N), rng) for N in Ns]          # tokens D = C / 6N
    a2, a1, a0 = np.polyfit(np.log10(Ns), losses, 2)          # parabola in log N
    best.append(10 ** (-a1 / (2 * a2)))                       # its minimum
slope, icpt = np.polyfit(np.log10(budgets), np.log10(best), 1)
print(np.round(np.array(best) / 1e9, 2))   # [0.24 0.68 1.84 5.43]  best sizes, billions
print(round(slope, 3))                     # 0.451  allocation exponent

# Predict the best size for a 100x larger budget and check against the "truth".
C_big = 1e24
N_pred = 10 ** (icpt + slope * np.log10(C_big))
Ng = np.logspace(9, 12, 4000)
N_true = Ng[np.argmin(E + A / Ng**alpha + B / (C_big / (6 * Ng))**beta)]
print(f"{N_pred/1e9:.1f}B predicted vs {N_true/1e9:.1f}B true")   # 42.7B vs 41.3B

# Fit a saturating power law L = E + (x0/x)^a to a size sweep with lots of data.
Nsw = np.logspace(7, 10, 8)
Lsw = np.array([run(N, 1e14, rng) for N in Nsw])
f = lambda x, E_, lx0, a_: E_ + np.exp(a_ * (lx0 - np.log(x)))   # x0 fitted in log space
p, cov = curve_fit(f, Nsw, Lsw, p0=[1.5, np.log(1e6), 0.3], maxfev=20000)
print(np.round(p, 3), np.round(np.sqrt(np.diag(cov)), 3))
# [ 1.744 17.651  0.342] [0.003 0.011 0.002]
```

The sweep recovers an exponent of 0.451, matching the value the formula predicts, $\beta / (\alpha + \beta) = 0.28 / 0.62 = 0.452$. A hundredfold extrapolation lands within 4 percent of the true optimum (the noise was small and the law exact). The fitted floor is 1.744, not 1.69: with $10^{14}$ tokens the data term still contributes $410.7 / 10^{14 \times 0.28} \approx 0.049$, which the one-variable fit quietly absorbs into $E$. A "floor" is only the floor of the experiment you ran.

## Checking a scaling-law fit

A fit that looks straight on a log-log plot can still mislead. Before trusting a prediction:

- **Hold out the top.** Fit on all but the largest one or two runs and predict them. If the error is larger than the gain you are trying to predict, the fit is not ready.
- **Report uncertainty.** Bootstrap over runs (refit on resampled sets) and report a range for the exponent and the predicted optimum, not one number.
- **Look at residuals in log space.** A systematic curve means the form is wrong, often a missing floor.
- **Check the IsoFLOP points bracket the minimum.** If the lowest loss is at the edge of the size grid, the parabola's minimum is an extrapolation; add sizes on that side.

The extrapolation figure shows why the floor matters most.

~~~dg-fig 13.05-extrap.svg
Eight small runs across two decades. A pure power law (no floor) fits them almost perfectly and then predicts a loss of 1.03 at 1000x the scale, against a true 2.13. A fit with a floor estimates E = 1.93 and lands at 2.07.
~~~

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Fitting without a floor | predicted loss keeps falling; big runs come out much worse than forecast | fit $E$ explicitly; check the bend at the top of your range
| One schedule length for all runs | small models look worse than they are, optimum shifts toward bigger models | end each run's learning-rate decay at its own token count
| Hyperparameters tuned only at small scale | large runs underperform the curve | fit laws for learning rate and batch size too, or use a parameterisation such as muP that makes them transfer
| Too narrow a fit range | exponent has wide error bars, extrapolation drifts | span at least 2 to 3 orders of magnitude; hold out the largest runs
| Assuming fresh data forever | over-trained small models plateau | count unique tokens; a few epochs of repetition cost little, many epochs give diminishing returns
| Predicting benchmark accuracy directly | accuracy sits near chance, then jumps | predict loss first, then map loss to the task with a separate fit
| Changing the data mixture or tokenizer midway | curves from old runs no longer line up | refit; loss values are only comparable with the same data and tokenizer

One more limit: beyond the **critical batch size**, which grows as the loss falls, extra data parallelism stops saving optimisation steps, which caps how fast a budget can be spent on a large cluster (13.02).

## Going deeper

**The optimal split, derived.** Minimise $A N^{-\alpha} + B D^{-\beta}$ subject to $D = C / 6N$. Substituting gives $A N^{-\alpha} + B (C/6)^{-\beta} N^{\beta}$. Setting the derivative in $N$ to zero:

$$\alpha A N^{-\alpha - 1} = \beta B \left(\tfrac{C}{6}\right)^{-\beta} N^{\beta - 1} \quad\Rightarrow\quad N_{\text{opt}} = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}} \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}.$$

So the allocation exponent is $a = \beta / (\alpha + \beta)$ and tokens grow as $C^{\alpha / (\alpha + \beta)}$. With the published values, $a = 0.45$ and the token exponent is 0.55. Equal exponents $\alpha = \beta$ give exactly $a = 0.5$, which is what methods 1 and 2 found.

**Why tokens per parameter drifts.** If $a$ is exactly 0.5, $D/N$ is constant across budgets. With $a = 0.45$, $D/N \propto C^{0.1}$ and drifts upward, from about 30 at $10^{19}$ FLOPs to 77 at $10^{23}$ in our simulation. "20 tokens per parameter" is best read as a value near Chinchilla's own budgets under methods 1 and 2, not a law of nature. Besiroglu et al. (2024) reconstructed the approach-3 data from the paper's plots, found the published fit inconsistent with it (and its confidence intervals implausibly narrow), and obtained a refit compatible with the other two methods.

**Why flatness follows.** Near the optimum an IsoFLOP curve is close to a parabola in $\log N$ whose curvature is set by $\alpha$ and $\beta$; small exponents give a wide, shallow valley.

**Where the power law comes from.** No derivation is settled. One argument treats language as a long tail of skills or facts with power-law frequencies, so each extra unit of scale learns the next rarest ones: smooth in aggregate loss, abrupt-looking on single tasks. Treat every fitted law as a local regularity, valid for one recipe and only as reliable as its held-out check.

## Key terms

| Term | Meaning
| --- | --- |
| Scaling law | A fitted curve, usually a power law plus a floor, linking loss to parameters, data or compute
| Irreducible loss $E$ | The floor no amount of scale removes; roughly the text's own unpredictability
| Scaling exponent | The slope of reducible loss in log-log space; 10x scale divides it by $10^{\alpha}$
| Compute budget $C$ | Training FLOPs, about $6ND$ for $N$ parameters and $D$ tokens
| IsoFLOP curve | Loss of several model sizes trained with the same compute, against size
| Allocation exponent | $a$ in $N_{\text{opt}} \propto C^{a}$; about 0.73 for Kaplan, about 0.5 for Chinchilla
| Compute-optimal | The size and token count that minimise loss for a training budget, ignoring serving
| Over-training | Training a smaller model on more tokens than compute-optimal to cut inference cost

<div class="take">
<div class="lab">Key takeaway</div>

loss falls as a power law toward a floor, so small, well-tuned runs can predict big ones; split a training budget roughly equally between parameters and tokens (about 20 tokens per parameter) when only training cost matters, and over-train a smaller model when serving dominates, always checking the fit on held-out large runs before trusting it.

</div>
