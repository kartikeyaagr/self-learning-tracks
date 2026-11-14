---
id: "13.05"
course: "ai-research-track-web"
module: "13"
slug: "scaling-laws"
title: "Scaling laws"
keyIdeas: ["Pretraining loss falls as a power law in model size, data and compute, $L(x) \\approx E + (x_0/x)^{\\alpha}$, over many orders of magnitude. That regularity is what lets small runs predict big ones.", "Kaplan et al. (2020) concluded that extra compute should go mostly into a bigger model ($N \\propto C^{0.73}$). Hoffmann et al. (2022, Chinchilla) found parameters and tokens should grow about equally ($N, D \\propto C^{0.5}$), which works out to roughly 20 tokens per parameter.", "An IsoFLOP sweep trains several model sizes at each fixed compute budget, finds the loss-minimising size per budget, and fits a line to those optima in log-log space. Its slope is the allocation exponent.", "Near the optimum the loss is flat along an IsoFLOP curve, so a model several times smaller than compute-optimal costs only a little loss. Since inference costs about $2N$ FLOPs per token, deployed models are deliberately over-trained.", "Extrapolate with care: fits are only as good as the range they span, they assume the learning-rate schedule and hyperparameters are tuned at every scale, and loss predictions do not transfer directly to downstream accuracy."]
question: "You have a training budget of $2 \\times 10^{23}$ FLOPs and expect to serve $10^{13}$ tokens over the model's lifetime. Compute the Chinchilla-style compute-optimal size and token count at 20 tokens per parameter, then the tokens a 7B model would see with the same budget. Compare lifetime FLOPs (training plus about $2N$ per served token). What would you need to check before committing to the 7B option?"
answer: "Compute-optimal: $N = \\sqrt{2 \\times 10^{23}/120} = 40.8$B parameters, $D = 20N = 816$B tokens. A 7B model with the same budget sees $2 \\times 10^{23}/(6 \\times 7 \\times 10^9) = 4.76$T tokens, about 680 tokens per parameter. Serving cost: $2 \\times 40.8\\text{B} \\times 10^{13} = 8.2 \\times 10^{23}$ FLOPs versus $2 \\times 7\\text{B} \\times 10^{13} = 1.4 \\times 10^{23}$. Lifetime totals: about $1.02 \\times 10^{24}$ for the 41B model versus $3.4 \\times 10^{23}$ for the 7B, three times cheaper overall, because serving dominates. The catch is quality: the 7B model is off the compute-optimal point, so at equal training compute its loss is somewhat higher (the approach-3 fit predicts 1.988 versus 1.980, under half a percent, but do not trust the fit's third digit). Before committing, check that the extra tokens exist at the same quality (4.76T unique tokens, or acceptable repetition), validate with your own IsoFLOP or over-training points at smaller scale that the loss gap is as small as predicted, and check downstream metrics rather than loss alone. If the gap matters, spending part of the inference savings on more training tokens for the 7B model is usually the better trade."
sources: [{"title": "Hoffmann et al.: Training Compute-Optimal Large Language Models (2022)", "url": "https://arxiv.org/abs/2203.15556", "note": "Section 3: the three estimation approaches, the IsoFLOP figures and the Chinchilla versus Gopher comparison."}, {"title": "Kaplan et al.: Scaling Laws for Neural Language Models (2020)", "url": "https://arxiv.org/abs/2001.08361", "note": "The original power-law fits for N, D and C and the critical batch size discussion."}, {"title": "How to Scale Your Model (JAX scaling book)", "url": "https://jax-ml.github.io/scaling-book/", "note": "How compute budgets translate into hardware time and model shape choices."}]
practice: ["isoflop-sweep-exponent", "chinchilla-compute-optimal", "saturating-power-law-fit", "fixed-ratio-budget-split", "fit-learning-rate-and-batch-size-scaling"]
tags: ["scaling laws", "Chinchilla", "compute-optimal"]
tools: ["NumPy", "SciPy"]
---

## What a scaling law is

Train a family of models that differ only in size, data or compute, plot final loss against that quantity on log-log axes, and you get something very close to a straight line, bending only as it approaches an irreducible floor. The working form is

$$L(x) = E + \left(\frac{x_0}{x}\right)^{\alpha},$$

where $E$ is the loss no amount of $x$ removes (the entropy of the text, roughly) and $\alpha$ is the exponent. Without $E$ the fit is a line in log-log space, $\log L = \alpha \log x_0 - \alpha \log x$, fitted by least squares on the logs. With $E$ you fit nonlinearly, and $E$ is usually the least well-determined parameter. The practical value is planning: fit on runs costing a thousandth of the target, then predict the target's loss and pick its shape.

## Kaplan et al.: grow the model

The 2020 OpenAI study fitted loss against non-embedding parameter count, dataset size and compute across several orders of magnitude, with exponents of about 0.076 for parameters and 0.095 for data. Two findings held up: model shape (depth versus width) matters little at a fixed parameter count, and loss is a smooth, predictable function of scale. Their compute-optimal recommendation did not hold up: they concluded that most of a growing budget should go into parameters ($N_{\text{opt}} \propto C^{0.73}$), with comparatively few tokens, training large models well short of convergence. GPT-3 (175B parameters on about 300B tokens) reflects that regime.

## Chinchilla: grow both

Hoffmann et al. re-ran the question with over 400 models and three independent methods, and all three found that parameters and tokens should scale in roughly equal proportion, with exponents near 0.5 for each. One reason for the disagreement, the authors argued, is that Kaplan's runs used a fixed learning-rate schedule length, so smaller models trained on fewer tokens were not evaluated at the end of a matched schedule; later analyses also point to counting only non-embedding parameters and to hyperparameters tuned at small scale. The headline result: Chinchilla, 70B parameters on 1.4T tokens, outperformed Gopher, 280B parameters on 300B tokens, at a similar compute budget ($5.9 \times 10^{23}$ versus $5.0 \times 10^{23}$ FLOPs by $6ND$). The rule of thumb that falls out is about 20 tokens per parameter, so $C = 6ND = 120N^2$ at the optimum.

## IsoFLOP sweeps

Their most transparent method is the IsoFLOP sweep, which you can reproduce at small scale:

~~~dg-flow An IsoFLOP sweep, from budgets to an allocation exponent.
Pick budgets C1..Ck -> Train 5 to 9 sizes per budget, D = C / 6N -> Fit a parabola in log N per budget -> *Minimum N_opt(C) -> Regress log N_opt on log C -(slope a)-> N_opt proportional to C^a
~~~

Each learning-rate schedule must be set to end at that run's token count, or the comparison is biased toward whichever size happened to finish its decay. Approach 3 of the paper fits a joint form $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$ to all runs; the published values are $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$. A later replication argued that the reported fit was imprecise and that it disagrees with the other two methods, so use it for intuition about shape rather than exact optima.

## Extrapolation and its traps

- Fit range: an exponent fitted over two orders of magnitude extrapolated four further compounds small errors; check by holding out the largest runs and predicting them.
- Schedules and hyperparameters: the law describes well-tuned models. The best learning rate falls and the best batch size grows with scale, so they need their own fitted laws (or a parameterisation like muP that makes them transfer).
- Batch size: beyond a critical batch size, which grows as the loss falls, more data parallelism stops saving optimisation steps, which limits how fast a budget can be spent.
- Data: the laws assume fresh, same-quality tokens. Repeating data a few times costs little, but many epochs give diminishing returns, and a mixture change moves the whole curve.
- Downstream metrics: loss extrapolates smoothly, while accuracy on a specific benchmark can sit near chance and then rise sharply, so predict loss, then map it to tasks with a separate fit.

## Over-training for inference

Compute-optimal means optimal for training cost alone. A model is trained once and then served for a very large number of tokens, each costing about $2N$ FLOPs. Lifetime cost is roughly $6ND + 2N D_{\text{serve}}$, and when $D_{\text{serve}}$ is large the cheapest model for a target quality is smaller than Chinchilla-optimal and trained on far more tokens. Because the IsoFLOP curve is flat near its minimum, the training-side penalty is small. Llama 3 8B, trained on over 15T tokens, is about 1,900 tokens per parameter, almost 100 times the Chinchilla ratio.

<div class="ex">
<div class="lab">Worked example</div>

**sizing a 10^24 FLOP run**

Compute-optimal at 20 tokens per parameter: $C = 120N^2$, so $N = \sqrt{10^{24}/120} = 91$B parameters and $D = 20N = 1.83$T tokens.

Flatness, using the approach-3 fit at Chinchilla's budget of $5.88 \times 10^{23}$ FLOPs: 70B on 1.4T tokens predicts a loss of 1.937; an 8B model on $5.88 \times 10^{23}/(6 \times 8\text{B}) = 12.25$T tokens predicts 1.953, only 0.9 percent worse.

That 8B model costs 16 GFLOPs per generated token against 140 for the 70B, 8.75 times cheaper to serve. (Taken literally the fit's own optimum here is near 33B, not 70B, one symptom of the disagreement between methods.)

</div>

<div class="take">
<div class="lab">Key takeaway</div>

fit power laws on small, properly tuned runs; allocate compute roughly equally to parameters and tokens (about 20 tokens per parameter) when training cost is all that matters, and over-train smaller models when inference dominates.

</div>
