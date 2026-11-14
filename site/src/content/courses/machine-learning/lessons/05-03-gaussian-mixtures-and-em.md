---
id: "05.03"
course: "machine-learning"
module: "05"
slug: "gaussian-mixtures-and-em"
title: "Gaussian mixtures and EM"
keyIdeas: ["A Gaussian mixture models the data density as $p(x) = \\sum_k \\pi_k \\mathcal N(x \\mid \\mu_k, \\Sigma_k)$: each point comes from a hidden component $z$, which turns clustering into maximum-likelihood estimation.", "Clustering is soft: the responsibility $r_{ik} = P(z_i = k \\mid x_i)$ is a posterior probability, so points between clusters get split membership.", "EM alternates an E-step (compute responsibilities with current parameters) and an M-step (weighted means, covariances and weights). Each iteration never decreases the log-likelihood.", "EM finds local optima and can diverge to infinite likelihood when a component collapses onto one point; use k-means initialisation, restarts and a small covariance regulariser.", "Covariance type (full, diag, tied, spherical) sets the shape and parameter count; choose components and type with BIC. k-means is the limit of a spherical GMM with variance going to zero."]
question: "You fit a 6-component, full-covariance GMM to 300 points in 40 dimensions. The log-likelihood is enormous and keeps increasing, and one component has a covariance with near-zero determinant and weight 1/300. What happened, and what would you change?"
answer: "One component has collapsed onto a single point: its mean equals that point and its covariance is shrinking towards singular, so the density at that point, and with it the log-likelihood, goes to infinity. This is the known degeneracy of maximum likelihood for mixtures, and it is made much more likely here because each full covariance in 40 dimensions has 820 parameters, estimated with an average of only 50 points per component. Fixes: switch to diagonal or tied covariances (or reduce dimension with PCA first) so each component has far fewer parameters; increase `reg_covar` so every covariance has a floor; initialise from k-means and use several restarts; and select the number of components and covariance type by BIC, which will penalise the 6 × 820 covariance parameters heavily. A Bayesian GMM with a prior on covariances is another option."
sources: [{"title": "scikit-learn User Guide: Gaussian mixture models", "url": "https://scikit-learn.org/stable/modules/mixture.html", "note": "EM, covariance types, BIC selection and the variational Bayesian GMM."}, {"title": "Stanford CS229 notes: Mixtures of Gaussians and the EM algorithm", "url": "https://cs229.stanford.edu/notes2020spring/cs229-notes8.pdf", "note": "Derives EM from Jensen's inequality and the ELBO."}, {"title": "Python Data Science Handbook: In Depth: Gaussian Mixture Models", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html", "note": "Free chapter comparing GMM with k-means visually, including density estimation."}, {"title": "Probabilistic Machine Learning: An Introduction (Murphy)", "url": "https://probml.github.io/pml-book/book1.html", "note": "Free draft; the mixture model and EM sections give the general latent-variable view."}]
practice: ["gaussian-mixture-models-em", "maximum-likelihood-estimation", "kmeans-step"]
tags: ["Gaussian mixture", "EM algorithm"]
tools: ["NumPy", "SciPy", "scikit-learn"]
---

## Clustering as a density model

k-means (lesson 05.01) gives each point exactly one cluster and treats every cluster as a ball of the same size. A Gaussian mixture model (GMM) generalises both choices by writing down a generative story:

1. Pick a component $z \in \{1, \dots, K\}$ with probability $\pi_k$.
2. Draw $x$ from $\mathcal N(\mu_z, \Sigma_z)$.

The density of $x$ is then $p(x) = \sum_k \pi_k\, \mathcal N(x \mid \mu_k, \Sigma_k)$. Fitting means maximising the log-likelihood

$$\ell(\theta) = \sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k\, \mathcal N(x_i \mid \mu_k, \Sigma_k)$$

Having a likelihood brings three things k-means lacks: elliptical clusters of different sizes (through $\Sigma_k$), soft assignments, and a density $p(x)$ you can use to score new points, which lesson 05.05 uses for anomaly detection. Maximum likelihood for a single Gaussian is covered in ML Maths (Inference); the difficulty here is the sum inside the log, which couples all parameters and has no closed-form solution.

## Soft assignments

If we knew the parameters, Bayes' rule would give the posterior that point $i$ came from component $k$:

$$r_{ik} = \frac{\pi_k\, \mathcal N(x_i \mid \mu_k, \Sigma_k)}{\sum_j \pi_j\, \mathcal N(x_i \mid \mu_j, \Sigma_j)}$$

These responsibilities sum to 1 over $k$. A point deep inside a cluster gets $r \approx 1$; a point halfway between two equal clusters gets about 0.5 each.

<div class="ex">
<div class="lab">Worked example</div>

**responsibilities in one dimension**

Two components with equal weights, means 0 and 4, both with σ = 1.

At x = 2 (halfway) the two densities are equal, so r = (0.5, 0.5).

At x = 1 the unnormalised densities are proportional to e^−0.5 and e^−4.5. So r₁ = 1 / (1 + e^−4) = 0.982: mostly component 1, but not entirely.

If component 2 instead had σ = 3, its wider tail would claim a larger share of points far to the left of 0. Shape matters, not just distance to the mean.

</div>

## The EM algorithm

Expectation-maximisation breaks the problem into two easy steps by pretending the responsibilities are known:

1. **E-step**: compute $r_{ik}$ from the current parameters.
2. **M-step**: re-estimate every component from all points, each weighted by its responsibility (formulas below).
3. Repeat until the log-likelihood improves by less than a tolerance.

With $N_k = \sum_i r_{ik}$, the effective number of points in component $k$, the M-step is

$$\pi_k = \frac{N_k}{n}, \quad \mu_k = \frac{1}{N_k}\sum_i r_{ik}\, x_i, \quad \Sigma_k = \frac{1}{N_k}\sum_i r_{ik}\,(x_i - \mu_k)(x_i - \mu_k)^\top$$

That is just weighted maximum likelihood for each Gaussian, with weights $r_{ik}$.

~~~dg-seq One EM iteration. The E-step makes a lower bound on the log-likelihood that touches it at the current parameters; the M-step maximises that bound.
Parameters > E-step : current π, μ, Σ
E-step > M-step : responsibilities r (n × K)
M-step > Parameters : weighted π, μ, Σ
Parameters -> Parameters : log-likelihood can only go up
~~~

Why it works: for any distribution $q$ over $z_i$, Jensen's inequality gives $\log p(x_i) \ge \mathbb E_q[\log p(x_i, z_i)] - \mathbb E_q[\log q(z_i)]$, the evidence lower bound (ELBO). The E-step sets $q$ to the true posterior, which makes the bound tight. The M-step raises the bound, so the likelihood rises too. The same bound reappears in variational autoencoders (Deep Learning, generative models), where $q$ is learned by a network instead of computed exactly.

Compute the E-step in log space. In 50 dimensions, Gaussian densities are easily $10^{-300}$, so multiply-then-normalise underflows. Use log-densities and log-sum-exp:

```python
import numpy as np
from scipy.special import logsumexp
def e_step(log_pi, log_dens):               # log_dens[i, k] = log N(x_i | mu_k, Sigma_k)
    log_r = log_pi + log_dens
    ll = logsumexp(log_r, axis=1)            # per-point log-likelihood
    return np.exp(log_r - ll[:, None]), ll.sum()
```

## Failure modes and remedies

- **Local optima**: the likelihood is multimodal. Initialise from k-means (scikit-learn's default) and use several restarts (`n_init`).
- **Singularities**: if a component's mean sits on one point and its variance shrinks to zero, that point's density goes to infinity, and so does the likelihood. This is a degenerate maximum, not a good fit. Adding a small $\epsilon I$ to every covariance (`reg_covar`, default $10^{-6}$) prevents it.
- **Slow convergence**: EM can crawl when components overlap heavily; cap iterations and monitor the log-likelihood.
- **Wrong $K$**: likelihood always rises with more components, so compare models with BIC, $-2\ell + p \log n$, which penalises the parameter count $p$.

## Covariance types, and k-means as a special case

The covariance structure controls both cluster shape and the number of parameters. For $d = 10$ and $K = 5$ (counting means, covariances and $K-1$ free weights):

| Type | Shape | Parameters |
| --- | --- | --- |
| full | any ellipse per component | 5 × (10 + 55) + 4 = 329 |
| tied | one shared ellipse | 5 × 10 + 55 + 4 = 109 |
| diag | axis-aligned ellipses | 5 × (10 + 10) + 4 = 104 |
| spherical | balls of different sizes | 5 × (10 + 1) + 4 = 59 |

Full covariance grows as $d^2$, so with high $d$ or small $n$, diag or tied are safer.

Now take a spherical GMM with every $\Sigma_k = \sigma^2 I$ and equal weights, and let $\sigma \to 0$. The responsibilities become 1 for the nearest mean and 0 for the rest, and the M-step becomes "mean of assigned points". That is exactly Lloyd's algorithm, which is why k-means struggles with exactly the shapes a full-covariance GMM handles.

<div class="take">
<div class="lab">Key takeaway</div>

a GMM is a probabilistic, soft, elliptical generalisation of k-means, fitted by EM, which alternates responsibilities and weighted maximum-likelihood updates; guard against local optima and collapsing components, and choose $K$ and covariance type with BIC.

</div>
