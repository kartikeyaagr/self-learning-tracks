---
id: "05.03"
course: "machine-learning"
module: "05"
slug: "gaussian-mixtures-and-em"
title: "Gaussian mixtures and EM"
keyIdeas: ["A Gaussian mixture models the data density as $p(x) = \\sum_k \\pi_k \\mathcal N(x \\mid \\mu_k, \\Sigma_k)$: each point comes from a hidden component $z$, which turns clustering into maximum-likelihood estimation.", "Clustering is soft: the responsibility $r_{ik} = P(z_i = k \\mid x_i)$ is a posterior probability from Bayes' rule, so points between clusters get split membership, and a wide component can win points that are closer to a narrow one.", "EM alternates an E-step (compute responsibilities with current parameters) and an M-step (weighted means, covariances and weights). Each iteration never decreases the log-likelihood, because the E-step makes the ELBO tight and the M-step raises it.", "EM finds local optima and can diverge to infinite likelihood when a component collapses onto one point; use k-means initialisation, several restarts (`n_init`) and a small covariance floor (`reg_covar`).", "Covariance type (full, diag, tied, spherical) sets the shape and parameter count; choose components and type with BIC. k-means is the limit of a spherical GMM with variance going to zero."]
question: "You fit a 6-component, full-covariance GMM to 300 points in 40 dimensions. The log-likelihood is enormous and keeps increasing, and one component has a covariance with near-zero determinant and weight 1/300. What happened, and what would you change?"
answer: "One component has collapsed onto a single point: its mean equals that point and its covariance is shrinking towards singular, so the density at that point, and with it the log-likelihood, goes to infinity. The weight of 1/300 is the giveaway: the component owns exactly one point. This is the known degeneracy of maximum likelihood for mixtures (the likelihood is unbounded, so its 'maximum' is a spike, not a model), and it is made much more likely here because each full covariance in 40 dimensions has 40 × 41 / 2 = 820 parameters, estimated from an average of only 50 points per component; with fewer points than dimensions a component's sample covariance is singular even without collapse. Fixes: switch to diagonal or tied covariances (or reduce dimension with PCA first) so each component has far fewer parameters; increase `reg_covar` so every covariance has a floor; standardise the features; initialise from k-means and use several restarts (`n_init`), discarding solutions with components of weight near 1/n; and select the number of components and covariance type by BIC, which will penalise the roughly 6 × 820 covariance parameters heavily and will likely prefer far fewer components or a simpler covariance. A Bayesian GMM (`BayesianGaussianMixture`) with a prior on covariances is another option, because the prior keeps covariances away from singular and can switch off unneeded components."
sources: [{"title": "scikit-learn User Guide: Gaussian mixture models", "url": "https://scikit-learn.org/stable/modules/mixture.html", "note": "EM, covariance types, BIC selection and the variational Bayesian GMM."}, {"title": "Stanford CS229 notes: Mixtures of Gaussians and the EM algorithm", "url": "https://cs229.stanford.edu/notes2020spring/cs229-notes8.pdf", "note": "Derives EM from Jensen's inequality and the ELBO."}, {"title": "Python Data Science Handbook: In Depth: Gaussian Mixture Models", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html", "note": "Free chapter comparing GMM with k-means visually, including density estimation."}, {"title": "Probabilistic Machine Learning: An Introduction (Murphy)", "url": "https://probml.github.io/pml-book/book1.html", "note": "Free draft; the mixture model and EM sections give the general latent-variable view."}]
practice: ["gaussian-mixture-models-em", "maximum-likelihood-estimation", "kmeans-step"]
tags: ["Gaussian mixture", "EM algorithm"]
tools: ["NumPy", "SciPy", "scikit-learn"]
---

## What you will learn

- What a Gaussian mixture model (GMM) is: a story of how the data was generated, with a hidden group label behind every point.
- How **soft** cluster membership works: each point gets a probability for every cluster, computed with Bayes' rule.
- How the EM algorithm fits the model by alternating two easy steps, and why each round can only improve the fit.
- How to fit, read and choose a GMM in scikit-learn: covariance types, restarts and BIC.
- The ways EM goes wrong (local optima, collapsing components, too many parameters) and how to guard against each.

*Before you start:* you should know k-means (05.01), the normal (Gaussian) distribution and what a covariance matrix describes, and the idea of maximum likelihood (lesson 01.01, and ML Maths, Inference). Bayes' rule is used once; we spell it out.

## From hard clusters to a density model

In 05.01, k-means gave every point exactly one cluster and treated every cluster as a round ball of about the same size. In 05.02, hierarchical and density-based methods relaxed the shape assumption but still gave hard labels. This lesson takes a different route: instead of a rule for cutting the data, we write down a **probability model** of where the data came from, then fit it. Clusters fall out as a by-product, together with three things the earlier methods cannot give you: elliptical clusters of different sizes, a probability of membership for each point, and a density $p(x)$ that can score how typical a new point is.

## The idea in one picture

A café logs how long each customer stays. Most people grab a coffee and leave; others sit down with a laptop. Nobody records which kind of visit it was, but the histogram of 300 visit lengths has two clear humps.

~~~dg-fig 05.03-mixture.svg
Two weighted bell curves (dashed) add up to the mixture density (thick line) that tracks the histogram. The narrow hump is takeaway visits, the wide one sit-in visits.
~~~

A Gaussian mixture with two components describes this as:

| Component | Weight $\pi$ | Mean $\mu$ (min) | Spread $\sigma$ (min)
| --- | --- | --- | --- |
| takeaway | 0.60 | 6.1 | 1.9
| sit-in | 0.40 | 34.1 | 9.4

Read it as: 60% of visits are takeaway (about 6 minutes, give or take 2) and 40% are sit-in (about 34, give or take 9). Now ask about single visits:

| Visit length | 5 min | 10 min | 12 min | 14 min
| --- | --- | --- | --- | --- |
| P(sit-in) | 0.001 | 0.039 | 0.49 | 0.985

A 12-minute visit is a coin toss. A 14-minute visit is almost surely sit-in, even though 14 is much closer to 6 than to 34, because the takeaway bell is so narrow that 14 minutes is about four of its standard deviations away. Distance alone would get this wrong; shape matters.

The new words:

- A **component** is one of the bell curves: one hidden group.
- Its **mixing weight** $\pi_k$ is the share of data it accounts for. Weights are positive and add up to 1.
- The **latent variable** $z$ is the hidden group label of a point (takeaway or sit-in). We never observe it.
- The **responsibility** $r_{ik}$ is the probability that point $i$ came from component $k$, given its value. It is the "P(sit-in)" in the table.

## Step 1: write down the story

A GMM says each data point was made in two moves:

1. Pick a component $z \in \{1, \dots, K\}$, with probability $\pi_k$ for component $k$.
2. Draw $x$ from that component's Gaussian, $\mathcal N(\mu_z, \Sigma_z)$.

Here $\mu_k$ is the centre of component $k$ and $\Sigma_k$ its **covariance matrix**, which sets the size, stretch and tilt of its ellipse (in one dimension it is just the variance $\sigma_k^2$). Because we do not see $z$, the probability of $x$ adds up every way it could have been produced:

$$p(x) = \sum_{k=1}^{K} \pi_k\, \mathcal N(x \mid \mu_k, \Sigma_k).$$

This is the **mixture density**, the thick curve in the figure. It is a proper probability density: it is positive and integrates to 1 because the weights do.

## Step 2: what "fitting" means

Fitting means choosing all the parameters $\theta = (\pi_k, \mu_k, \Sigma_k)$ to make the observed data as probable as possible, which is maximum likelihood. With $n$ independent points, the **log-likelihood** is

$$\ell(\theta) = \sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k\, \mathcal N(x_i \mid \mu_k, \Sigma_k).$$

For a single Gaussian this is easy: the best mean is the sample average and the best covariance is the sample covariance. The trouble is the **sum inside the log**. It ties all components together, so setting the derivative to zero gives equations where each component's best parameters depend on everyone else's. There is no closed-form answer. But notice what would make it easy: if we knew each point's label $z_i$, we would just fit one Gaussian per group. EM exploits exactly this.

## Step 3: soft assignments with Bayes' rule

Suppose for a moment we did know the parameters. Then Bayes' rule gives the probability that point $i$ came from component $k$:

$$r_{ik} = P(z_i = k \mid x_i) = \frac{\pi_k\, \mathcal N(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j\, \mathcal N(x_i \mid \mu_j, \Sigma_j)}.$$

The numerator is "how common is component $k$" times "how well does it explain $x_i$". The denominator is the same quantity summed over all components, so the responsibilities of each point add up to 1. A point deep inside one cluster gets $r \approx 1$ for it; a point halfway between two equal clusters gets about 0.5 each.

<div class="ex">
<div class="lab">Worked example</div>

**responsibilities in one dimension**

Two components with equal weights 0.5, means 0 and 4, both with σ = 1.

At x = 2 (halfway) the two densities are equal, so r = (0.5, 0.5).

At x = 1, the weights and the normalising constants $1/\sqrt{2\pi}$ are the same for both components and cancel. What is left is $e^{-(1-0)^2/2} = e^{-0.5}$ for component 1 and $e^{-(1-4)^2/2} = e^{-4.5}$ for component 2.

So $r_1 = e^{-0.5} / (e^{-0.5} + e^{-4.5}) = 1 / (1 + e^{-4}) = 1 / 1.0183 = 0.982$: mostly component 1, but not entirely.

If component 2 instead had σ = 3, its wider tail would claim a larger share of points far to the left of 0. Shape matters, not just distance to the mean.

</div>

~~~dg-fig 05.03-resp.svg
Responsibility of component 1 along the line. With equal widths it slides smoothly from 1 to 0 and crosses 0.5 at the midpoint. When component 2 is wider (dashed), its tails win at both ends, even far to the left of component 1.
~~~

The dashed curve is worth a second look: with $\sigma_2 = 3$, a point at $x = -4$ goes to component 2 with probability 0.97, although it is four units from $\mu_1$ and eight from $\mu_2$. Far from everything, the widest component wins. That is the same effect as the 14-minute café visit.

## Step 4: EM, the alternating fix

We now have a chicken-and-egg problem: responsibilities need parameters, and parameters would be easy if we had responsibilities. **Expectation-maximisation (EM)** simply alternates:

1. **E-step** (expectation): with the current parameters, compute $r_{ik}$ for every point and component.
2. **M-step** (maximisation): re-estimate every component from all points, each point weighted by its responsibility.
3. Repeat until the log-likelihood improves by less than a small tolerance.

Write $N_k = \sum_i r_{ik}$ for the **effective number of points** in component $k$ (a soft count, usually not a whole number). The M-step is

$$\pi_k = \frac{N_k}{n}, \quad \mu_k = \frac{1}{N_k}\sum_i r_{ik}\, x_i, \quad \Sigma_k = \frac{1}{N_k}\sum_i r_{ik}\,(x_i - \mu_k)(x_i - \mu_k)^\top.$$

In words: the weight is the component's share of the soft counts; the mean is a weighted average; the covariance is a weighted average of squared deviations. These are exactly the single-Gaussian maximum-likelihood formulas, with each point counted fractionally.

~~~dg-seq One EM iteration. The E-step makes a lower bound on the log-likelihood that touches it at the current parameters; the M-step maximises that bound.
Parameters > E-step : current π, μ, Σ
E-step > M-step : responsibilities r (n × K)
M-step > Parameters : weighted π, μ, Σ
Parameters -> Parameters : log-likelihood can only go up
~~~

Here is what that looks like on 360 points in two dimensions, starting from three round guesses placed in the wrong spots:

~~~dg-fig 05.03-em.svg
EM in action. Ellipses are each component's two-standard-deviation contour; crosses are the means. Within a few iterations the components move, stretch and rotate onto the three groups, and the log-likelihood climbs from −2670 to −1343.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one full EM iteration by hand**

Data: five numbers $x = (1, 2, 4, 7, 9)$, two components. Start with $\mu = (2, 8)$, $\sigma = (2, 2)$, $\pi = (0.5, 0.5)$.

E-step. Weights and widths are equal, so only the exponents $-(x - \mu)^2 / (2 \cdot 2^2)$ matter. For $x = 4$: $-(4-2)^2/8 = -0.5$ and $-(4-8)^2/8 = -2$, so $r_1 = 1/(1 + e^{-1.5}) = 1/1.223 = 0.818$.

Doing the same for every point gives $r_1 = (0.998, 0.989, 0.818, 0.047, 0.002)$ and $r_2 = 1 - r_1 = (0.002, 0.011, 0.182, 0.953, 0.998)$. Point 4 is mostly, but not entirely, in component 1.

Soft counts: $N_1 = 0.998 + 0.989 + 0.818 + 0.047 + 0.002 = 2.854$ and $N_2 = 5 - 2.854 = 2.146$. New weights: $\pi = (2.854/5, 2.146/5) = (0.571, 0.429)$.

New mean 1: $(0.998 \cdot 1 + 0.989 \cdot 2 + 0.818 \cdot 4 + 0.047 \cdot 7 + 0.002 \cdot 9) / 2.854 = 6.600 / 2.854 = 2.31$.

New mean 2: $(0.002 \cdot 1 + 0.011 \cdot 2 + 0.182 \cdot 4 + 0.953 \cdot 7 + 0.998 \cdot 9) / 2.146 = 16.40 / 2.146 = 7.64$.

New spreads (weighted average squared distance to the new means): $\sigma_1^2 = 5.30 / 2.854 = 1.86$, so $\sigma_1 = 1.36$; $\sigma_2^2 = 5.11 / 2.146 = 2.38$, so $\sigma_2 = 1.54$.

Check: the log-likelihood rose from −12.14 to −11.41. Run to convergence and EM settles at $\mu = (2.34, 8.0)$, $\sigma = (1.25, 1.0)$, $\pi = (0.6, 0.4)$, with log-likelihood −11.12.

</div>

## Step 5: why EM never makes things worse

For any guess $q$ of the distribution of a hidden label $z_i$, Jensen's inequality (the log of an average is at least the average of the logs) gives

$$\log p(x_i) \ge \mathbb E_{q}[\log p(x_i, z_i)] - \mathbb E_{q}[\log q(z_i)].$$

The right side is the **evidence lower bound (ELBO)**. The gap between the two sides is the KL divergence between $q$ and the true posterior of $z_i$, which is zero exactly when $q$ equals that posterior. So:

- The E-step sets $q$ to the true posterior (the responsibilities). The bound now touches the log-likelihood at the current parameters.
- The M-step maximises the bound over the parameters. The bound goes up, and since the log-likelihood sits above the bound and equalled it before the step, the log-likelihood goes up at least as much.

So EM's log-likelihood curve never dips. The same bound reappears in variational autoencoders (Deep Learning, generative models), where $q$ is produced by a network instead of computed exactly.

## Gaussian mixtures in code

scikit-learn's `GaussianMixture` runs EM for you, initialised from k-means by default:

```python
import numpy as np
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(0)
visits = np.r_[rng.normal(6, 2, 180), rng.normal(35, 9, 120)]
visits = visits[visits > 0.5].reshape(-1, 1)        # minutes, shape (n, 1)

gmm = GaussianMixture(n_components=2, covariance_type="full",
                      n_init=5, random_state=0).fit(visits)
print(gmm.weights_.round(2))                         # [0.6 0.4]
print(gmm.means_.ravel().round(1))                   # [ 6.1 34.1]
print(np.sqrt(gmm.covariances_.ravel()).round(1))    # [1.9 9.4]
print(gmm.converged_, gmm.n_iter_)                   # True 3

print(gmm.predict_proba([[5.0], [10.0], [12.0], [14.0]]).round(3))
# [[0.999 0.001] [0.961 0.039] [0.508 0.492] [0.015 0.985]]
print(gmm.score_samples([[6.0], [20.0], [90.0]]).round(2))   # log p(x)
# [ -2.08  -5.2  -21.74]   a 90-minute visit is very unusual

for K in [1, 2, 3, 4]:
    bic = GaussianMixture(K, n_init=5, random_state=0).fit(visits).bic(visits)
    print(K, round(bic, 1))   # 1 2486.9 | 2 2044.6 | 3 2050.7 | 4 2062.8
```

`predict` gives the hard label (the largest responsibility), `predict_proba` the soft one, and `score_samples` the log-density, which is what anomaly detection (05.05) uses. BIC is lowest at $K = 2$, matching the two kinds of visit.

Writing EM yourself takes a dozen lines. The one trick that matters is to work in **log space**: in 50 dimensions Gaussian densities are easily $10^{-300}$, so "multiply then normalise" underflows to 0/0. Add log-weights to log-densities and normalise with log-sum-exp:

```python
from scipy.stats import norm
from scipy.special import logsumexp

def em_1d(x, mu, sigma, pi, iters=50, tol=1e-6):
    prev = -np.inf
    for t in range(iters):
        log_p = np.log(pi) + norm.logpdf(x[:, None], mu, sigma)   # (n, K)
        ll_i = logsumexp(log_p, axis=1)                            # log p(x_i)
        r = np.exp(log_p - ll_i[:, None])                          # E-step
        if ll_i.sum() - prev < tol: break
        prev = ll_i.sum()
        Nk = r.sum(0)                                              # M-step
        pi, mu = Nk / len(x), (r * x[:, None]).sum(0) / Nk
        sigma = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / Nk + 1e-6)
    return mu, sigma, pi, prev

x = np.array([1, 2, 4, 7, 9.])
print(em_1d(x, np.array([2., 8.]), np.array([2., 2.]), np.array([.5, .5])))
# mu ≈ [2.34 8.0], sigma ≈ [1.25 1.0], pi ≈ [0.6 0.4], log-lik ≈ -11.12
```

The `+ 1e-6` inside the square root is a tiny variance floor; the next section explains why you want it.

## Reading and checking the result

- **Watch the log-likelihood.** It must never decrease between iterations; if it does, your E-step or M-step has a bug. Check `converged_` in scikit-learn; if it is False, raise `max_iter` or loosen `tol`.
- **Look at the weights.** A component with weight near $1/n$ is usually a collapsed one sitting on a single point (see below), not a real cluster.
- **Look at the responsibilities.** A histogram of the largest responsibility per point tells you how confident the clustering is. Many points near 0.5 means heavily overlapping components; the hard labels for those points mean little.
- **Compare restarts.** If different seeds give different final log-likelihoods, keep the best (`n_init` does this).
- **Choose $K$ and the covariance type with BIC.** The likelihood always rises with more components, so you need a penalty. BIC is $-2\ell + p \log n$, where $p$ is the number of free parameters.

~~~dg-fig 05.03-bic.svg
BIC for 1 to 7 components on the three-group data. Full covariance bottoms out at the true K = 3. The simpler shapes keep needing more components, because they must tile the stretched group with several small blobs.
~~~

The covariance type sets both the cluster shape and the parameter count. For $d = 10$ features and $K = 5$ components (counting means, covariances and $K - 1$ free weights):

| Type | Shape | Parameters
| --- | --- | --- |
| full | any ellipse per component | 5 × (10 + 55) + 4 = 329
| tied | one shared ellipse | 5 × 10 + 55 + 4 = 109
| diag | axis-aligned ellipses | 5 × (10 + 10) + 4 = 104
| spherical | balls of different sizes | 5 × (10 + 1) + 4 = 59

A full covariance has $d(d+1)/2$ numbers, which grows as $d^2$, so with many features or few points, diag or tied are safer.

## What goes wrong

EM is a local hill-climber on a bumpy surface, and maximum likelihood for mixtures has a trap door.

~~~dg-fig 05.03-optima.svg
Left: ten random starts on six blobs. Every curve rises, but they level off at different heights; only one run finds the best fit. Right: a component pinned on one point with a shrinking width sends the log-likelihood to infinity, beating any honest fit.
~~~

| Problem | Symptom | Fix
| --- | --- | --- |
| Local optima | different seeds give different final log-likelihoods and clusters | k-means initialisation (the default), `n_init` of 5 to 10, keep the best
| Collapse (singularity) | log-likelihood huge and growing; a component with weight about $1/n$ and near-zero variance | `reg_covar` (a floor $\epsilon I$ added to each covariance), fewer parameters, Bayesian GMM
| Too many parameters | collapse or unstable ellipses in high $d$ | diag or tied covariance, or PCA first (05.04)
| Wrong $K$ | extra components split real clusters | compare BIC across $K$ and covariance types
| Unscaled features | one feature with a big range dominates the shapes | standardise first (diag/spherical are especially sensitive)
| Non-Gaussian clusters | bananas or rings split into several components | accept several components per cluster, or use DBSCAN (05.02)

The collapse deserves a sentence of explanation. If a component's mean sits exactly on one data point and its variance shrinks towards zero, the density at that point grows without limit, and so does the log-likelihood. The "maximum" is then a degenerate spike, not a better model. The default `reg_covar=1e-6` puts a floor under every variance so the spike cannot form; raise it if you still see collapses.

## Going deeper

**k-means is a special case.** Take a spherical GMM with every $\Sigma_k = \sigma^2 I$ and equal weights, and let $\sigma \to 0$. The responsibilities become 1 for the nearest mean and 0 for the rest, and the M-step becomes "mean of the assigned points". That is exactly Lloyd's algorithm. It also explains k-means' blind spots: it is a GMM forced to use identical round shapes.

~~~dg-fig 05.03-kmeans.svg
Two long parallel bands. k-means, which can only draw a straight boundary between round clusters, cuts across both. A full-covariance GMM fits one stretched ellipse per band and recovers them almost perfectly.
~~~

**Decision boundaries are curved.** Setting two weighted log-densities equal gives a quadratic equation in $x$, so GMM boundaries are conics (ellipses, parabolas, hyperbolas). With a tied covariance the quadratic terms cancel and boundaries become straight lines, the same result as linear discriminant analysis.

**Convergence is only local and linear.** EM reaches a stationary point of the likelihood (usually a local maximum), at a rate set by how much the components overlap: heavy overlap means slow convergence.

**Beyond plain maximum likelihood.** `BayesianGaussianMixture` puts priors on weights and covariances and can switch off components it does not need, which makes the choice of $K$ softer. The same E-step/M-step recipe also fits hidden Markov models and missing-data problems: whenever the likelihood would be easy if a hidden variable were known, try EM.

## Key terms

| Term | Meaning
| --- | --- |
| Gaussian mixture model | A density made of $K$ weighted Gaussian bells, $p(x) = \sum_k \pi_k \mathcal N(x \mid \mu_k, \Sigma_k)$
| Component | One Gaussian in the mixture: one hidden group
| Mixing weight $\pi_k$ | The share of data a component accounts for; weights sum to 1
| Latent variable $z$ | The unobserved group label of a point
| Responsibility $r_{ik}$ | Posterior probability that point $i$ came from component $k$
| EM algorithm | Alternates E-step (responsibilities) and M-step (weighted maximum likelihood) until the log-likelihood stops rising
| ELBO | Lower bound on the log-likelihood that the E-step makes tight and the M-step raises
| Collapse / singularity | A component shrinking onto one point, sending the likelihood to infinity
| BIC | $-2\ell + p \log n$; penalised fit used to choose $K$ and covariance type

<div class="take">
<div class="lab">Key takeaway</div>

a GMM is a probabilistic, soft, elliptical generalisation of k-means, fitted by EM, which alternates responsibilities and weighted maximum-likelihood updates; guard against local optima and collapsing components, and choose $K$ and covariance type with BIC.

</div>
