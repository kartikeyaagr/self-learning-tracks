---
id: "07.02"
course: "machine-learning"
module: "07"
slug: "encoding-categorical-variables"
title: "Encoding categorical variables"
keyIdeas: ["**One-hot** encoding gives each category its own column; it is exact and safe for low-cardinality features, but needs a policy for rare and unseen categories.", "**Ordinal** (integer) encoding is right for truly ordered categories and acceptable for trees, but it imposes a fake order and spacing on linear and distance-based models.", "**Target encoding** replaces a category with a smoothed mean of the target, $\\frac{n_c \\bar y_c + m \\bar y}{n_c + m}$. Computed in-sample it leaks the label, so it must be **cross-fitted**: each training row is encoded by a model that never saw it.", "For very high cardinality, **feature hashing** bounds the dimension at the cost of collisions, and **learned embeddings** let a neural network place similar categories close together.", "Gradient-boosting libraries can split on categories natively; that is often the simplest strong option for tabular models."]
question: "A teammate target-encodes a high-cardinality product_id by computing the mean conversion rate per product on the whole training set, then runs 5-fold cross-validation of a gradient-boosted model on the encoded data and reports a CV AUC of 0.94. Production AUC is 0.71. Explain exactly how the leak arises, why CV did not catch it, and give two correct alternatives."
answer: "Each product's encoded value includes the labels of all its training rows, so a row's own conversion label is part of its feature; for rare products that contribution is large (a product with 2 rows has half its encoding from each row). The model learns to map high encodings to conversion. CV did not catch it because the encoding was computed **before** splitting: validation rows were encoded with statistics that include their own labels too, so they leak in exactly the same way as training rows. In production, new rows' labels are unknown when encoded, so the shortcut disappears and AUC drops. Correct options: (1) put the target encoder inside the CV pipeline and cross-fit it, e.g. scikit-learn's TargetEncoder via fit_transform, so each training row is encoded out-of-fold and each validation fold is encoded only from training folds, with smoothing; or (2) use the boosting library's native categorical handling or a hashed or embedded representation that never touches the target."
sources: [{"title": "scikit-learn: Encoding categorical features", "url": "https://scikit-learn.org/stable/modules/preprocessing.html", "note": "OneHotEncoder (infrequent and unknown categories), OrdinalEncoder and TargetEncoder, including the cross-fitting note."}, {"title": "scikit-learn: Importance of cross fitting in target encoding", "url": "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html", "note": "Runnable demonstration of the leak and how cross-fitting removes it."}, {"title": "CatBoost: unbiased boosting with categorical features (Prokhorenkova et al.)", "url": "https://arxiv.org/abs/1706.09516", "note": "Target leakage and prediction shift explained, and ordered target statistics as the fix."}, {"title": "Feature hashing for large scale multitask learning (Weinberger et al.)", "url": "https://arxiv.org/abs/0902.2206", "note": "The hashing trick, signed hashing, and collision analysis."}]
practice: ["categorical-encoding", "stacking-ensemble", "k-fold-cross-validation"]
tags: ["categorical encoding", "target encoding", "data leakage"]
tools: ["scikit-learn"]
---

## Nominal, ordinal and high-cardinality

A categorical feature takes values from a finite set: country, product type, browser. Before encoding, ask two questions. Is there a natural **order** (education level, size S/M/L, star rating)? And how many **levels** are there: a handful (payment method), hundreds (country), or hundreds of thousands (user id, zip code, merchant)? The answers pick the encoding.

~~~dg-flow Choosing an encoding: cardinality and model type decide most cases.
Categorical feature -(ordered?)-> Ordinal codes -(else, few levels)-> One-hot -(many levels, tree model)-> Native categorical splits -(many levels, any model)-> *Cross-fitted target encoding or embeddings
~~~

## One-hot encoding

One-hot creates one binary column per level. For a linear model this is exact: each level gets its own coefficient, with no assumed order. Practical details:

- **Collinearity**: the one-hot columns of a feature sum to 1, the same as the intercept. Unregularised linear regression then has no unique solution, so drop one level (`drop="first"`). With regularisation, keep all levels; the penalty resolves the redundancy and no level becomes the hidden reference.
- **Rare levels**: a level seen 3 times gets a noisy coefficient. Group levels below a frequency into an "infrequent" bucket (`min_frequency`).
- **Unseen levels** at serving time must not crash the service. Map them to all zeros or to the infrequent bucket (`handle_unknown="infrequent_if_exist"`), and monitor how often it happens.
- **Width**: 30,000 zip codes become 30,000 sparse columns. Linear models handle sparse input well; dense tree learners and k-NN do not.

## Ordinal encoding

Mapping levels to integers is right for genuinely ordered categories, where the model can exploit the order. For nominal categories it invents structure: coding red, green, blue as 0, 1, 2 tells a linear model that green is halfway between red and blue. Trees are less affected, since they can isolate any code with two splits, but they need more splits to do it and the arbitrary order influences which groupings are reachable.

## Target encoding and its leakage

Target encoding replaces each level $c$ with a statistic of the target for that level, typically the mean $\bar y_c$ (the fraction of positives, for classification). It compresses any number of levels into one dense, informative column. Two problems must be handled.

**Noise on small levels.** A level seen twice, both times positive, would get the value 1.0. Shrink towards the global mean $\bar y$:

$$\text{enc}(c) = \frac{n_c\,\bar y_c + m\,\bar y}{n_c + m}$$

where $m$ acts as a prior count. Levels with $n_c \gg m$ keep their own mean; rare levels stay near the global rate.

**Label leakage.** If you compute $\bar y_c$ on the training set and then train on the encoded column, each row's own label is part of its feature. For a level with 3 rows, a third of the encoding *is* the row's label. The model learns to trust the column, training scores soar, and validation collapses. The fix is **cross-fitting**: split the training data into $k$ folds and encode each fold with statistics computed from the other folds only. Validation, test and production rows are encoded with statistics from the full training set.

~~~dg-flow Cross-fitted target encoding: no training row is encoded with its own label.
Training data -(split into k folds)-> Fold j -(encode using the other k-1 folds)-> *Out-of-fold encodings -(train)-> Model
Full training data -(fit statistics once)-> Encoder for validation, test, serving
~~~

scikit-learn's `TargetEncoder` does this automatically: `fit_transform` cross-fits, while `fit` followed by `transform` does not, so calling them separately on training data reintroduces the leak. CatBoost uses a related idea, **ordered target statistics**, where each row is encoded using only the rows before it in a random permutation. If the data is time-ordered, compute encodings from the past only, or the encoding leaks the future as well as the label.

```python
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
te = TargetEncoder(smooth="auto", cv=5, random_state=0)
X_tr_enc = te.fit_transform(X_tr[["merchant_id"]], y_tr)   # cross-fitted: safe to train on
X_va_enc = te.transform(X_va[["merchant_id"]])             # full-training-set statistics
ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
```

<div class="ex">
<div class="lab">Worked example</div>

**a merchant id with 50,000 levels**

A fraud model has merchant_id with 50,000 levels, most with fewer than 5 transactions, and a 2% fraud rate. In-sample target encoding: training AUC 0.99, validation AUC 0.63. The model learned that "encoding near 1" means fraud, which is true only because the row's own label set it.

With 5-fold cross-fitting and $m = 20$: a merchant seen 3 times, all fraudulent, encodes as $(3 \times 1 + 20 \times 0.02)/23 = 0.148$ rather than 1.0. Training AUC drops to 0.81 and validation AUC rises to 0.79. The gap between the two now reflects ordinary overfitting, not leakage.

</div>

## Hashing and embeddings

**Feature hashing** maps each level to one of $D$ columns with a hash function, $j = h(c) \bmod D$, often with a second hash for the sign. The dimension is fixed in advance, there is no vocabulary to store or update, and unseen levels need no special case. The price is collisions: with $D = 2^{18}$ (262,144 columns) and 50,000 levels, about 17% of levels, nearly 9,000, share a column with at least one other level. Linear models at web scale use this routinely.

**Embeddings** give each level a learned dense vector of, say, 16 to 64 dimensions, trained with the rest of a neural network. Similar levels end up close, which lets rare levels borrow strength from similar common ones. They need enough data per level to learn anything and a plan for unseen ids (a shared "unknown" vector or hashing into buckets). Lesson 08.04 uses exactly this for users and items.

## Native categorical support

LightGBM, CatBoost, XGBoost and scikit-learn's `HistGradientBoosting` can split a categorical feature into two groups of levels directly, for example by sorting the levels by their gradient statistics at each node and choosing the best cut in that order. This avoids one-hot width and the leakage questions of target encoding, and is often the best first try for tabular data with moderate cardinality. Very high-cardinality features can still overfit through these splits; regularisation parameters such as a minimum count per category matter.

<div class="take">
<div class="lab">Key takeaway</div>

one-hot for few levels, ordinal codes only for real orders, native splits for tree models, and target encoding only when it is smoothed and cross-fitted, because an in-sample target encoding is the label in disguise.

</div>
