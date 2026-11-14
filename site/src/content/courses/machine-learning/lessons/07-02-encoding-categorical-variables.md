---
id: "07.02"
course: "machine-learning"
module: "07"
slug: "encoding-categorical-variables"
title: "Encoding categorical variables"
keyIdeas: ["Ask two questions about a categorical column first: is there a real **order**, and how many **levels** (cardinality) does it have? The answers pick the encoding.", "**One-hot** encoding gives each level its own 0/1 column; it is exact and assumes no order, but needs a policy for rare levels (`min_frequency`) and unseen ones (`handle_unknown`), and one level dropped for unregularised linear models.", "**Ordinal** (integer) codes are right for genuinely ordered categories when you pass the order explicitly; for nominal data they invent an order and spacing that linear and distance-based models take literally.", "**Target encoding** replaces a level by a smoothed mean of the target, $\\frac{n_c \\bar y_c + m \\bar y}{n_c + m}$. Computed in-sample it leaks each row's own label, so it must be **cross-fitted**: each training row is encoded from folds that never saw it (`TargetEncoder.fit_transform`).", "For very high cardinality, **feature hashing** fixes the width at the cost of collisions, and **embeddings** let a neural network place similar levels close together.", "Gradient-boosting libraries can split on categories natively; that is often the simplest strong option for tabular models."]
question: "A teammate target-encodes a high-cardinality product_id by computing the mean conversion rate per product on the whole training set, then runs 5-fold cross-validation of a gradient-boosted model on the encoded data and reports a CV AUC of 0.94. Production AUC is 0.71. Explain exactly how the leak arises, why CV did not catch it, and give two correct alternatives."
answer: "Each product's encoded value is an average that includes the labels of all its training rows, so every row's own conversion label is part of its own feature. For common products the contribution is negligible, but for rare ones it is large: a product with 2 rows takes half its encoding from each row, and on average a converted row's encoding is $1/n_c$ higher than a non-converted row's in a product of the same size, even when product_id carries no real signal. A gradient-boosted model finds this shortcut and learns that a high encoding means conversion. CV did not catch it because the encoding was computed **before** the data was split: the validation rows in each fold were encoded with statistics that include their own labels, so they leak in exactly the same way as the training rows and the CV score is inflated just as much. In production, a new row's label is unknown when it is encoded, the shortcut disappears, and AUC falls to what the real signal supports. Correct options: (1) put the target encoder inside the CV pipeline and cross-fit it, for example scikit-learn's `TargetEncoder` via `fit_transform` on each training split (each training row encoded out-of-fold, each validation fold encoded from the training folds only), with smoothing so rare products stay near the global rate; or (2) avoid the target altogether: use the boosting library's native categorical splits, or a hashed or embedded representation of product_id, with a minimum-count or infrequent bucket for rare products. A useful check afterwards is that the CV estimate and a time-ordered holdout now agree."
sources: [{"title": "scikit-learn: Encoding categorical features", "url": "https://scikit-learn.org/stable/modules/preprocessing.html", "note": "OneHotEncoder (infrequent and unknown categories), OrdinalEncoder and TargetEncoder, including the cross-fitting note."}, {"title": "scikit-learn: Importance of cross fitting in target encoding", "url": "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html", "note": "Runnable demonstration of the leak and how cross-fitting removes it."}, {"title": "CatBoost: unbiased boosting with categorical features (Prokhorenkova et al.)", "url": "https://arxiv.org/abs/1706.09516", "note": "Target leakage and prediction shift explained, and ordered target statistics as the fix."}, {"title": "Feature hashing for large scale multitask learning (Weinberger et al.)", "url": "https://arxiv.org/abs/0902.2206", "note": "The hashing trick, signed hashing, and collision analysis."}]
practice: ["categorical-encoding", "stacking-ensemble", "k-fold-cross-validation"]
tags: ["categorical encoding", "target encoding", "data leakage"]
tools: ["scikit-learn"]
---

## What you will learn

- What makes a column "categorical", and the two questions (is it ordered? how many levels?) that pick its encoding.
- How one-hot, ordinal, target, hashing and embedding encodings turn labels like "Centre" or "merchant 48213" into numbers, and what each one assumes.
- How to compute a smoothed target encoding by hand, and why the raw version is the label in disguise.
- How cross-fitting removes that leak, and how to do it correctly with scikit-learn's `TargetEncoder`.
- How to handle rare and never-seen categories so a model does not crash or silently misbehave in production.

*Before you start:* you need linear regression and what a coefficient means (02.01), the idea of a validation set and k-fold cross-validation (module 06), and the fit/transform habit from 07.01. Lesson 07.05 later wraps all of this into pipelines.

## From numbers to labels

Lesson 07.01 reshaped numeric columns: scaling, logs, binning. Many of the most useful columns in real data are not numbers at all. They are **categorical**: each value is a label drawn from a finite set, such as a country, a payment method, a product id or a shirt size. Each distinct label is called a **level** (or category). Models do arithmetic, so every level has to become one or more numbers first. That conversion is called **encoding**, and a bad one can quietly cripple a good model or, worse, make a broken model look excellent.

## The idea in one picture

You are predicting monthly rent again, but now the only feature is the neighbourhood. Ten flats:

| Neighbourhood | Centre | Centre | Centre | Centre | Riverside | Riverside | Riverside | Hills | Hills | Old Mill
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rent (thousand / month) | 22 | 26 | 23 | 25 | 16 | 18 | 17 | 13 | 15 | 30

A tempting shortcut is to number the levels alphabetically (Centre = 0, Hills = 1, Old Mill = 2, Riverside = 3) and give that one column to a linear model. The figure shows the result on a larger sample of three neighbourhoods.

~~~dg-fig 07.02-ordinal-vs-onehot.svg
With integer codes, a linear model has one slope, so it must draw a single straight line across Centre, Hills and Riverside and misses Hills badly. One-hot encoding lets each neighbourhood have its own level; the error drops several-fold.
~~~

With integer codes the model can only say "rent changes by $w$ per step of the code", and Hills is cheaper than both its code neighbours, which no straight line can express. With one column per neighbourhood the model simply learns three averages. That is **one-hot encoding**, the default for a reason.

Two properties of a categorical column decide almost everything:

- **Order.** A **nominal** feature has no natural order (colour, country, browser). An **ordinal** feature does (S < M < L, a 1-5 star rating, school level).
- **Cardinality**, the number of distinct levels: a handful (payment method), hundreds (country), or hundreds of thousands (user id, zip code, merchant).

~~~dg-flow Choosing an encoding: order, cardinality and model type decide most cases.
Categorical feature -(ordered?)-> Ordinal codes -(else, few levels)-> One-hot -(many levels, tree model)-> Native categorical splits -(many levels, any model)-> *Cross-fitted target encoding, hashing or embeddings
~~~

## Step 1: one-hot encoding

One-hot creates one binary column per level. A row gets a 1 in the column of its own level and 0 everywhere else:

| Neighbourhood | is_Centre | is_Hills | is_OldMill | is_Riverside
| --- | --- | --- | --- | --- |
| Centre | 1 | 0 | 0 | 0
| Hills | 0 | 1 | 0 | 0
| Riverside | 0 | 0 | 0 | 1

A linear model on these columns computes $\hat y = b + w_{\text{Centre}}\,x_{\text{Centre}} + \dots + w_{\text{Riverside}}\,x_{\text{Riverside}}$. Because exactly one $x$ is 1, the prediction for a Hills flat is just $b + w_{\text{Hills}}$: each level gets its own number, and no order or spacing is assumed. Four practical details matter.

**The dummy-variable trap.** A feature's one-hot columns always sum to 1, exactly the intercept column. Adding a constant to every level weight and subtracting it from $b$ changes no prediction, so unregularised least squares has no unique answer (the collinearity problem of 02.01). Drop one level (`drop="first"`); it becomes the **reference**, and other weights read as "difference from the reference". With regularisation (ridge, lasso, scikit-learn's default `LogisticRegression`) keep all levels: the penalty picks one solution and no level is silently made the baseline.

**Rare levels.** A level seen 3 times gets a coefficient from 3 rows: mostly noise. Real columns have a long tail of such levels.

~~~dg-fig 07.02-longtail.svg
Rows per level for a 2000-level feature, on log scales. 88% of the levels sit below the min_frequency line yet hold only 18% of the rows; merging them into one "infrequent" column removes over 1,700 noisy columns.
~~~

Grouping levels below a threshold into one "infrequent" bucket (`min_frequency=20`) keeps the columns that carry data and drops those that carry noise.

**Unseen levels.** At serving time new levels appear (a new merchant). `handle_unknown="infrequent_if_exist"` sends them to the infrequent bucket, or to all zeros if there is none. Monitor how often it happens.

**Width.** 30,000 zip codes become 30,000 mostly-zero columns: fine for sparse linear models, bad for k-NN and dense learners.

## Step 2: ordinal encoding

**Ordinal encoding** maps levels to integers: S, M, L become 0, 1, 2. For a genuinely ordered feature this is the right choice: one column instead of three, and a linear model can learn "each size step adds $w$". Always pass the order explicitly (`categories=[["S", "M", "L"]]`); left to itself, an encoder sorts alphabetically and you get L = 0, M = 1, S = 2.

Even for real orders, integers assume equal spacing (1 to 2 stars counts the same as 4 to 5). If that is doubtful, one-hot the feature instead.

For nominal features, integer codes invent structure, as the opening figure showed. **Trees** suffer less: threshold splits can isolate any code, though it takes more splits and the arbitrary order limits which groups one split can reach. So: acceptable for trees, wrong for linear and distance-based models.

## Step 3: target encoding

One-hot gets too wide with thousands of levels. **Target encoding** replaces each level with a statistic of the target for that level, usually the mean of $y$ (for a yes/no label, the fraction of positives): "replace each neighbourhood by its average rent". Any number of levels becomes one dense, informative column.

The raw version is $\bar y_c$, the mean target over the $n_c$ training rows with level $c$. A level seen once gets exactly that row's target, which is pure noise. The fix is to **shrink** towards the global mean $\bar y$:

$$\text{enc}(c) = \frac{n_c\,\bar y_c + m\,\bar y}{n_c + m}$$

The symbols:

- $n_c$ is how many training rows have level $c$; $\bar y_c$ is their mean target.
- $\bar y$ is the mean target over all training rows.
- $m$ is the **smoothing** strength, best read as a **prior count**: the formula behaves as if every level had $m$ extra imaginary rows that all sit at the global mean.

When $n_c$ is much bigger than $m$ the level keeps its own mean; when $n_c$ is small it stays near $\bar y$. The figure shows this for a merchant whose every transaction was fraud, with a global fraud rate of 2%.

~~~dg-fig 07.02-shrinkage.svg
Smoothed encoding of a merchant whose every transaction was fraud. With 3 rows and m = 20 the value is 0.148, not 1.0; it only approaches the raw mean once the merchant has many more rows than the prior count.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**smoothed target encoding of the ten flats, by hand**

Totals: Centre $22+26+23+25 = 96$ over 4 flats (mean 24); Riverside $16+18+17 = 51$ over 3 (mean 17); Hills $13+15 = 28$ over 2 (mean 14); Old Mill $30$ over 1. All rents sum to $96+51+28+30 = 205$, so $\bar y = 205/10 = 20.5$.

Take $m = 2$, so each level gets 2 imaginary flats at 20.5, worth $2 \times 20.5 = 41$.

Centre: $(96 + 41)/(4 + 2) = 137/6 = 22.83$. Riverside: $(51 + 41)/(3 + 2) = 92/5 = 18.40$. Hills: $(28 + 41)/(2 + 2) = 69/4 = 17.25$.

Old Mill: $(30 + 41)/(1 + 2) = 71/3 = 23.67$. Its raw mean would be 30, the one flat's own rent; smoothing pulls it a long way back towards 20.5 because one flat is weak evidence.

Now the leak. Train on the *raw* encoding and the Old Mill row's feature is 30, exactly its own label: a perfect predictor for that row. Even smoothed, 30 of the 71 in Old Mill's numerator is the row being predicted.

</div>

## Step 4: stop the leak with cross-fitting

If you compute the encoding on the training set and train on that same column, each row's own label is baked into its own feature: barely for common levels, heavily for rare ones. A flexible model learns "high encoding means positive", training scores soar, and validation collapses, because new rows' labels are unknown when they are encoded.

Below, a product id with 1,500 levels is added to a model, although the target does *not depend on the product at all*.

~~~dg-fig 07.02-leakage.svg
A product id with no real signal. Encoded in-sample, it lifts training AUC far above the truth and drags validation AUC below the model that ignores it. Cross-fitted, it is correctly treated as noise: training and validation agree.
~~~

In numbers (logistic regression, 3,000 training and 3,000 validation rows): with only the real numeric feature the AUC is 0.74 on both sets. Adding the in-sample encoding of the useless product id pushes training AUC to 0.91 and drops validation to 0.62, worse than ignoring the column. The cross-fitted encoding gives 0.74 and 0.74: the model correctly learns to give it almost no weight.

The fix is **cross-fitting** (also called out-of-fold encoding), the same idea as k-fold cross-validation from module 06:

1. Split the training rows into $k$ folds (5 is typical).
2. For each fold, compute the level statistics from the *other* $k - 1$ folds only, and use them to encode this fold.
3. Train the model on these out-of-fold encodings. No training row was ever encoded with its own label.
4. Separately, compute the statistics once on the full training set. Use that encoder for validation, test and production rows.

~~~dg-flow Cross-fitted target encoding: no training row is encoded with its own label.
Training data -(split into k folds)-> Fold j -(encode using the other k-1 folds)-> *Out-of-fold encodings -(train)-> Model
Full training data -(fit statistics once)-> Encoder for validation, test, serving
~~~

Two more rules. When tuning with cross-validation, refit the encoder inside every split (a pipeline, 07.05); encoding everything before splitting leaks into the validation folds too. For time-ordered data, encode each row from the past only. CatBoost builds this in as **ordered target statistics**: each row is encoded from the rows before it in a random permutation.

<div class="ex">
<div class="lab">Worked example</div>

**a merchant id with 50,000 levels**

A fraud model has merchant_id with 50,000 levels, most with fewer than 5 transactions, and a 2% fraud rate. In-sample target encoding gives training AUC 0.99 and validation AUC 0.63: "encoding near 1" meant fraud only because the row's own label put it there.

With 5-fold cross-fitting and $m = 20$, a merchant seen 3 times, all fraudulent, encodes as $(3 \times 1 + 20 \times 0.02)/(3 + 20) = 3.4/23 = 0.148$ instead of 1.0. Training AUC drops to 0.81 and validation rises to 0.79. The remaining small gap is ordinary overfitting, not leakage.

</div>

## Encoding in code

The ten flats again, with a size column added, through scikit-learn's encoders. Printed results are shown as comments.

```python
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

df = pd.DataFrame({
    "area": ["Centre"]*4 + ["Riverside"]*3 + ["Hills"]*2 + ["Old Mill"],
    "size": ["M", "L", "S", "M", "M", "L", "S", "S", "M", "L"],
    "rent": [22, 26, 23, 25, 16, 18, 17, 13, 15, 30.],
})

# One-hot with a rare bucket; Old Mill (1 row) falls below min_frequency=2
ohe = OneHotEncoder(sparse_output=False, min_frequency=2,
                    handle_unknown="infrequent_if_exist")
ohe.fit(df[["area"]])
print(ohe.get_feature_names_out())
# ['area_Centre' 'area_Hills' 'area_Riverside' 'area_infrequent_sklearn']
print(ohe.transform(pd.DataFrame({"area": ["Old Mill", "Docklands"]})))
# [[0. 0. 0. 1.]     rare level -> infrequent column
#  [0. 0. 0. 1.]]    never-seen level -> same column, no crash

# Ordinal with an explicit order (otherwise it would sort alphabetically)
ordn = OrdinalEncoder(categories=[["S", "M", "L"]])
print(ordn.fit_transform(df[["size"]]).ravel())
# [1. 2. 0. 1. 1. 2. 0. 0. 1. 2.]

# Target encoding: fit() stores full-data statistics (for val/test/serving)
te = TargetEncoder(smooth=2.0, target_type="continuous",
                   cv=KFold(5, shuffle=True, random_state=0))
train_enc = te.fit_transform(df[["area"]], df["rent"])   # cross-fitted
print(dict(zip(te.categories_[0], te.encodings_[0].round(2))))
# {'Centre': 22.83, 'Hills': 17.25, 'Old Mill': 23.67, 'Riverside': 18.4}
print(train_enc.ravel().round(2))
# [23.05 22.1 22.95 22.55 18.69 18.56 18.62 18.92 18.25 19.88]
```

The stored encodings match the hand calculation (`smooth=2.0` is $m = 2$), but the training rows did *not* receive them: each was encoded without its own fold. Old Mill's row gets 19.88, near the mean of the other flats, not 30. That is the leak removed.

The crucial API rule: `fit_transform` on training data cross-fits; `fit` then `transform` on the same rows does *not*, and silently brings the leak back. `smooth="auto"` (the default) picks the shrinkage from the data and is a good start.

## Step 5: very many levels, hashing and embeddings

**Feature hashing** (the hashing trick) skips the vocabulary entirely. A hash function maps each level to one of $D$ columns, $j = h(c) \bmod D$, often with a second hash choosing a sign of +1 or -1 so that collisions tend to cancel rather than pile up. The width is fixed in advance, nothing has to be stored or updated, and unseen levels need no special case. The price is **collisions**: unrelated levels sharing a column, so the model cannot tell them apart. With $L$ levels spread uniformly over $D$ columns, the chance that a given level shares its column is $1 - (1 - 1/D)^{L-1}$.

~~~dg-fig 07.02-hashing.svg
Share of 50,000 levels that collide with at least one other level, for different numbers of hash columns. Each doubling of D roughly halves collisions while they are rare; at 2^18 columns about 17% of levels still share.
~~~

For 50,000 merchants in $D = 2^{18} = 262{,}144$ columns that is $1 - e^{-49{,}999/262{,}144} \approx 0.17$, about 8,700 merchants sharing a column. Web-scale linear models (ads, spam) use hashing routinely because a few collisions among rare levels cost little. scikit-learn's version is `FeatureHasher`.

**Embeddings** give each level a learned dense vector of, say, 16 to 64 numbers, trained together with a neural network. Levels that behave alike end up with nearby vectors, so a rare level can borrow strength from similar common ones. They need enough data per level to learn anything, and a plan for unseen ids: a shared "unknown" vector, or hashing ids into buckets first. Lesson 08.04 uses exactly this for users and items.

## Step 6: let the tree model do it

LightGBM, CatBoost, XGBoost and scikit-learn's `HistGradientBoosting` models (pandas `category` columns with `categorical_features="from_dtype"`) split a categorical feature into two groups of levels directly: at each node they sort the levels by their gradient statistics and take the best cut in that order, instead of trying all $2^{L-1}$ subsets. No width, no target leakage questions: often the best first try for tabular data. High-cardinality features can still overfit, so settings such as a minimum row count per category (`min_data_per_group`, `cat_smooth` in LightGBM) matter.

## Checking your encoding

- **Compare training and validation scores.** A big jump in the gap right after adding a target-encoded column is the signature of leakage, not of a great feature.
- **Shuffle test.** Permute the category column and refit; if validation barely moves, the feature carried little signal.
- **Count the unknowns.** The share of serving rows hitting the infrequent bucket or unknown vector should resemble validation's.
- **Inspect the encodings.** Sort levels by encoded value next to their counts. Extreme values on tiny counts mean the smoothing is too weak.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Integer codes for a nominal feature in a linear or k-NN model | odd, unstable fit; results change if levels are renamed | one-hot, or target encoding
| Keeping all one-hot levels with an unregularised linear model | singular-matrix warning, huge offsetting weights | `drop="first"`, or add a penalty
| No policy for unseen levels | crash or error on new data in production | `handle_unknown="infrequent_if_exist"` or `"ignore"`
| Target encoding computed in-sample | training score far above validation and production | cross-fit with `fit_transform`; encode inside each CV split
| Encoding fitted on all data before cross-validation | CV score looks great, production does not | put the encoder in a pipeline (07.05)
| Weak smoothing on rare levels | extreme encodings for 1-3 row levels | larger $m$ or `smooth="auto"`; min-frequency grouping
| Using future rows for time-ordered data | strong backtest, weak live results | encode each row from earlier rows only

## Going deeper

**Why shrinkage is the right shape.** The smoothed formula is a Bayesian posterior mean. For a 0/1 target, give each level's true rate a Beta prior with mean $\bar y$ and pseudo-count $m$; after $n_c$ rows with $n_c \bar y_c$ positives, the posterior mean is exactly $(n_c \bar y_c + m \bar y)/(n_c + m)$. For a continuous target a normal prior on the level mean gives the same form with $m = \sigma^2_{\text{within}} / \tau^2$: noise variance inside a level over the variance of true level means. scikit-learn's `smooth="auto"` estimates this ratio from the data, the classic **empirical Bayes** (random-effects) estimate.

**How big the leak is.** For a level with $n_c$ rows, the in-sample raw encoding of row $i$ contains its own label with weight $1/n_c$. Take a binary target and a level whose true rate is $p$, with no real signal. Averaged over such levels, a positive row's encoding is $p + (1-p)/n_c$ and a negative row's is $p - p/n_c$: positives sit higher by exactly $1/n_c$, whatever $p$ is. That gap is pure label information the model can exploit, and it is largest precisely in the long tail where most levels live. Leave-one-out encoding (each row encoded from the other rows of its level) removes the self-term but creates the reverse artefact: for a fixed level, the encoding is lower for positives than for negatives, and a tree can still read the label from it. K-fold cross-fitting with smoothing is the robust compromise.

**Collisions as variance.** With signed hashing a collision adds an unrelated weight times $\pm 1$, so inner products are preserved in expectation: the error is variance, not bias, and it shrinks as $D$ grows (Weinberger et al.).

**Interactions.** Crossing two features (country by device) multiplies cardinalities; hashing or cross-fitted target encoding of the crossed value keeps it manageable.

## Key terms

| Term | Meaning
| --- | --- |
| Level / category | One distinct value of a categorical feature
| Cardinality | The number of distinct levels a feature has
| Nominal / ordinal | Without / with a natural order among levels
| One-hot encoding | One 0/1 column per level; exactly one is 1 in each row
| Reference level | The dropped one-hot level that other weights are compared to
| Target encoding | Replacing a level by a (smoothed) mean of the target for that level
| Smoothing / prior count $m$ | How strongly rare levels are pulled towards the global mean
| Cross-fitting | Encoding each training fold with statistics from the other folds only
| Feature hashing | Mapping levels to a fixed number of columns with a hash function
| Embedding | A learned dense vector per level, trained with a neural network

<div class="take">
<div class="lab">Key takeaway</div>

one-hot for few levels, ordinal codes only for real orders, native splits for tree models, and target encoding only when it is smoothed and cross-fitted, because an in-sample target encoding is the label in disguise.

</div>
