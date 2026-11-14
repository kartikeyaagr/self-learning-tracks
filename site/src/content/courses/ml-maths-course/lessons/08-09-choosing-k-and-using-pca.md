---
id: "08.09"
course: "ml-maths-course"
module: "08"
slug: "choosing-k-and-using-pca"
title: "Choosing k and Using PCA"
keyIdeas: ["Pick ⟦1199⟧ by cumulative explained variance, an elbow in the scree plot, or downstream validation.", "Standardise features with different units first, or the feature with the biggest units dominates.", "High variance is not the same as relevance: a low-variance direction can be the one that separates classes."]
question: "Covariance eigenvalues are 4.0, 2.5, 1.0, 0.3, 0.2. How many components explain at least 85%? Keeping 2, what is the mean squared reconstruction error?"
answer: "Step 1, total 8.0. Step 2, cumulative: 50%, 81.25%, 93.75%. So 3 components. Step 3, keeping 2 discards ⟦2442⟧, i.e. 18.75% of the total."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 9-10"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 12-13"}]
tags: ["Principal component analysis"]
tools: ["NumPy", "scikit-learn"]
topic: {"id": "08.03", "title": "Principal component analysis"}
---

## The big picture

PCA always produces all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> components. The practical questions are how many to keep, how to prepare the data, and when PCA is the wrong tool.

## Choosing k

~~~dg-raw
%%viz f_bars v=50,81.25,93.75,97.5,100 l=k=1,k=2,k=3,k=4,k=5 hi=none yl=cumulative_explained_variance_(%) | Cumulative variance for eigenvalues 4.0, 2.5, 1.0, 0.3, 0.2. Three components pass 85%; after that each extra component adds little.
~~~

```
lam = np.array([4.0, 2.5, 1.0, 0.3, 0.2])
print(np.round(100 * np.cumsum(lam) / lam.sum(), 2))  # [ 50.    81.25  93.75  97.5  100.  ]
```

| Method | Use it when |
| --- | --- |
| cumulative variance (90 to 95%) | you want a simple, reportable threshold |
| elbow in the scree plot | eigenvalues drop sharply then flatten |
| downstream validation | PCA feeds a model: tune <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> like a hyperparameter |

## Standardise or not

Variance depends on units. Income in dollars (variance near <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>9</mn></msup></mrow></math></span>) beside age in years (variance near 200) makes PC1 "income" just because of the units. With mixed units, scale every feature to variance 1 first; that is PCA on the correlation matrix. With shared units (pixels, counts), usually don't.

## Uses

- Plotting embeddings in 2D. Try it before t-SNE or UMAP, which keep neighbourhoods but can mislead about global distances.
- Compressing and decorrelating features before a linear model.
- **Whitening**: divide each score by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><msub><mi>λ</mi><mi>i</mi></msub></msqrt></mrow></math></span> to get uncorrelated features with variance 1.

<div class="take">
<div class="lab">Key takeaway</div>

choose <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> from the cumulative variance or validation, standardise mixed units, and remember that variance is not the same as usefulness.

</div>

## Where this breaks in practice

- Forgetting to centre makes PC1 point at the mean. Fit the mean on training data only and reuse it.
- PCA is linear and variance-driven; a low-variance direction can be exactly what separates two classes. Supervised methods (LDA, or the model itself) handle that.
- Outliers inflate variance along their direction and can take over a whole component.
