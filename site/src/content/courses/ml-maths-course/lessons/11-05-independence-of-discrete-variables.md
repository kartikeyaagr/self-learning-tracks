---
id: "11.05"
course: "ml-maths-course"
module: "11"
slug: "independence-of-discrete-variables"
title: "Independence of Discrete Variables"
keyIdeas: ["⟦513⟧ and ⟦512⟧ are independent when ⟦3642⟧ for every cell.", "Equivalently, every row of the conditional table ⟦3633⟧ is the same.", "With data, test independence with a chi-squared test rather than expecting exact equality."]
question: "Is the table ⟦3621⟧ independent?"
answer: "Step 1, the marginals are ⟦3643⟧ and ⟦309⟧. Step 2, independence would need ⟦3644⟧, but it is 0.4. Not independent."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Two discrete variables"]
tools: ["NumPy", "SciPy"]
topic: {"id": "11.01", "title": "Two discrete variables"}
---

## The big picture

Independence turns a big table into two small lists. Checking it tells you whether one variable carries information about the other.

## The product test

The independent table with given marginals is their outer product.

```
P = np.array([[0.4, 0.2], [0.1, 0.3]])
pX, pY = P.sum(1), P.sum(0)
print(np.outer(pX, pY))
# [[0.3 0.3]
#  [0.2 0.2]]
```

~~~dg-raw
%%viz f_bars v=0.4,0.3,0.2,0.3,0.1,0.2,0.3,0.2 l=(0~0),·,(0~1),·,(1~0),·,(1~1),· hi=none yl=actual_cell_then_product_of_marginals | Each pair of bars compares a cell of the table with the product of its marginals. They differ, so X and Y are dependent.
~~~

## Testing with data

```
from scipy.stats import chi2_contingency
counts = np.array([[400, 200], [100, 300]])
print(chi2_contingency(counts)[1] < 1e-6)  # True
```

With 1000 samples from this table, the p-value is tiny: the dependence is real, not noise.

<div class="take">
<div class="lab">Key takeaway</div>

independent means every cell is the product of its marginals.

</div>

## Where this breaks in practice

- With huge samples, a chi-squared test flags tiny, unimportant dependence. Report a measure of strength (mutual information, correlation) as well.
