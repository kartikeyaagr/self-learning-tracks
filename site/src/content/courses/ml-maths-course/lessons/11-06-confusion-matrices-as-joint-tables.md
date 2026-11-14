---
id: "11.06"
course: "ml-maths-course"
module: "11"
slug: "confusion-matrices-as-joint-tables"
title: "Confusion Matrices as Joint Tables"
keyIdeas: ["A normalised confusion matrix is the joint distribution of (true label, predicted label).", "Recall ⟦3645⟧ normalises a row; precision ⟦3646⟧ normalises a column.", "Accuracy is the sum of the diagonal."]
question: "90 true positives, 10 false negatives, 60 false positives, 840 true negatives. Find recall, precision and accuracy."
answer: "Step 1, recall ⟦3647⟧. Step 2, precision ⟦3648⟧. Step 3, accuracy ⟦3649⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Two discrete variables"]
tools: ["NumPy"]
topic: {"id": "11.01", "title": "Two discrete variables"}
---

## The big picture

Classification metrics look like a zoo of names. They are all marginals and conditionals of one joint table.

## One table, many metrics

| | predicted 0 | predicted 1 |
| --- | --- | --- |
| true 0 | 0.840 | 0.060 |
| true 1 | 0.010 | 0.090 |

```
C = np.array([[840, 60], [10, 90]]) / 1000   # rows: true 0, 1; columns: predicted 0, 1
recall = C[1, 1] / C[1].sum()
precision = C[1, 1] / C[:, 1].sum()
print(round(recall, 2), round(precision, 2), round(np.trace(C), 2))  # 0.9 0.6 0.93
```

~~~dg-raw
%%viz f_bars v=93,90,60 l=accuracy,recall,precision hi=none yl=% | Three metrics from one confusion matrix. Accuracy looks fine because negatives dominate; precision shows that 40% of positive predictions are wrong.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

recall and precision are two conditionals of the same joint table; which row or column you normalise decides the metric.

</div>

## Where this breaks in practice

- Accuracy is dominated by the majority class on imbalanced data. Report precision and recall, or the full table.
