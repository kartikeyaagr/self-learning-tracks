---
id: "11.22"
course: "ml-maths-course"
module: "11"
slug: "correlation"
title: "Correlation"
keyIdeas: ["⟦3768⟧ lies in ⟦1879⟧ and has no units.", "Correlation is the cosine of the angle between the centred variables.", "⟦3769⟧ only for an exact straight-line relation."]
question: "For the study data, ⟦3764⟧, ⟦3765⟧, ⟦3766⟧. Find ⟦3767⟧."
answer: "Step 1, ⟦3772⟧. Step 2, ⟦3773⟧, a strong positive linear relationship."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Covariance and correlation"]
tools: ["NumPy"]
topic: {"id": "11.04", "title": "Covariance and correlation"}
---

## The big picture

Correlation is covariance with the units removed, so values can be compared across problems.

## A cosine

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>ρ</mi><mrow><mi>X</mi><mi>Y</mi></mrow></msub><mo>=</mo><mfrac><mrow><mi mathvariant="normal">Cov</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>X</mi><mo separator="true">,</mo><mi>Y</mi><mo stretchy="false">)</mo></mrow><mrow><msub><mi>σ</mi><mi>X</mi></msub><msub><mi>σ</mi><mi>Y</mi></msub></mrow></mfrac><mo>=</mo><mi>cos</mi><mo>⁡</mo><mi mathvariant="normal">∠</mi><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mover accent="true"><mi>x</mi><mo>ˉ</mo></mover><mo separator="true">,</mo><mtext> </mtext><mi>y</mi><mo>−</mo><mover accent="true"><mi>y</mi><mo>ˉ</mo></mover><mo stretchy="false">)</mo></mrow></math></span></span>
By Cauchy–Schwarz, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>ρ</mi><mi mathvariant="normal">∣</mi><mo>≤</mo><mn>1</mn></mrow></math></span>.

```
x = np.array([1.0, 2, 3, 4]); y = np.array([50.0, 60, 65, 85])
xc, yc = x - x.mean(), y - y.mean()
print(round(np.corrcoef(x, y)[0, 1], 3), round(xc @ yc / (np.linalg.norm(xc) * np.linalg.norm(yc)), 3))  # 0.965 0.965
```

~~~dg-raw
%%viz f_bars v=0.965,0,-0.965 l=study_data,no_trend,mirror_image hi=none yl=ρ | Correlation ranges from −1 to 1. Flipping the scores upside down flips its sign.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

correlation is a unit-free cosine between centred variables.

</div>

## Where this breaks in practice

- Pearson correlation is dominated by outliers. Spearman correlation, computed on ranks, is the robust alternative.
