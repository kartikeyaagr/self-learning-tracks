---
id: "05.05"
course: "ml-maths-course"
module: "05"
slug: "powers-and-the-spectral-radius"
title: "Powers and the Spectral Radius"
keyIdeas: ["On an eigenvector, ⟦1655⟧.", "A general vector is a mix of eigenvectors, and the part with the largest ⟦1635⟧ takes over.", "The spectral radius ⟦1656⟧: if ⟦1657⟧ everything dies out; if ⟦1658⟧ something blows up."]
question: "A linear RNN updates ⟦1651⟧ with eigenvalues ⟦1652⟧ and ⟦1653⟧. What happens to ⟦1654⟧ after 100 steps, and to gradients flowing back 100 steps?"
answer: "Step 1, write ⟦1674⟧ in the eigenvector basis. Step 2, ⟦1675⟧. Step 3, ⟦1676⟧ and ⟦1677⟧: the ⟦1678⟧ part grows, the ⟦1351⟧ part vanishes. Step 4, backpropagation multiplies by ⟦1679⟧ (same eigenvalues) each step, so gradients explode along one direction and vanish along the other."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Invariant directions and spectra"]
tools: ["NumPy"]
topic: {"id": "05.01", "title": "Invariant directions and spectra"}
---

## The big picture

Many systems apply the same matrix again and again: an RNN over time, a Markov chain over steps, gradient descent near a minimum. Eigenvalues predict the long run at a glance: directions with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>λ</mi><mi mathvariant="normal">∣</mi><mo>&gt;</mo><mn>1</mn></mrow></math></span> explode, those with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>λ</mi><mi mathvariant="normal">∣</mi><mo>&lt;</mo><mn>1</mn></mrow></math></span> fade.

## Powers on eigenvectors

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>v</mi><mo>=</mo><mi>λ</mi><mi>v</mi></mrow></math></span> then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mn>2</mn></msup><mi>v</mi><mo>=</mo><mi>A</mi><mo stretchy="false">(</mo><mi>λ</mi><mi>v</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>λ</mi><mn>2</mn></msup><mi>v</mi></mrow></math></span>, and in general
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>A</mi><mi>k</mi></msup><mi>v</mi><mo>=</mo><msup><mi>λ</mi><mi>k</mi></msup><mi>v</mi></mrow></math></span></span>
That is easy to reason about: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1.</mn><msup><mn>1</mn><mn>50</mn></msup><mo>≈</mo><mn>117</mn></mrow></math></span>, while <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.</mn><msup><mn>9</mn><mn>50</mn></msup><mo>≈</mo><mn>0.005</mn></mrow></math></span>.

## A mix of eigenvectors

Write <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><msub><mi>c</mi><mn>1</mn></msub><msub><mi>v</mi><mn>1</mn></msub><mo>+</mo><msub><mi>c</mi><mn>2</mn></msub><msub><mi>v</mi><mn>2</mn></msub></mrow></math></span>. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi>k</mi></msup><mi>x</mi><mo>=</mo><msub><mi>c</mi><mn>1</mn></msub><msubsup><mi>λ</mi><mn>1</mn><mi>k</mi></msubsup><msub><mi>v</mi><mn>1</mn></msub><mo>+</mo><msub><mi>c</mi><mn>2</mn></msub><msubsup><mi>λ</mi><mn>2</mn><mi>k</mi></msubsup><msub><mi>v</mi><mn>2</mn></msub></mrow></math></span>. Each part grows or shrinks at its own rate, so the part with the largest <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>λ</mi><mi mathvariant="normal">∣</mi></mrow></math></span> takes over.

<div class="ex">
<div class="lab">Worked example</div>

**watching one direction win**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>+</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi>k</mi></msup><mi>x</mi><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><msup><mn>3</mn><mi>k</mi></msup><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>+</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

At <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo>=</mo><mn>5</mn></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>122</mn><mo separator="true">,</mo><mn>121</mn><mo stretchy="false">)</mo></mrow></math></span>, almost exactly along <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

</div>

```
A = np.array([[2, 1], [1, 2]])
print(np.linalg.matrix_power(A, 5) @ np.array([1, 0]))  # [122 121]
```

~~~dg-raw
%%viz f_curve f=Math.pow(1.05,x) x=0,100 y=0,140 tk=20 pts=50,k_=_50:_11.5;80,k_=_80:_50 mark=max ml=k_=_100:_132 | The size of the v₁ part, 1.05ᵏ, after k steps. Almost flat for the first 30 steps, then it takes off: about 132 after 100 steps.
~~~

## The spectral radius

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>ρ</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><munder><mrow><mi>max</mi><mo>⁡</mo></mrow><mi>i</mi></munder><mi mathvariant="normal">∣</mi><msub><mi>λ</mi><mi>i</mi></msub><mi mathvariant="normal">∣</mi></mrow></math></span></span>

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ρ</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>&lt;</mo><mn>1</mn></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi>k</mi></msup><mi>x</mi><mo>→</mo><mn>0</mn></mrow></math></span> for every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ρ</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>&gt;</mo><mn>1</mn></mrow></math></span>: some inputs blow up.

```
print(round(1.05 ** 100), 0.6 ** 100 < 1e-22)  # 132 True
```

## Where this breaks in practice

- **Short-run growth.** A matrix whose eigenvectors are far from perpendicular can grow a lot before it decays, even with all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>λ</mi><mi mathvariant="normal">∣</mi><mo>&lt;</mo><mn>1</mn></mrow></math></span>.

```
A = np.array([[0.9, 10], [0, 0.9]])
x = np.array([0.0, 1.0])
print([float(round(np.linalg.norm(np.linalg.matrix_power(A, k) @ x), 1)) for k in (1, 5, 20, 100)])  # [10.0, 32.8, 27.0, 0.0]
```

Both eigenvalues are 0.9, yet the length passes 30 before shrinking. Eigenvalues describe the long run only.

<div class="take">
<div class="lab">Key takeaway</div>

repeated multiplication raises each eigenvalue to the power <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span>. The largest <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>λ</mi><mi mathvariant="normal">∣</mi></mrow></math></span> decides whether things explode or fade.

</div>
