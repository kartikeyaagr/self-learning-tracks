---
id: "09.42"
course: "ml-maths-course"
module: "09"
slug: "the-fundamental-theorem-of-calculus"
title: "The Fundamental Theorem of Calculus"
keyIdeas: ["If ⟦2984⟧, then ⟦2985⟧.", "Conversely, ⟦2986⟧: the running area grows at rate ⟦90⟧.", "For a density ⟦102⟧, the CDF ⟦2987⟧ satisfies ⟦2988⟧."]
question: "Compute ⟦2980⟧ with an antiderivative. For the density ⟦2981⟧ on ⟦2982⟧, find ⟦2983⟧."
answer: "Step 1, ⟦2996⟧ has derivative ⟦2328⟧, so ⟦2997⟧. Step 2, ⟦2998⟧ has derivative ⟦2981⟧, so ⟦2999⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["From partitions to volume"]
tools: ["NumPy", "SciPy"]
topic: {"id": "09.06", "title": "From partitions to volume"}
---

## The big picture

Limits of sums are hard to compute directly. The fundamental theorem says integration undoes differentiation, so any antiderivative gives the answer in one line. It also links densities to CDFs.

## Both directions

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msubsup><mo>∫</mo><mi>a</mi><mi>b</mi></msubsup><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi><mo>=</mo><mi>F</mi><mo stretchy="false">(</mo><mi>b</mi><mo stretchy="false">)</mo><mo>−</mo><mi>F</mi><mo stretchy="false">(</mo><mi>a</mi><mo stretchy="false">)</mo><mspace width="1em"/><mtext>when </mtext><msup><mi>F</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo>=</mo><mi>f</mi></mrow></math></span></span>
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mfrac><mi>d</mi><mrow><mi>d</mi><mi>x</mi></mrow></mfrac><msubsup><mo>∫</mo><mi>a</mi><mi>x</mi></msubsup><mi>f</mi><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>t</mi><mo>=</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span></span>

<div class="ex">
<div class="lab">Worked example</div>

**an exponential density**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>λ</mi><msup><mi>e</mi><mrow><mo>−</mo><mi>λ</mi><mi>x</mi></mrow></msup></mrow></math></span> for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>≥</mo><mn>0</mn></mrow></math></span>. Antiderivative: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>λ</mi><mi>x</mi></mrow></msup></mrow></math></span>.

Total: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mi mathvariant="normal">∞</mi></msubsup><mi>p</mi><mo>=</mo><mn>0</mn><mo>−</mo><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></math></span>. CDF: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>F</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn><mo>−</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>λ</mi><mi>x</mi></mrow></msup></mrow></math></span>.

Mean (by parts): <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mi mathvariant="normal">∞</mi></msubsup><mi>x</mi><mi>λ</mi><msup><mi>e</mi><mrow><mo>−</mo><mi>λ</mi><mi>x</mi></mrow></msup><mi>d</mi><mi>x</mi><mo>=</mo><mn>1</mn><mi mathvariant="normal">/</mi><mi>λ</mi></mrow></math></span>.

</div>

~~~dg-raw
%%viz f_curve f=2*Math.exp(-2*x) g=1-Math.exp(-2*x) x=0,3 y=0,2 tk=0.5 fl=density_p(x)_=_2e^(−2x) gl=CDF_F(x)_=_1_−_e^(−2x) | The exponential density (green) and its running area, the CDF (orange). The CDF's slope at each x equals the density there; it levels off at 1, the total area.
~~~

```
from scipy.integrate import quad
lam = 2.0
print(round(quad(lambda x: lam * np.exp(-lam * x), 0, np.inf)[0], 6))  # 1.0
print(round(quad(lambda x: lam * np.exp(-lam * x), 0, 0.5)[0], 4))  # 0.6321
print(round(quad(lambda x: x * lam * np.exp(-lam * x), 0, np.inf)[0], 6))  # 0.5
```

<div class="take">
<div class="lab">Key takeaway</div>

integrate by finding an antiderivative; the running integral of a density is its CDF, whose derivative is the density.

</div>
