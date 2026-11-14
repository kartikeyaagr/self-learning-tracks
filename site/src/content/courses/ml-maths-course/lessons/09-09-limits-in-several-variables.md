---
id: "09.09"
course: "ml-maths-course"
module: "09"
slug: "limits-in-several-variables"
title: "Limits in Several Variables"
keyIdeas: ["⟦2529⟧ means ⟦99⟧ gets as close to ⟦491⟧ as you like whenever ⟦7⟧ is close enough to ⟦364⟧, from any direction.", "For sums, products and compositions of continuous functions, just plug in.", "If two paths into ⟦364⟧ give different limits, the limit does not exist."]
question: "Find ⟦2527⟧ and ⟦2528⟧."
answer: "Step 1, the first is a polynomial, so plug in: ⟦2540⟧. Step 2, for the second, ⟦2541⟧ because ⟦2542⟧. Step 3, ⟦2539⟧, so the limit is 0 (squeeze)."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Local change in many variables"]
tools: ["NumPy"]
topic: {"id": "09.02", "title": "Local change in many variables"}
---

## The big picture

Limits in several variables follow the same definition as in one variable, with distance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mo>−</mo><mi>a</mi><mi mathvariant="normal">∥</mi></mrow></math></span> in place of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>x</mi><mo>−</mo><mi>a</mi><mi mathvariant="normal">∣</mi></mrow></math></span>. Most limits you meet are easy; the next lesson shows how they fail.

## Definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><munder><mrow><mi>lim</mi><mo>⁡</mo></mrow><mrow><mi>x</mi><mo>→</mo><mi>a</mi></mrow></munder><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>L</mi></mrow></math></span></span>
means: for every tolerance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ε</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span> there is a radius <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>δ</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span> such that <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo>&lt;</mo><mi mathvariant="normal">∥</mi><mi>x</mi><mo>−</mo><mi>a</mi><mi mathvariant="normal">∥</mi><mo>&lt;</mo><mi>δ</mi></mrow></math></span> implies <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>−</mo><mi>L</mi><mi mathvariant="normal">∣</mi><mo>&lt;</mo><mi>ε</mi></mrow></math></span>. The condition must hold for **all** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> in the ball, whatever direction they come from.

## Plugging in

Polynomials, exponentials, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>sin</mi><mo>⁡</mo></mrow></math></span>, and their sums, products and compositions are continuous wherever defined, so the limit is the value.

## Squeezing

When plugging in gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mi mathvariant="normal">/</mi><mn>0</mn></mrow></math></span>, bound the expression by something that goes to 0.

<div class="ex">
<div class="lab">Worked example</div>

**a squeeze**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mstyle displaystyle="true" scriptlevel="0"><mfrac><mrow><msup><mi>x</mi><mn>2</mn></msup><mi>y</mi></mrow><mrow><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup></mrow></mfrac></mstyle></mrow></math></span> near the origin. Since <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mstyle displaystyle="true" scriptlevel="0"><mfrac><msup><mi>x</mi><mn>2</mn></msup><mrow><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup></mrow></mfrac></mstyle><mo>≤</mo><mn>1</mn></mrow></math></span>, the whole thing is at most <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>y</mi><mi mathvariant="normal">∣</mi></mrow></math></span> in size.

As <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>→</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>y</mi><mi mathvariant="normal">∣</mi><mo>→</mo><mn>0</mn></mrow></math></span>, so the limit is 0.

</div>

```
f = lambda x, y: x ** 2 * y / (x ** 2 + y ** 2)
for r in (1e-1, 1e-3, 1e-6):
    th = np.linspace(0, 2 * np.pi, 9)[:-1]
    print(r, f"{np.abs(f(r * np.cos(th), r * np.sin(th))).max():.1e}")
```

On circles of shrinking radius, the largest value shrinks with the radius.

<div class="take">
<div class="lab">Key takeaway</div>

a limit in several variables must be the same along every route into the point; for continuous formulas, just plug in.

</div>
