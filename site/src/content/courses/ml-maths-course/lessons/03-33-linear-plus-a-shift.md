---
id: "03.33"
course: "ml-maths-course"
module: "03"
slug: "linear-plus-a-shift"
title: "Linear Plus a Shift"
keyIdeas: ["An affine map is ⟦1226⟧: a linear map followed by a shift.", "It sends lines to lines and keeps ratios along a line, but it moves the origin unless ⟦730⟧.", "Every dense layer, standardisation and batch norm at inference is affine."]
question: "For ⟦1221⟧ and ⟦1222⟧, compute ⟦1223⟧, ⟦1224⟧ and ⟦1225⟧. Is the midpoint kept?"
answer: "Step 1, ⟦1240⟧. Step 2, ⟦1231⟧. Step 3, ⟦1241⟧. Step 4, ⟦855⟧ is the midpoint of ⟦852⟧ and ⟦1118⟧, and ⟦1242⟧ is the midpoint of ⟦1243⟧ and ⟦1244⟧. Yes."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Affine transformations"]
tools: ["NumPy"]
topic: {"id": "03.05", "title": "Affine transformations"}
---

## The big picture

"Multiply by a matrix, then add a vector" is the most common operation in deep learning: every dense layer does it. This combination is called an **affine map**. It behaves almost like a linear map, with one difference that matters: the origin moves.

## The definition

A **linear map** is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>↦</mo><mi>A</mi><mi>x</mi></mrow></math></span>; it always sends 0 to 0. An **affine map** adds a shift:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>A</mi><mi>x</mi><mo>+</mo><mi>b</mi></mrow></math></span></span>
Linear maps are the special case <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><mn>0</mn></mrow></math></span>.

Affine maps send straight lines to straight lines, keep parallel lines parallel, and keep ratios along a line (midpoints go to midpoints). They do not keep the origin, lengths or angles in general.

~~~dg-raw
%%viz m_aff | A linear map followed by a shift. The origin is moved to b.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one map, three points**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> (stretch <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> by 2), <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: the origin moved, so this is not linear.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> is the midpoint of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>5</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

</div>

```
A = np.array([[2, 0], [0, 1]]); b = np.array([1, -1])
f = lambda x: A @ x + b
print(f(np.array([0, 0])), f(np.array([2, 2])), f(np.array([1, 1])))  # [ 1 -1] [5 1] [3 0]
```

## Affine maps in ML

- A dense layer: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi><mo>=</mo><mi>W</mi><mi>x</mi><mo>+</mo><mi>b</mi></mrow></math></span>.
- Standardisation: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi><mo>=</mo><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mi>μ</mi><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mi>σ</mi></mrow></math></span>, which is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>D</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>x</mi><mo>−</mo><msup><mi>D</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>μ</mi></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>D</mi><mo>=</mo><mrow><mi mathvariant="normal">d</mi><mi mathvariant="normal">i</mi><mi mathvariant="normal">a</mi><mi mathvariant="normal">g</mi></mrow><mo stretchy="false">(</mo><mi>σ</mi><mo stretchy="false">)</mo></mrow></math></span>.
- Batch norm at inference: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mi>γ</mi><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mi>μ</mi><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mi>σ</mi><mo>+</mo><mi>β</mi></mrow></math></span> with fixed numbers.

<div class="ex">
<div class="lab">Worked example</div>

**standardising with training statistics**

Training values 10, 20, 30 have mean 20 and standard deviation about 8.16.

A test value of 40 maps to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>40</mn><mo>−</mo><mn>20</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>8.16</mn><mo>≈</mo><mn>2.45</mn></mrow></math></span>: clearly unusual, which is what the model should see.

</div>

```
train = np.array([10.0, 20, 30]); test = np.array([40.0])
mu, sd = train.mean(), train.std()
print(round(mu, 2), round(sd, 2))  # 20.0 8.16
print(np.round((test - mu) / sd, 3))  # [2.449]
```

## Where this breaks in practice

- **Standardising test data with its own statistics.** That leaks information and puts test points in a different coordinate system. Fit <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi></mrow></math></span> on the training set only.

<div class="take">
<div class="lab">Key takeaway</div>

affine = linear + shift. It keeps lines and ratios, but moves the origin.

</div>
