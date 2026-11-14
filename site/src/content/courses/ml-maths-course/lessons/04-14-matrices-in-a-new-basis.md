---
id: "04.14"
course: "ml-maths-course"
module: "04"
slug: "matrices-in-a-new-basis"
title: "Matrices in a New Basis"
keyIdeas: ["A linear map with matrix ⟦8⟧ in standard coordinates has matrix ⟦1485⟧ in basis ⟦41⟧.", "Read ⟦1486⟧ right to left: new coordinates to standard, apply ⟦8⟧, back to new coordinates.", "A good basis can make a messy matrix simple, even diagonal."]
question: "⟦1483⟧, basis ⟦1440⟧, ⟦1441⟧. Find ⟦8⟧ in this basis and use it to compute ⟦1484⟧."
answer: "Step 1, ⟦1493⟧ and ⟦1494⟧. Step 2, so in this basis ⟦1490⟧. Step 3, ⟦731⟧ has coordinates ⟦836⟧; applying ⟦1491⟧ gives ⟦1495⟧. Step 4, back to standard: ⟦1496⟧. Direct check: ⟦1497⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Bases as coordinate systems"]
tools: ["NumPy"]
topic: {"id": "04.03", "title": "Bases as coordinate systems"}
---

## The big picture

A linear map is a fixed action, but the **matrix** that describes it depends on the axes. In the right axes, a matrix that mixes everything up can become simple stretching along each axis. Finding such axes is the goal of eigenvectors (module 05) and PCA (module 08).

## The formula

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is the matrix in standard coordinates and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> holds the new basis, the same map in new coordinates is
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>A</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo>=</mo><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>A</mi><mi>B</mi></mrow></math></span></span>
Read right to left, like a pipeline:

1. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>: turn new coordinates into standard coordinates.
2. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>: apply the map.
3. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span>: turn the result back into new coordinates.

~~~dg-flow Change of basis: the map is the same, only the coordinates change.
new coords -(B)-> standard coords -(A)-> standard coords -(B^-1)-> *new coords
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a messy matrix becomes simple**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> mixes the two coordinates.

In the basis <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>b</mi><mn>1</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>b</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><msub><mi>b</mi><mn>1</mn></msub><mo>=</mo><mn>3</mn><msub><mi>b</mi><mn>1</mn></msub></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><msub><mi>b</mi><mn>2</mn></msub><mo>=</mo><msub><mi>b</mi><mn>2</mn></msub></mrow></math></span>.

So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo>=</mo><mrow><mi mathvariant="normal">d</mi><mi mathvariant="normal">i</mi><mi mathvariant="normal">a</mi><mi mathvariant="normal">g</mi></mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: stretch the first axis by 3, leave the second alone.

</div>

```
A = np.array([[2.0, 1], [1, 2]]); B = np.array([[1.0, 1], [1, -1]])
Ap = np.linalg.inv(B) @ A @ B
print(np.round(Ap, 10))
```

## Columns of A′

Column <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>j</mi></mrow></math></span> of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup></mrow></math></span> is the coordinate vector of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><msub><mi>b</mi><mi>j</mi></msub></mrow></math></span> in the new basis. That is often the quickest way to compute it by hand: apply <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> to each basis vector and express the result in the basis.

<div class="take">
<div class="lab">Key takeaway</div>

in basis <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> the map <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> has matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>A</mi><mi>B</mi></mrow></math></span>. Choose <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> to make that matrix simple.

</div>
