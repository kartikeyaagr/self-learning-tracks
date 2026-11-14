---
id: "04.11"
course: "ml-maths-course"
module: "04"
slug: "coordinates-in-a-basis"
title: "Coordinates in a Basis"
keyIdeas: ["Every vector in ⟦385⟧ is a combination of the basis vectors in exactly one way; those weights are its coordinates.", "With the basis as columns of ⟦41⟧, coordinates ⟦119⟧ solve ⟦1442⟧, so ⟦1443⟧ for a square basis.", "The same vector has different coordinates in different bases."]
question: "Find the coordinates of ⟦1439⟧ in the basis ⟦1440⟧, ⟦1441⟧."
answer: "Step 1, solve ⟦1452⟧: ⟦1453⟧ and ⟦1454⟧. Step 2, adding: ⟦1337⟧, then ⟦1455⟧. Step 3, check: ⟦1456⟧. Coordinates ⟦836⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Bases as coordinate systems"]
tools: ["NumPy"]
topic: {"id": "04.03", "title": "Bases as coordinate systems"}
---

## The big picture

The point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> means "3 steps along <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>e</mi><mn>1</mn></msub></mrow></math></span>, 1 step along <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>e</mi><mn>2</mn></msub></mrow></math></span>". With different axes, the same point gets different numbers. This lesson shows how to find those numbers.

## Unique coordinates

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>b</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>b</mi><mi>k</mi></msub></mrow></math></span> is a basis of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span>, every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> can be written as
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>x</mi><mo>=</mo><msub><mi>c</mi><mn>1</mn></msub><msub><mi>b</mi><mn>1</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>c</mi><mi>k</mi></msub><msub><mi>b</mi><mi>k</mi></msub></mrow></math></span></span>
in **exactly one** way. (If there were two ways, subtracting them would give a non-zero mix equal to 0, contradicting independence.) The numbers <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><msub><mi>c</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>c</mi><mi>k</mi></msub><mo stretchy="false">)</mo></mrow></math></span> are the **coordinates** of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> in this basis.

~~~dg-raw
%%viz f_plane x=-1,4 y=-2,3 v=3,1,x,k va=0,0,2,2,2b₁,a;2,2,3,1,b₂,f | x = (3, 1) is 2b₁ + 1b₂: two steps along b₁ = (1, 1), then one along b₂ = (1, −1). Its coordinates in this basis are (2, 1).
~~~

## Finding them

Put the basis vectors in the columns of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>. Then "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> has coordinates <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi></mrow></math></span>" means <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mi>B</mi><mi>c</mi></mrow></math></span>, so solve:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>c</mi><mo>=</mo><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>x</mi></mrow></math></span></span>

```
B = np.array([[1.0, 1], [1, -1]])
x = np.array([3.0, 1])
print(np.linalg.solve(B, x))  # [2. 1.]
```

<div class="ex">
<div class="lab">Worked example</div>

**one vector, two coordinate lists**

Standard basis: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> has coordinates <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

Basis <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>b</mi><mn>1</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>b</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: coordinates <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

Same arrow, two lists of numbers.

</div>

## Going back

Given coordinates <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi></mrow></math></span>, the vector is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi><mi>c</mi></mrow></math></span>. So "from coordinates to vector" is multiplication by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>, and "from vector to coordinates" is multiplication by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

coordinates are the unique weights that build <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> from the basis. Find them by solving <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi><mi>c</mi><mo>=</mo><mi>x</mi></mrow></math></span>.

</div>
