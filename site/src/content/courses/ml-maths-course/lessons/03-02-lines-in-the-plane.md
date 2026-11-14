---
id: "03.02"
course: "ml-maths-course"
module: "03"
slug: "lines-in-the-plane"
title: "Lines in the Plane"
keyIdeas: ["A line in the plane can be written ⟦678⟧, or compactly ⟦679⟧.", "To draw it, find two points, for example by setting ⟦680⟧ and then ⟦681⟧.", "Multiplying ⟦378⟧ and ⟦682⟧ by the same non-zero number gives the same line."]
question: "Find two points on ⟦676⟧, and check whether ⟦677⟧ is on it."
answer: "Step 1, set ⟦680⟧: ⟦703⟧, so ⟦704⟧. Step 2, set ⟦681⟧: ⟦705⟧, so ⟦706⟧. Step 3, check ⟦677⟧: ⟦707⟧. Yes, it is on the line."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Lines, planes and geometric constraints"]
tools: ["NumPy"]
topic: {"id": "03.01", "title": "Lines, planes and geometric constraints"}
---

## The big picture

The simplest decision boundary is a straight line: points on one side are one class, points on the other side are the other. Before using lines as classifiers, you need to read and draw them from their equation.

## The equation of a line

In the plane, the set
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">{</mo><mi>x</mi><mo>∈</mo><msup><mi mathvariant="normal">ℝ</mi><mn>2</mn></msup><mo>:</mo><msub><mi>w</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>w</mi><mn>2</mn></msub><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mi>b</mi><mo stretchy="false">}</mo></mrow></math></span></span>
is a line, as long as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><msub><mi>w</mi><mn>1</mn></msub><mo separator="true">,</mo><msub><mi>w</mi><mn>2</mn></msub><mo stretchy="false">)</mo></mrow></math></span> is not <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>. In vector form this is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span>. Read it: "all points <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> whose dot product with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> equals <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span>".

## Drawing it from two points

Two points fix a line. The quickest two are where it crosses the axes.

<div class="ex">
<div class="lab">Worked example</div>

**finding points on a line**

Take <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><mn>4</mn><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>5</mn></mrow></math></span>. Set <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>=</mo><mn>0</mn></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>4</mn><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>5</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>1.25</mn></mrow></math></span>. Point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1.25</mn><mo stretchy="false">)</mo></mrow></math></span>.

Set <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>0</mn></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><msub><mi>x</mi><mn>1</mn></msub><mo>=</mo><mn>5</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>≈</mo><mn>1.67</mn></mrow></math></span>. Point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1.67</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>.

Check another: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>9</mn><mo>−</mo><mn>4</mn><mo>=</mo><mn>5</mn></mrow></math></span>. Also on the line.

</div>

~~~dg-raw
%%viz f_plane x=-1,4 y=-2,3 ln=3,4,5,3x₁+4x₂=5,a,0.62 pt=0,1.25,(0~_1.25);1.667,0,(1.67~_0);3,-1,(3~_−1) | The line 3x₁ + 4x₂ = 5 through the three points found above.
~~~

## Checking many points at once

```
w = np.array([3, 4]); b = 5
X = np.array([[0, 1.25], [3, -1], [1, 1]])
print(X @ w == b)  # [ True  True False]
```

## Same line, different equation

Multiply both sides by 2: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>6</mn><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><mn>8</mn><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>10</mn></mrow></math></span>. Every point that satisfied the first equation satisfies this one, so it is the same line. An equation describes a line only up to this kind of scaling.

## Slope form

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>w</mi><mn>2</mn></msub><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span> you can solve for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>2</mn></msub></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mi>b</mi><mo>−</mo><msub><mi>w</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><msub><mi>w</mi><mn>2</mn></msub></mrow></math></span>. For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><mn>4</mn><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>5</mn></mrow></math></span> this is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>1.25</mn><mo>−</mo><mn>0.75</mn><msub><mi>x</mi><mn>1</mn></msub></mrow></math></span>: slope <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>0.75</mn></mrow></math></span>, crossing at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1.25</mn></mrow></math></span>. The <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> form is preferred because it also handles vertical lines (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>w</mi><mn>2</mn></msub><mo>=</mo><mn>0</mn></mrow></math></span>) and works in any dimension.

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> is a line in the plane. Find two points to draw it, and remember that scaling <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> together changes nothing.

</div>
