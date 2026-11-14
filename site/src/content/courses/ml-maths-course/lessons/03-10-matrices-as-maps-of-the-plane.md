---
id: "03.10"
course: "ml-maths-course"
module: "03"
slug: "matrices-as-maps-of-the-plane"
title: "Matrices as Maps of the Plane"
keyIdeas: ["A ⟦838⟧ matrix ⟦8⟧ moves every point ⟦7⟧ to ⟦839⟧.", "⟦8⟧ sends ⟦310⟧ to its first column and ⟦5⟧ to its second column, so the columns show where the unit square goes.", "The unit square becomes a parallelogram with sides equal to the columns."]
question: "Where does ⟦837⟧ send ⟦310⟧, ⟦5⟧ and ⟦111⟧?"
answer: "Step 1, ⟦848⟧ is the first column, ⟦849⟧. Step 2, ⟦850⟧ is the second column, ⟦111⟧. Step 3, ⟦851⟧, the sum of the two columns. So the unit square becomes the parallelogram with corners ⟦852⟧, ⟦853⟧, ⟦854⟧, ⟦855⟧."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Determinants and singularity"]
tools: ["NumPy"]
topic: {"id": "03.02", "title": "Determinants and singularity"}
---

## The big picture

A matrix is not just a table of numbers. It is a machine that moves points: stretching, squashing, rotating and flipping space. Picturing what a matrix does to a simple square is the key to understanding determinants, inverses and eigenvectors.

## Columns are where the axes go

Take
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>a</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>b</mi></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>c</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>d</mi></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span></span>
Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mi>a</mi><mo separator="true">,</mo><mi>c</mi><mo stretchy="false">)</mo></mrow></math></span>, the first column, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mi>b</mi><mo separator="true">,</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></math></span>, the second column. Every other point is a mix of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>, so it moves to the same mix of the columns: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mn>1</mn></msub><mo separator="true">,</mo><msub><mi>x</mi><mn>2</mn></msub><mo stretchy="false">)</mo><mo>=</mo><msub><mi>x</mi><mn>1</mn></msub><mo stretchy="false">(</mo><mi>a</mi><mo separator="true">,</mo><mi>c</mi><mo stretchy="false">)</mo><mo>+</mo><msub><mi>x</mi><mn>2</mn></msub><mo stretchy="false">(</mo><mi>b</mi><mo separator="true">,</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></math></span>.

~~~dg-raw
%%viz f_plane x=-1,4 y=-1,3 pg=0:0_1:0_1:1_0:1,m;0:0_2:0_3:1_1:1,a v=2,0,Ae₁;1,1,Ae₂,f | The grey unit square and its image under A = [[2, 1], [0, 1]]. The arrows are the columns of A.
~~~

## Familiar maps

| matrix | what it does |
| --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span> | stretch <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> by 2 and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> by 3 |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span> | shear: push the top sideways |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mn>1</mn></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span> | rotate 90 degrees anticlockwise |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span> | mirror across the diagonal |

```
A = np.array([[2, 1], [0, 1]])
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
print((A @ square).T.tolist())  # [[0, 0], [2, 0], [3, 1], [1, 1]]
```

Putting the points as columns lets one product move them all.

## Straight lines stay straight

A matrix map sends lines to lines and keeps the origin fixed. Grid lines become a grid of parallelograms. That is what "linear" means geometrically.

<div class="take">
<div class="lab">Key takeaway</div>

read a matrix by its columns: they are where the axes land, and the unit square becomes the parallelogram they span.

</div>
