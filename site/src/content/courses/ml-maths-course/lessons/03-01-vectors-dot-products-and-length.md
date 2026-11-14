---
id: "03.01"
course: "ml-maths-course"
module: "03"
slug: "vectors-dot-products-and-length"
title: "Vectors, Dot Products and Length"
keyIdeas: ["A vector in ⟦2⟧ is a list of ⟦3⟧ numbers. The dot product ⟦128⟧ multiplies matching entries and adds them.", "The length (norm) is ⟦659⟧.", "⟦660⟧ means ⟦378⟧ and ⟦7⟧ are perpendicular (orthogonal)."]
question: "For ⟦655⟧ and ⟦656⟧, compute ⟦128⟧, ⟦394⟧ and ⟦657⟧. Is ⟦658⟧ perpendicular to ⟦378⟧?"
answer: "Step 1, ⟦672⟧. Step 2, ⟦673⟧. Step 3, ⟦674⟧. Step 4, ⟦675⟧, so yes, they are perpendicular."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Lines, planes and geometric constraints"]
tools: ["NumPy"]
topic: {"id": "03.01", "title": "Lines, planes and geometric constraints"}
steps: [{"kind": "explain", "title": "A vector is a list", "body": "Take $w = (3, 4)$. It is a **vector** in $\\mathbb{R}^2$: a list of 2 numbers. In $\\mathbb{R}^d$ a vector is a list of $d$ numbers, and a 768-number sentence embedding is just a longer list.\n\nIn 2D you can draw $w$ as an arrow from the origin to the point $(3, 4)$. In higher dimensions the picture goes, but the arithmetic stays the same: add two vectors entry by entry, and scale a vector by multiplying every entry.\n\n$$(3, 4) + (2, 1) = (5, 5), \\qquad 2 \\cdot (3, 4) = (6, 8)$$", "question": {"q": "What is $(3, 4) + 2 \\cdot (2, 1)$?", "options": ["(7, 6)", "(10, 10)", "(5, 5)"], "answer": 0, "why": "Scale first, $2 \\cdot (2, 1) = (4, 2)$, then add entry by entry: $(3 + 4, 4 + 2) = (7, 6)$.", "hint": "Do the scaling before the adding, and keep each entry in its own position."}}, {"kind": "explain", "title": "The dot product", "body": "The **dot product** of $w$ and $x$ multiplies matching entries and adds the results:\n\n$$w^\\top x = w_1 x_1 + w_2 x_2 + \\dots + w_d x_d$$\n\nThe $\\top$ stands for \"transpose\"; for now read $w^\\top x$ as \"w dot x\". The answer is a single number, not a vector. Both vectors must have the same length $d$, so that every entry has a partner.", "question": {"q": "For $w = (3, 4)$ and $x = (2, 1)$, what is $w^\\top x$?", "options": ["(6, 4)", "10", "11"], "answer": 1, "why": "$3 \\cdot 2 + 4 \\cdot 1 = 6 + 4 = 10$. The products $(6, 4)$ are only the halfway point; the dot product adds them into one number.", "hint": "Pair the first entries, pair the second entries, multiply each pair, then combine the results into a single number."}}, {"kind": "explain", "title": "A model score", "body": "A linear model is a dot product. With weights $w = (0.5, -1, 2)$, an input $x = (4, 1, 0.5)$ gets the score\n\n$$w^\\top x = 0.5 \\cdot 4 + (-1) \\cdot 1 + 2 \\cdot 0.5 = 2 - 1 + 1 = 2$$\n\nEach weight multiplies its own feature. A positive weight pushes the score up as its feature grows; a negative weight pulls it down.", "question": {"q": "The second feature rises from 1 to 3, so $x = (4, 3, 0.5)$. What is the new score?", "options": ["6", "4", "0"], "answer": 2, "why": "$2 - 3 + 1 = 0$. The weight on that feature is $-1$, so raising the feature by 2 lowers the score by 2.", "hint": "Recompute the middle term with its sign: the weight there is negative."}}, {"kind": "explain", "title": "Length", "body": "The **length**, or **norm**, of a vector is the square root of the vector dotted with itself:\n\n$$\\|w\\| = \\sqrt{w^\\top w} = \\sqrt{w_1^2 + \\dots + w_d^2}$$\n\nFor $w = (3, 4)$: $\\sqrt{9 + 16} = \\sqrt{25} = 5$. This is Pythagoras. The arrow is the long side of a right triangle whose other two sides are 3 and 4.", "question": {"q": "What is the length of $x = (2, 1)$?", "options": ["3", "√5 ≈ 2.24", "5"], "answer": 1, "why": "$\\sqrt{2^2 + 1^2} = \\sqrt{5} \\approx 2.24$. The 5 is $x^\\top x$ before the square root; the 3 just adds the entries.", "hint": "Square each entry, add, and don't forget the last operation in the formula."}}, {"kind": "explain", "title": "Dot product and angle", "body": "The dot product also carries the angle $\\theta$ between two arrows:\n\n$$w^\\top x = \\|w\\|\\,\\|x\\| \\cos\\theta$$\n\nLengths are never negative, so the sign of $w^\\top x$ is the sign of $\\cos\\theta$:\n\n- positive: the angle is under 90 degrees;\n- zero: exactly 90 degrees;\n- negative: over 90 degrees, so the arrows point broadly apart.\n\nFor $w = (3, 4)$ and $x = (2, 1)$ the dot product is 10, and the angle is about 27 degrees.", "question": {"q": "Two vectors have $u^\\top v = -6$. What can you say about the angle between them?", "options": ["Less than 90°", "More than 90°", "Exactly 90°"], "answer": 1, "why": "$\\|u\\|\\,\\|v\\|$ is positive, so a negative dot product forces $\\cos\\theta < 0$, which means an angle between 90 and 180 degrees.", "hint": "The two lengths in the formula can't be negative. Which factor must carry the minus sign?"}}, {"kind": "explain", "title": "Perpendicular vectors", "body": "When $w^\\top x = 0$, the formula gives $\\cos\\theta = 0$, so the angle is 90 degrees. Such vectors are **perpendicular**, also called **orthogonal**.\n\nYou never need to draw them to check. Take $(3, 4)$ and $(-8, 6)$: $3 \\cdot (-8) + 4 \\cdot 6 = -24 + 24 = 0$, so they meet at a right angle. In 2D, swapping the two entries and flipping one sign always gives a perpendicular vector.", "question": {"q": "Which vector is perpendicular to $w = (3, 4)$?", "options": ["(4, −3)", "(−3, −4)", "(4, 3)"], "answer": 0, "why": "$3 \\cdot 4 + 4 \\cdot (-3) = 0$. $(-3, -4)$ points the opposite way (dot product $-25$), and $(4, 3)$ is close to $w$ (dot product 24).", "hint": "Compute the dot product of each option with $(3, 4)$. Opposite is not the same as perpendicular."}}, {"kind": "question", "title": "Recap and a check", "body": "A dot product multiplies matching entries and adds them. Length is the square root of a vector dotted with itself. A zero dot product means perpendicular.", "question": {"q": "Take $v = (8, -6)$ and $w = (3, 4)$. Which is true?", "options": ["Length 14, perpendicular to w", "Length 10, dot product 48", "Length 10, perpendicular to w"], "answer": 2, "why": "$\\|v\\| = \\sqrt{64 + 36} = 10$, and $w^\\top v = 24 - 24 = 0$, so $v$ is perpendicular to $w$.", "hint": "For the length, square the entries before adding. For the dot product, keep the minus sign on $-6$."}}]
---

## The big picture

Every model input is a vector: an image is a long list of pixel values, a sentence becomes a list of embedding numbers. Lines, planes and linear systems are all written with three vector tools: the vector itself, the dot product, and the length. This lesson is a short refresher on those three.

## Vectors

A **vector** in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>d</mi></msup></mrow></math></span> is a list of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> numbers, like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>2</mn></msup></mrow></math></span>. You can picture a 2D or 3D vector as an arrow from the origin to that point. In higher dimensions you can't picture it, but the arithmetic is the same.

You add vectors entry by entry and scale them entry by entry: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>5</mn><mo separator="true">,</mo><mn>5</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>⋅</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>6</mn><mo separator="true">,</mo><mn>8</mn><mo stretchy="false">)</mo></mrow></math></span>.

~~~dg-raw
%%viz f_plane x=-1,5 y=-1,5 v=3,4,w;2,1,x,f | Two vectors drawn as arrows from the origin: w = (3, 4) and x = (2, 1).
~~~

## The dot product

The **dot product** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span> multiplies matching entries and adds them:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><msub><mi>w</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>w</mi><mn>2</mn></msub><msub><mi>x</mi><mn>2</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>w</mi><mi>d</mi></msub><msub><mi>x</mi><mi>d</mi></msub></mrow></math></span></span>
For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><mo>⋅</mo><mn>2</mn><mo>+</mo><mn>4</mn><mo>⋅</mo><mn>1</mn><mo>=</mo><mn>10</mn></mrow></math></span>. The little <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">⊤</mi></mrow></math></span> means "transpose"; for now, read <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span> as "w dot x".

## Length

The **length** (or norm) is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi><mo>=</mo><msqrt><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>w</mi></mrow></msqrt></mrow></math></span>. For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mrow><mn>9</mn><mo>+</mo><mn>16</mn></mrow></msqrt><mo>=</mo><mn>5</mn></mrow></math></span>. This is Pythagoras: the arrow is the long side of a right triangle with sides 3 and 4.

## Angles and perpendicular vectors

The dot product also measures angle: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi><mtext> </mtext><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mi>cos</mi><mo>⁡</mo><mi>θ</mi></mrow></math></span>. If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span> then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>cos</mi><mo>⁡</mo><mi>θ</mi><mo>=</mo><mn>0</mn></mrow></math></span>, so the angle is 90 degrees. Such vectors are **perpendicular**, also called **orthogonal**.

```
w = np.array([3, 4]); x = np.array([2, 1])
print(w @ x)  # 10
print(np.linalg.norm(w))  # 5.0
print(w @ np.array([4, -3]))  # 0
```

<div class="ex">
<div class="lab">Worked example</div>

**a model score is a dot product**

A linear model with weights <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><mn>0.5</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span> scores an input <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0.5</mn><mo stretchy="false">)</mo></mrow></math></span> as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mn>2</mn><mo>−</mo><mn>1</mn><mo>+</mo><mn>1</mn><mo>=</mo><mn>2</mn></mrow></math></span>.

Each weight multiplies its own feature, and the results are added. That is all a dot product is.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

dot product = multiply matching entries and add. Length = square root of a vector dotted with itself. Zero dot product = perpendicular.

</div>
