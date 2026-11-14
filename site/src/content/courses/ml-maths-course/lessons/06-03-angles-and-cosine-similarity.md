---
id: "06.03"
course: "ml-maths-course"
module: "06"
slug: "angles-and-cosine-similarity"
title: "Angles and Cosine Similarity"
keyIdeas: ["The angle between ⟦7⟧ and ⟦509⟧ satisfies ⟦1847⟧; this ratio is the cosine similarity.", "Cosine 1 means same direction, 0 perpendicular, ⟦499⟧ opposite. It ignores length.", "For unit vectors, ⟦1848⟧, so ranking by distance and by cosine agree."]
question: "Embeddings ⟦1844⟧, ⟦1845⟧, ⟦1846⟧. Give the cosine similarity and the Euclidean distance of ⟦682⟧ and of ⟦119⟧ from ⟦364⟧."
answer: "Step 1, ⟦1861⟧, ⟦1862⟧, ⟦1863⟧, so ⟦1864⟧. Distance ⟦1865⟧. Step 2, ⟦1866⟧, so ⟦1860⟧. Distance ⟦1867⟧. Step 3, ⟦682⟧ points exactly the same way as ⟦364⟧ but is twice as long; cosine ignores that, distance does not."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Geometry from inner products"]
tools: ["NumPy"]
topic: {"id": "06.01", "title": "Geometry from inner products"}
steps: [{"kind": "explain", "title": "Direction, not length", "body": "Two documents on the same topic, one short and one long, get embeddings $a = (3, 4)$ and $b = (6, 8)$. They point the same way; $b$ is just twice as long.\n\nThe plain dot product can't see that. $a^\\top b = 50$ but $a^\\top a = 25$, so $b$ looks more similar to $a$ than $a$ is to itself. For \"similar meaning\" you usually want to compare **direction** only and ignore length.", "question": {"q": "Which pair points in exactly the same direction?", "options": ["(3, 4) and (6, 8)", "(3, 4) and (4, 3)", "(3, 4) and (4, −3)"], "answer": 0, "why": "$(6, 8) = 2 \\cdot (3, 4)$: one is a positive multiple of the other. $(4, 3)$ is close but tilted, and $(4, -3)$ is at a right angle.", "hint": "Same direction means one vector is a positive number times the other."}}, {"kind": "explain", "title": "Cosine similarity", "body": "The dot product equals $\\|x\\|\\,\\|y\\|\\cos\\theta$, where $\\theta$ is the angle between the arrows. Divide out the lengths and only the angle is left:\n\n$$\\cos\\theta = \\frac{x^\\top y}{\\|x\\|\\,\\|y\\|}$$\n\nThis number is the **cosine similarity**. In 2D it is the cosine of the angle you can draw; in $n$ dimensions it is how the angle is defined. For $a = (3, 4)$ and $b = (6, 8)$: $\\frac{50}{5 \\cdot 10} = 1$.", "question": {"q": "What is the cosine similarity of $(3, 4)$ and $(4, 3)$?", "options": ["24", "2.4", "0.96"], "answer": 2, "why": "$\\frac{12 + 12}{5 \\cdot 5} = \\frac{24}{25} = 0.96$, close to 1, as you'd expect for two arrows only 16 degrees apart.", "hint": "Compute the dot product, then divide by the product of the two lengths, not their sum."}}, {"kind": "explain", "title": "Reading the cosine", "body": "The cosine always lies between $-1$ and $1$:\n\n| $\\cos\\theta$ | angle | meaning |\n| --- | --- | --- |\n| $1$ | 0° | same direction |\n| $0$ | 90° | perpendicular, \"unrelated\" |\n| $-1$ | 180° | opposite directions |\n\nFor example, $(3, 4)$ and $(4, -3)$ have dot product $12 - 12 = 0$, so their cosine is 0.", "question": {"q": "What is the cosine similarity of $(3, 4)$ and $(-6, -8)$?", "options": ["−1", "1", "0"], "answer": 0, "why": "$\\frac{-18 - 32}{5 \\cdot 10} = -1$. The arrows lie on the same line but point opposite ways, which is the far end of the scale, not a match.", "hint": "Lying on the same line is not enough. Check the sign of the dot product."}}, {"kind": "explain", "title": "Cosine ignores length", "body": "Scale either vector by a positive number and its cosine with anything stays the same, because the scale appears once on top and once in the length underneath. Distance does not work that way.\n\n$a = (3, 4)$ and $b = (6, 8)$ have cosine 1 but distance $\\|a - b\\| = \\|(-3, -4)\\| = 5$. Make $b$ longer and the distance grows, though the direction never changed.", "question": {"q": "You replace $b$ by $2b = (12, 16)$. What happens to its cosine and distance from $a$?", "options": ["Both stay the same", "Cosine stays, distance rises", "Cosine halves, distance doubles"], "answer": 1, "why": "The cosine is still 1. The distance becomes $\\|(-9, -12)\\| = 15$, three times as large, purely from the change in length.", "hint": "Scaling by 2 multiplies the dot product and one length by 2. What happens to the ratio? Then compute $a - 2b$."}}, {"kind": "explain", "title": "Normalise first", "body": "Scale every vector to length 1 first: $\\hat{a} = a / \\|a\\|$, so $\\hat{a} = (0.6, 0.8)$. For such **unit vectors**, distance and cosine become tied together:\n\n$$\\|x - y\\|^2 = \\|x\\|^2 + \\|y\\|^2 - 2x^\\top y = 2 - 2\\cos\\theta$$\n\nCheck it with $\\hat{a} = (0.6, 0.8)$ and $\\hat{c} = (0.8, -0.6)$: they are perpendicular, so $\\cos\\theta = 0$, and $\\|\\hat{a} - \\hat{c}\\|^2 = 0.04 + 1.96 = 2$.", "question": {"q": "Two unit vectors have cosine similarity 0.5. What is their squared distance?", "options": ["0.5", "1", "1.5"], "answer": 1, "why": "$2 - 2 \\cdot 0.5 = 1$. Each length contributes 1, and the dot product is subtracted twice.", "hint": "Use $2 - 2\\cos\\theta$, with both 2s."}}, {"kind": "explain", "title": "Rankings agree", "body": "Since $\\|x - y\\|^2 = 2 - 2\\cos\\theta$ for unit vectors, a higher cosine always means a smaller distance. Ranking neighbours by cosine and by distance gives the same order.\n\nThat is why vector databases normalise every embedding once when it is stored. After that they can use fast Euclidean nearest-neighbour search and still return the most similar items by cosine.", "question": {"q": "A unit query has cosine 0.9 with $p_1$, 0.2 with $p_2$ and $-0.5$ with $p_3$, all unit vectors. Which is nearest by distance?", "options": ["p₃", "p₂", "p₁"], "answer": 2, "why": "Squared distances are $2 - 1.8 = 0.2$, $2 - 0.4 = 1.6$ and $2 + 1 = 3$. The highest cosine is the closest point; the most negative is the farthest.", "hint": "Plug each cosine into $2 - 2\\cos\\theta$. A negative cosine makes the distance bigger, not smaller."}}, {"kind": "explain", "title": "A shared direction", "body": "Raw transformer embeddings often all lean along one large common direction. Then even unrelated inputs come out with cosines of 0.8 or more.\n\nA 2D version: $u = (10, 1)$ and $v = (10, -1)$ have cosine $\\frac{99}{101} \\approx 0.98$, almost entirely because of the shared first entry. Subtracting the **mean embedding** removes what every vector has in common. Here the mean is $(10, 0)$, which leaves $(0, 1)$ and $(0, -1)$.", "question": {"q": "After subtracting the mean, what is the cosine of $(0, 1)$ and $(0, -1)$?", "options": ["0", "0.98", "−1"], "answer": 2, "why": "They point in opposite directions, so the cosine is $-1$. The 0.98 came from the shared direction, not from the parts that distinguish the two inputs.", "hint": "Look at the vectors left after the subtraction, not the originals."}}, {"kind": "question", "title": "Recap and a check", "body": "Cosine similarity divides the dot product by both lengths, so it measures direction only: 1 same, 0 perpendicular, $-1$ opposite. For unit vectors, distance and cosine give the same ranking.", "question": {"q": "Query $q = (1, 0)$; documents $p = (10, 1)$ and $r = (0.5, 0.5)$, not normalised. Which is true?", "options": ["Both rank p first", "Cosine ranks p first, distance r", "Both rank r first"], "answer": 1, "why": "Cosine: $p$ gives $\\frac{10}{\\sqrt{101}} \\approx 0.99$ and $r$ gives about 0.71. Raw distance: $r$ is about 0.71 away and $p$ about 9.1. Without normalising, length decides the distance ranking.", "hint": "Compute both measures for both documents. The agreement only holds after every vector has length 1."}}]
---

## The big picture

Two documents about the same topic can have embeddings of very different lengths (one is long, one is short). Cosine similarity compares only the **direction**, which is usually what "similar meaning" should depend on.

## The angle formula

In 2D, the law of cosines gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo>=</mo><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mtext> </mtext><mi mathvariant="normal">∥</mi><mi>y</mi><mi mathvariant="normal">∥</mi><mi>cos</mi><mo>⁡</mo><mi>θ</mi></mrow></math></span>, where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>θ</mi></mrow></math></span> is the angle between the arrows. In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> dimensions we use this as the definition of the angle:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>cos</mi><mo>⁡</mo><mi>θ</mi><mo>=</mo><mfrac><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi></mrow><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mtext> </mtext><mi mathvariant="normal">∥</mi><mi>y</mi><mi mathvariant="normal">∥</mi></mrow></mfrac></mrow></math></span></span>
This number is the **cosine similarity**.

~~~dg-raw
%%viz dot live | Drag a and b. The dot product, the angle and the projection update together.
~~~

| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>cos</mi><mo>⁡</mo><mi>θ</mi></mrow></math></span> | Angle | Meaning |
| --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn></mrow></math></span> | 0 degrees | same direction |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn></mrow></math></span> | 90 degrees | perpendicular, "unrelated" |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1</mn></mrow></math></span> | 180 degrees | opposite directions |

```
cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
a = np.array([3.0, 4]); b = np.array([6.0, 8]); c = np.array([4.0, -3])
print(cos(a, b), cos(a, c))  # 1.0 0.0
print(np.linalg.norm(a - b), round(np.linalg.norm(a - c), 4))  # 5.0 7.0711
```

## Cosine versus distance

- Cosine ignores length; distance does not. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><mn>2</mn><mi>a</mi></mrow></math></span> have cosine 1 but distance 5.
- If every vector is first scaled to length 1, then
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mo>−</mo><mi>y</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><mn>1</mn><mo>+</mo><mn>1</mn><mo>−</mo><mn>2</mn><mi>cos</mi><mo>⁡</mo><mi>θ</mi><mo>=</mo><mn>2</mn><mo>−</mo><mn>2</mn><mi>cos</mi><mo>⁡</mo><mi>θ</mi></mrow></math></span></span>
so the closest point by distance is the most similar by cosine. Vector databases rely on this: normalise once, then use fast Euclidean search.

<div class="ex">
<div class="lab">Worked example</div>

**normalising first**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>a</mi><mo>^</mo></mover><mo>=</mo><mo stretchy="false">(</mo><mn>0.6</mn><mo separator="true">,</mo><mn>0.8</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>b</mi><mo>^</mo></mover><mo>=</mo><mo stretchy="false">(</mo><mn>0.6</mn><mo separator="true">,</mo><mn>0.8</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>c</mi><mo>^</mo></mover><mo>=</mo><mo stretchy="false">(</mo><mn>0.8</mn><mo separator="true">,</mo><mo>−</mo><mn>0.6</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mover accent="true"><mi>a</mi><mo>^</mo></mover><mo>−</mo><mover accent="true"><mi>b</mi><mo>^</mo></mover><mi mathvariant="normal">∥</mi><mo>=</mo><mn>0</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>cos</mi><mo>⁡</mo><mo>=</mo><mn>1</mn></mrow></math></span>; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mover accent="true"><mi>a</mi><mo>^</mo></mover><mo>−</mo><mover accent="true"><mi>c</mi><mo>^</mo></mover><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><mn>2</mn><mo>−</mo><mn>0</mn><mo>=</mo><mn>2</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>cos</mi><mo>⁡</mo><mo>=</mo><mn>0</mn></mrow></math></span>.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

cosine similarity is the dot product of the normalised vectors; it measures direction only.

</div>

## Where this breaks in practice

- Raw transformer embeddings often all share one large common direction, so cosines between unrelated inputs come out high (0.8 or more). Subtracting the mean embedding first fixes much of this.
