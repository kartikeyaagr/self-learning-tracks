---
id: "01.16"
course: "ml-maths-course"
module: "01"
slug: "open-closed-and-half-open-intervals"
title: "Open, Closed and Half-Open Intervals"
keyIdeas: ["⟦294⟧ includes both ends, ⟦221⟧ excludes both, ⟦295⟧ and ⟦296⟧ include one.", "Square bracket means \"included\", round bracket means \"excluded\". Infinity always gets a round bracket.", "Floats can only approximate an open end: `np.clip(x, 0, 1)` lands in ⟦4⟧, not ⟦5⟧."]
question: "Write in interval notation: (a) probabilities, (b) sigmoid outputs, (c) learning rates (positive), (d) a dropout rate that may be 0 but not 1. Which of these contain their endpoints?"
answer: "Step 1, (a) probabilities can be 0 or 1: ⟦4⟧, both ends included. Step 2, (b) a sigmoid gets arbitrarily close to 0 and 1 but never reaches them in exact maths: ⟦5⟧, no ends. Step 3, (c) any positive real: ⟦279⟧, with a round bracket at 0 and at infinity. Step 4, (d) ⟦33⟧: 0 included, 1 excluded (dropping every unit would delete the layer)."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Properties of sets"]
tools: ["NumPy"]
topic: {"id": "01.03", "title": "Properties of sets"}
---

## The big picture

Intervals are the simplest sets of real numbers, and whether the endpoints belong to them is not a detail. It decides whether a best value can be reached, whether a log is safe to take, and whether a constraint can be met.

## The four kinds

| Notation | Meaning | Ends included |
| --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mi>a</mi><mo separator="true">,</mo><mi>b</mi><mo stretchy="false">]</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>≤</mo><mi>x</mi><mo>≤</mo><mi>b</mi></mrow></math></span> | both (closed) |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>a</mi><mo separator="true">,</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>&lt;</mo><mi>x</mi><mo>&lt;</mo><mi>b</mi></mrow></math></span> | neither (open) |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mi>a</mi><mo separator="true">,</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>≤</mo><mi>x</mi><mo>&lt;</mo><mi>b</mi></mrow></math></span> | left only (half-open) |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>a</mi><mo separator="true">,</mo><mi>b</mi><mo stretchy="false">]</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>&lt;</mo><mi>x</mi><mo>≤</mo><mi>b</mi></mrow></math></span> | right only (half-open) |

Square bracket: the end is in. Round bracket: the end is out. Infinity is not a number, so it always gets a round bracket: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mi mathvariant="normal">∞</mi><mo stretchy="false">)</mo></mrow></math></span> is "0 or more".

<div class="ex">
<div class="lab">Worked example</div>

**is it in the interval?**

Is 1 in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>? No. Is 0 in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>? Yes. Is 0.9999 in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>? Yes.

Is 0 in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mi mathvariant="normal">∞</mi><mo stretchy="false">)</mo></mrow></math></span>? No: "positive" excludes 0.

</div>

~~~dg-raw
%%viz f_line x=-1,3 iv=0,2,cc,[0,2];0,2,oo,(0,2);0,2,co,[0,2) | Same endpoints, different rules at the ends. Filled dot: the endpoint is included. Hollow dot: it is left out.
~~~

## In ML

- `np.random.default_rng().random()` returns numbers in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: it can return 0 but never 1. Python's `range(0, 5)` is the integer version of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>5</mn><mo stretchy="false">)</mo></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mi>p</mi></mrow></math></span> needs <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>∈</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mi mathvariant="normal">∞</mi><mo stretchy="false">)</mo></mrow></math></span>. If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> can be exactly 0, add a small <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ε</mi></mrow></math></span> or clip first.

```
p = np.array([0.0, 0.3, 1.0])
print(np.clip(p, 1e-7, 1 - 1e-7))  # [1.000000e-07 3.000000e-01 9.999999e-01]
```

Clipping pushes values into the closed interval <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>7</mn></mrow></msup><mo separator="true">,</mo><mn>1</mn><mo>−</mo><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>7</mn></mrow></msup><mo stretchy="false">]</mo></mrow></math></span>, safely inside <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mi>p</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>p</mi><mo stretchy="false">)</mo></mrow></math></span> are both finite.

<div class="take">
<div class="lab">Key takeaway</div>

brackets tell you whether the endpoints are in. Logs and divisions care, so check before you compute.

</div>
