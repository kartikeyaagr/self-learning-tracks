---
id: "02.18"
course: "ml-maths-course"
module: "02"
slug: "laws-of-boolean-algebra"
title: "Laws of Boolean Algebra"
keyIdeas: ["Boolean algebra works on ⟦37⟧ with AND (⟦627⟧), OR (⟦628⟧) and NOT (⟦629⟧).", "Its laws mirror the set laws: commutative, associative, distributive both ways, De Morgan and absorption.", "Any Boolean function can be written as an OR of ANDs (disjunctive normal form), read straight off its truth table."]
question: "You want to drop examples that are \"too short and low quality\" and keep the rest. Write the keep condition with De Morgan, and name the common bug."
answer: "Step 1, drop = ⟦645⟧, so keep = ⟦646⟧. Step 2, by De Morgan, keep = ⟦647⟧: keep anything long enough or good enough. Step 3, the common bug is ⟦648⟧, which keeps only examples that are both long and good, and throws away far more data."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Boolean algebra"]
tools: ["Python"]
topic: {"id": "02.03", "title": "Boolean algebra"}
---

## The big picture

Booleans are the simplest numbers, just 0 and 1, and they follow a small set of algebra rules. With these rules you can simplify a messy condition, or rewrite it in a form that is easier to read, without building a truth table every time.

## The operations as arithmetic

Use 1 for true and 0 for false.

- AND, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>∧</mo><mi>b</mi></mrow></math></span>, is multiplication: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>⋅</mo><mi>b</mi></mrow></math></span>.
- OR, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>∨</mo><mi>b</mi></mrow></math></span>, is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>+</mo><mi>b</mi><mo>−</mo><mi>a</mi><mi>b</mi></mrow></math></span>, which equals <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>max</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>a</mi><mo separator="true">,</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span>.
- NOT, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mi>a</mi></mrow></math></span>, is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>−</mo><mi>a</mi></mrow></math></span>.

```
for a in (0, 1):
    for b in (0, 1):
        print(a, b, a * b, a + b - a * b, 1 - a)
```

## The laws

These follow the set laws from module 01: AND matches <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∩</mo></mrow></math></span>, OR matches <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∪</mo></mrow></math></span>, NOT matches complement.

- **Commutative:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>∧</mo><mi>b</mi><mo>=</mo><mi>b</mi><mo>∧</mo><mi>a</mi></mrow></math></span>, and the same for OR.
- **Associative:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>a</mi><mo>∧</mo><mi>b</mi><mo stretchy="false">)</mo><mo>∧</mo><mi>c</mi><mo>=</mo><mi>a</mi><mo>∧</mo><mo stretchy="false">(</mo><mi>b</mi><mo>∧</mo><mi>c</mi><mo stretchy="false">)</mo></mrow></math></span>.
- **Distributive, both ways:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>∧</mo><mo stretchy="false">(</mo><mi>b</mi><mo>∨</mo><mi>c</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mi>a</mi><mo>∧</mo><mi>b</mi><mo stretchy="false">)</mo><mo>∨</mo><mo stretchy="false">(</mo><mi>a</mi><mo>∧</mo><mi>c</mi><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>∨</mo><mo stretchy="false">(</mo><mi>b</mi><mo>∧</mo><mi>c</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mi>a</mi><mo>∨</mo><mi>b</mi><mo stretchy="false">)</mo><mo>∧</mo><mo stretchy="false">(</mo><mi>a</mi><mo>∨</mo><mi>c</mi><mo stretchy="false">)</mo></mrow></math></span>. The second has no version in ordinary arithmetic.
- **De Morgan:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mo stretchy="false">(</mo><mi>a</mi><mo>∧</mo><mi>b</mi><mo stretchy="false">)</mo><mo>=</mo><mi mathvariant="normal">¬</mi><mi>a</mi><mo>∨</mo><mi mathvariant="normal">¬</mi><mi>b</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mo stretchy="false">(</mo><mi>a</mi><mo>∨</mo><mi>b</mi><mo stretchy="false">)</mo><mo>=</mo><mi mathvariant="normal">¬</mi><mi>a</mi><mo>∧</mo><mi mathvariant="normal">¬</mi><mi>b</mi></mrow></math></span>.
- **Absorption:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>∨</mo><mo stretchy="false">(</mo><mi>a</mi><mo>∧</mo><mi>b</mi><mo stretchy="false">)</mo><mo>=</mo><mi>a</mi></mrow></math></span>.

With only two values, any law can be proved by checking every case:

```
from itertools import product
ok1 = all((not (a and b)) == ((not a) or (not b)) for a, b in product([0, 1], repeat=2))
ok2 = all((a or (b and c)) == ((a or b) and (a or c)) for a, b, c in product([0, 1], repeat=3))
print(ok1, ok2)  # True True
```

## Any function from its truth table

**Disjunctive normal form** (DNF) is "an OR of ANDs":

1. Find each row where the output is 1.
2. For that row, AND the inputs, with a NOT on any input that is 0 in that row.
3. OR those terms together.

<div class="ex">
<div class="lab">Worked example</div>

**XOR by the recipe**

XOR outputs 1 on rows <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mn>1</mn></msub><mo separator="true">,</mo><msub><mi>x</mi><mn>2</mn></msub><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

Row <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>∧</mo><mi mathvariant="normal">¬</mi><msub><mi>x</mi><mn>2</mn></msub></mrow></math></span>. Row <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><msub><mi>x</mi><mn>1</mn></msub><mo>∧</mo><msub><mi>x</mi><mn>2</mn></msub></mrow></math></span>.

So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mtext>XOR</mtext><mo stretchy="false">(</mo><msub><mi>x</mi><mn>1</mn></msub><mo separator="true">,</mo><msub><mi>x</mi><mn>2</mn></msub><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><msub><mi>x</mi><mn>1</mn></msub><mo>∧</mo><mi mathvariant="normal">¬</mi><msub><mi>x</mi><mn>2</mn></msub><mo stretchy="false">)</mo><mo>∨</mo><mo stretchy="false">(</mo><mi mathvariant="normal">¬</mi><msub><mi>x</mi><mn>1</mn></msub><mo>∧</mo><msub><mi>x</mi><mn>2</mn></msub><mo stretchy="false">)</mo></mrow></math></span>.

</div>

```
xor_dnf = lambda a, b: (a and not b) or (not a and b)
print([int(xor_dnf(a, b)) for a in (0, 1) for b in (0, 1)])  # [0, 1, 1, 0]
```

<div class="take">
<div class="lab">Key takeaway</div>

Boolean algebra has the same laws as sets. Use De Morgan to negate, and DNF to turn any truth table into a formula.

</div>
