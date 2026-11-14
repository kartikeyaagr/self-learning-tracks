---
id: "11.09"
course: "ml-maths-course"
module: "11"
slug: "conditional-densities"
title: "Conditional Densities"
keyIdeas: ["The conditional density ⟦3631⟧ is the slice of the surface at ⟦7⟧, rescaled to area 1.", "Conditioning on ⟦3663⟧ is an event of probability 0, so the conditional is defined by this ratio.", "For the uniform triangle, ⟦3664⟧ is uniform on ⟦3665⟧."]
question: "For the uniform triangle, find ⟦3661⟧ and ⟦3662⟧."
answer: "Step 1, ⟦3667⟧ on ⟦3668⟧ and ⟦3669⟧. Step 2, so ⟦3670⟧ on ⟦3668⟧: uniform, mean 0.25."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Two continuous variables"]
tools: ["NumPy"]
topic: {"id": "11.02", "title": "Two continuous variables"}
---

## The big picture

Every supervised model describes a conditional density: given the input, a distribution over outputs.

## Slices

For a fixed <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>, take the line <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>=</mo><mi>x</mi></mrow></math></span> through the surface. The slice is a curve in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span>. Divide by its area <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> to make it a density.

~~~dg-raw
%%viz f_plane x=-0.2,1.3 y=-0.2,1.3 tk=1 pg=0:0_1:0_1:1,a sg=0.5,0,0.5,0.5,f pt=0.5,0.25,mean_0.25,f | At x = 0.5 the slice (orange) runs from y = 0 to 0.5. Rescaled, it is uniform on that range with mean 0.25.
~~~

```
rng = np.random.default_rng(0)
u = rng.random((400000, 2))
x, y = u.max(1), u.min(1)
near = np.abs(x - 0.5) < 0.01
print(round(y[near].mean(), 2), round(y[near].max(), 2))  # 0.25 0.51
```

<div class="take">
<div class="lab">Key takeaway</div>

a conditional density is a slice of the joint, rescaled to have area 1.

</div>

## Where this breaks in practice

- Estimating <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>y</mi><mo>∣</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> by keeping samples with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> "near" a value needs many samples; in high dimensions almost none are near.
