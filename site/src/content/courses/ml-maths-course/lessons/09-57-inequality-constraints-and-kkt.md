---
id: "09.57"
course: "ml-maths-course"
module: "09"
slug: "inequality-constraints-and-kkt"
title: "Inequality Constraints and KKT"
keyIdeas: ["For constraints ⟦3193⟧, the KKT conditions are stationarity, feasibility, ⟦3194⟧, and complementary slackness ⟦3195⟧.", "Complementary slackness: each constraint is either tight or has multiplier zero.", "In an SVM the support vectors are the training points with ⟦3196⟧."]
question: "Minimise ⟦3192⟧ subject to ⟦502⟧. Find ⟦3131⟧ and ⟦1238⟧."
answer: "Step 1, the unconstrained minimum ⟦171⟧ breaks the constraint, so the wall is tight: ⟦3201⟧. Step 2, stationarity: ⟦3202⟧, so ⟦3203⟧. Valid."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Constrained and unconstrained optimization"]
tools: ["SciPy"]
topic: {"id": "09.08", "title": "Constrained and unconstrained optimization"}
---

## The big picture

Many constraints are inequalities: weights must be non-negative, a margin must be at least 1. KKT extends Lagrange multipliers to these.

## The KKT conditions

Minimise <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> subject to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>h</mi><mi>i</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≤</mo><mn>0</mn></mrow></math></span>. At an optimum (under mild regularity conditions) there are <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>μ</mi><mi>i</mi></msub><mo>≥</mo><mn>0</mn></mrow></math></span> with
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi><mo>+</mo><munder><mo>∑</mo><mi>i</mi></munder><msub><mi>μ</mi><mi>i</mi></msub><mi mathvariant="normal">∇</mi><msub><mi>h</mi><mi>i</mi></msub><mo>=</mo><mn>0</mn><mo separator="true">,</mo><mspace width="2em"/><msub><mi>h</mi><mi>i</mi></msub><mo stretchy="false">(</mo><msup><mi>x</mi><mo>⋆</mo></msup><mo stretchy="false">)</mo><mo>≤</mo><mn>0</mn><mo separator="true">,</mo><mspace width="2em"/><msub><mi>μ</mi><mi>i</mi></msub><msub><mi>h</mi><mi>i</mi></msub><mo stretchy="false">(</mo><msup><mi>x</mi><mo>⋆</mo></msup><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></mrow></math></span></span>

~~~dg-raw
%%viz f_curve f=(x-3)*(x-3) x=-1,5 y=0,17 tk=1 ytk=4 dom=-1,1,cc pts=1,best_allowed;3,unconstrained_min | Minimise (x − 3)² with x ≤ 1. The allowed part is solid. The best allowed point is at the wall, x = 1, where the slope is −4 and μ = 4 balances it.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**tight or slack**

Wall at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>≤</mo><mn>1</mn></mrow></math></span>: the constraint is tight and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi><mo>=</mo><mn>4</mn></mrow></math></span>.

Wall at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>≤</mo><mn>5</mn></mrow></math></span>: the unconstrained minimum <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>3</mn></mrow></math></span> is allowed, the constraint is slack, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi><mo>=</mo><mn>0</mn></mrow></math></span>.

</div>

```
from scipy.optimize import minimize
for wall in (1, 5):
    res = minimize(lambda x: (x[0] - 3) ** 2, x0=[0.0], constraints={'type': 'ineq', 'fun': lambda x: wall - x[0]})
    print(wall, round(res.x[0], 4))
# 1 1.0
# 5 3.0
```

~~~dg-flow How constraints are handled in practice.
Constraint -(convex problem, few constraints)-> *solve with a convex solver (cvxpy) using KKT
Constraint -(deep net, simple set)-> project after each step (clip, normalise, softmax parameterisation)
Constraint -(deep net, complex)-> add a penalty or a Lagrangian term to the loss
~~~

<div class="take">
<div class="lab">Key takeaway</div>

each inequality is either pressed against (multiplier positive) or ignored (multiplier zero).

</div>

## Where this breaks in practice

- Penalty methods satisfy constraints only approximately. Check feasibility at the end.
