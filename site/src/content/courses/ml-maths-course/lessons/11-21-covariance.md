---
id: "11.21"
course: "ml-maths-course"
module: "11"
slug: "covariance"
title: "Covariance"
keyIdeas: ["⟦3759⟧.", "⟦3760⟧.", "⟦3761⟧: shifts don't matter, scales multiply."]
question: "Hours studied and exam scores: ⟦3758⟧. Compute the covariance (divide by ⟦233⟧)."
answer: "Step 1, means 2.5 and 65. Step 2, products of deviations: ⟦3762⟧. Step 3, average ⟦3763⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Covariance and correlation"]
tools: ["NumPy"]
topic: {"id": "11.04", "title": "Covariance and correlation"}
---

## The big picture

Do two quantities move together? Covariance measures it, and it appears whenever you add correlated things.

## Signs of the products

For each point, multiply how far <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> is above its mean by how far <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi></mrow></math></span> is above its mean. Both above or both below gives a positive product; one above and one below gives a negative one.

~~~dg-raw
%%viz f_plane x=-2.5,2.5 y=-2.5,2.5 tk=1 pt=-1.5,-1.5,,a;-0.5,-0.5,,a;0.5,0,,k;1.5,2,,a | The study data as deviations from the means: hours across, score in tens of points up. Three points sit in the lower-left or upper-right quarter, where the product of deviations is positive; the fourth sits on the axis and adds 0.
~~~

```
x = np.array([1.0, 2, 3, 4]); y = np.array([50.0, 60, 65, 85])
print(((x - x.mean()) * (y - y.mean())).mean())  # 13.75
```

<div class="take">
<div class="lab">Key takeaway</div>

covariance is the average product of deviations; positive means the variables tend to move together.

</div>

## Where this breaks in practice

- Covariance has units ("hours × points"), so its size is hard to read. Correlation fixes that.
