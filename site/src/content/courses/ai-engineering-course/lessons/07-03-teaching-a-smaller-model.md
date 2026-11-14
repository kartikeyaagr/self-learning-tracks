---
id: "07.03"
course: "ai-engineering-course"
module: "07"
slug: "teaching-a-smaller-model"
title: "Teaching a smaller model"
keyIdeas: ["Small language models (1-8B parameters) are cheaper and faster to serve, can run on-device, and fine-tune easily.", "Distillation trains a small student to imitate a large teacher instead of learning only from raw data.", "Soft targets: the teacher's full probability distribution says which wrong answers are 'nearly right', which hard labels can't.", "Temperature > 1 softens the teacher's distribution so this information is visible; the loss is KL divergence between teacher and student.", "LLM-era distillation often means sequence-level: generate lots of teacher outputs (including reasoning) and fine-tune the student on them.", "Students rarely beat the teacher overall, but on a narrow task a distilled small model can match or beat a general large one."]
question: "Why do soft labels from a teacher carry more information than hard labels, and what does temperature do in distillation?"
answer: "A hard label only says which answer is right; the teacher's full distribution also says how plausible each wrong answer is (which classes or tokens are similar), so each example gives the student much more signal ('dark knowledge'). Temperature T > 1 divides the logits before softmax, flattening the distribution so those small probabilities become large enough to learn from; the student is trained to match the teacher's softened distribution via KL divergence (scaled by T^2)."
sources: [{"title": "Hinton et al.: Distilling the Knowledge in a Neural Network (2015)", "url": "https://arxiv.org/abs/1503.02531", "note": "Soft targets and temperature."}, {"title": "Sanh et al.: DistilBERT (2019)", "url": "https://arxiv.org/abs/1910.01108", "note": "40% smaller, 60% faster, 97% of BERT's language understanding."}, {"title": "Hsieh et al.: Distilling Step-by-Step (2023)", "url": "https://arxiv.org/abs/2305.02301", "note": "Train small models on LLM rationales; beat a much larger LLM with less data."}]
---

## Why small models

A smaller model is cheaper per token, has lower latency, fits on one GPU or even a phone, and is easier to fine-tune. For a narrow, high-volume task (classifying tickets, extracting fields, routing queries), a 1-8B model is often enough if it's trained well. Distillation is how you train it well.

## The original idea

Hinton et al. (2015) observed that a trained model's output distribution contains more than its top answer. For an image of a 2, the teacher might give "2: 0.9, 3: 0.05, 7: 0.04, cat: 0.0001". The fact that 3 and 7 are plausible and "cat" is absurd is **dark knowledge** about how the classes relate. A hard label ("2") throws this away.

So train the student to match the teacher's whole distribution:

```
p_teacher = softmax(z_teacher / T)
p_student = softmax(z_student / T)
loss = KL(p_teacher || p_student) x T^2  +  alpha x CE(hard label, student)
```

**Temperature** T > 1 flattens both distributions, making small probabilities large enough to learn from. The T^2 factor keeps gradient sizes comparable.

~~~dg-flow Classic logit distillation.
Input -(teacher, frozen)-> Soft probabilities at T -(KL loss)-> *Student update
Input -(student)-> Student probabilities at T -(KL loss)-> Student update
~~~

DistilBERT applied this to language: 40% fewer parameters, 60% faster, keeping 97% of BERT's performance on the GLUE benchmark.

## Distillation for LLMs

With generative models, two styles are common:

- **Logit distillation**: match the teacher's next-token distribution at every position. Needs access to the teacher's logits and the same tokenizer.
- **Sequence-level distillation**: have the teacher generate many responses, then fine-tune the student on them with ordinary supervised training. Works through an API, with any tokenizer.

Hsieh et al.'s "Distilling Step-by-Step" adds the teacher's reasoning (rationales) as an extra training target. Their 770M T5 model beat a 540B PaLM on some benchmarks using less training data than standard fine-tuning. Reasoning distillation is also how many small "thinking" models are made: a large reasoning model's chain-of-thought traces become the student's training data.

## Limits

- The student inherits the teacher's mistakes and biases.
- Students lose breadth: a model distilled on support tickets is great at tickets and worse at everything else.
- Check provider terms: some prohibit using outputs to train competing models.

## See what soft labels carry

A teacher's output distribution contains more than its top answer. Raise the distillation temperature below and see the teacher's view of which wrong answers are close.

~~~dg-raw
%%widget distill Teacher probabilities for one input at different distillation temperatures.
~~~

Hinton, Vinyals and Dean (2015) called this "dark knowledge". An image of a 2 that looks a bit like a 3 and a bit like a 7 produces tiny probabilities for 3 and 7, invisible at T = 1 but clear at T = 4. Training the student to match those softened probabilities teaches it how classes relate, which hard labels can't.

## The loss

```
loss = alpha * T^2 * KL( softmax(teacher_logits / T) || softmax(student_logits / T) )
     + (1 - alpha) * cross_entropy(student_logits, true_label)
```

The T² factor keeps the gradient of the soft term at a similar scale as T changes. Common choices: T between 2 and 5, alpha around 0.5.

## Distillation for LLMs, in three flavours

| Flavour | Needs | Example |
| --- | --- | --- |
| Logit distillation | Teacher's full token distributions | DistilBERT: about 40% smaller, 60% faster, retaining about 97% of BERT's language understanding score |
| Sequence-level (synthetic data) | Only teacher outputs as text | Fine-tune a small model on a large model's answers; works through APIs |
| Reasoning traces | Teacher's step-by-step solutions | DeepSeek-R1 distilled reasoning into 1.5B to 70B Qwen and Llama models |

The synthetic-data flavour is what most teams do in practice, because API models don't return full distributions. Check the teacher provider's terms first; several forbid using outputs to train competing models.

<div class="ex">
<div class="lab">Worked example</div>

**distilling a classifier from a large model**

Label 20,000 support tickets with a frontier model, using a careful prompt and a second pass for disagreements. Cost: a few dollars to tens of dollars.

Fine-tune a small open model on those labels. Evaluate both on 500 human-labelled tickets.

If the student matches the teacher within a point or two, you serve the student at a fraction of the cost. If not, the gap tells you which categories need more examples.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

distillation trains a small student on a large teacher's soft outputs or generated answers. Soft targets carry how-wrong-is-wrong information, and temperature makes it visible; for narrow tasks, the student can match the teacher at a fraction of the cost.

</div>
