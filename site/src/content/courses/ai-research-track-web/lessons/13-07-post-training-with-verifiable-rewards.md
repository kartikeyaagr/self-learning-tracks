---
id: "13.07"
course: "ai-research-track-web"
module: "13"
slug: "post-training-with-verifiable-rewards"
title: "Post-training with verifiable rewards"
keyIdeas: ["A modern post-training recipe runs supervised fine-tuning (SFT) on demonstrations, then preference tuning, then reinforcement learning with verifiable rewards (RLVR) on tasks a program can grade, such as maths answers and unit tests.", "SFT is next-token cross-entropy with a loss mask: only assistant tokens contribute, never the system prompt, user turns or padding. How you average over tokens (per token or per sequence) changes the weight long answers get.", "RLVR replaces a learned reward model with a verifier. That removes one way to hack the reward, but verifiers have their own failure modes (answer extraction, formatting, tests that can be special-cased), so they need engineering and auditing.", "GRPO-style training samples a group of completions per prompt and uses each one's reward minus the group mean, often divided by the group standard deviation, as its advantage. No value network is needed, and groups where every sample gets the same reward contribute nothing.", "Evaluation needs fixed sampling settings, unbiased pass@k estimates from several samples, and decontamination: check n-gram overlap between training data and benchmarks, because a contaminated benchmark measures memory, not skill."]
question: "In an RLVR run with GRPO you sample 64 prompts per step with 8 completions each. Logs show 20 prompts where all 8 completions are correct and 12 where all 8 are wrong. Meanwhile pass@1 on your held-out maths benchmark has risen from 40 to 70 percent in 200 steps, while two similar benchmarks moved by 2 points. What fraction of the batch is producing gradient, what would you change in sampling, and what do you suspect about the benchmark jump?"
answer: "Groups with identical rewards have zero advantage for every completion, so 32 of 64 prompts (256 of 512 completions, half the rollout compute) produce no policy gradient beyond any KL term. Filter prompts by pass rate: drop or down-sample those the model already always solves or always fails, and refill the batch until it holds enough mixed groups (sampling more prompts per step and discarding degenerate groups, or keeping a running per-prompt pass-rate estimate to drive a curriculum). Revisit the hardest prompts later, since a few lucky successes are the most informative signal. On the benchmark: a 30-point jump in 200 steps on one benchmark while similar benchmarks barely move suggests contamination or a verifier artefact rather than skill. Check long n-gram overlap (13-grams, say) between the RL and SFT prompt sets and the benchmark questions and answers; check whether the evaluation's answer extraction or template changed; and read a sample of newly-correct outputs to see whether they reason or recite. Report pass@1 averaged over several samples at fixed decoding settings, so the comparison is not sampling noise."
sources: [{"title": "Shao et al.: DeepSeekMath (2024)", "url": "https://arxiv.org/abs/2402.03300", "note": "Section 4: the GRPO objective, group-relative advantages and the outcome versus process supervision variants."}, {"title": "DeepSeek-AI: DeepSeek-R1 (2025)", "url": "https://arxiv.org/abs/2501.12948", "note": "R1-Zero: RL from a base model with rule-based accuracy and format rewards, and the multi-stage R1 recipe."}, {"title": "Lambert et al.: Tulu 3 (2024)", "url": "https://arxiv.org/abs/2411.15124", "note": "A fully open SFT, DPO and RLVR pipeline, including the verifiers and the benchmark decontamination procedure."}, {"title": "Rafailov et al.: Direct Preference Optimization (2023)", "url": "https://arxiv.org/abs/2305.18290", "note": "Section 4: deriving the DPO loss from the KL-regularised reward-maximisation objective."}]
practice: ["build-assistant-only-sft-loss-mask", "normalize-grpo-group-advantages", "grpo-group-relative-advantage", "compare-sequence-and-token-length-normalization", "dpo-closed-form-loss"]
tags: ["SFT", "RLVR", "GRPO"]
tools: ["PyTorch", "TRL", "vLLM"]
---

## The post-training stack

A pretrained model predicts web text; post-training turns it into an assistant that follows instructions and reasons reliably. Earlier modules covered PPO, GRPO and LoRA mechanics. This lesson is about how the pieces are arranged and what goes wrong at scale. Open recipes such as Tulu 3 make the stages explicit:

~~~dg-flow A typical open post-training pipeline.
Base model -(demonstrations)-> SFT -(chosen vs rejected pairs)-> Preference tuning (DPO or RM + PPO) -(prompts with checkable answers)-> *RLVR -> Evaluation and decontamination checks
~~~

Each stage starts from the previous checkpoint and uses a much smaller learning rate than pretraining. Data quality and mixture, again, matter more than clever objectives.

## SFT and the loss mask

SFT data are conversations rendered with the model's chat template. The loss is ordinary cross-entropy, but a mask zeroes every position that is not an assistant token: system prompts, user turns, tool outputs and padding. Training on user tokens teaches the model to generate user text, and training on padding wastes signal. The mask must line up with the shifted labels (the label at position $t$ is token $t+1$), and the end-of-turn token should be included so the model learns to stop.

Two details change results. Packing several short conversations into one sequence needs attention masks or position resets so examples do not attend to each other. And the reduction matters: averaging over all unmasked tokens in a batch weights a 2,000-token answer 20 times more than a 100-token answer, while averaging per sequence first weights them equally. With gradient accumulation, the token-mean must be taken over the whole effective batch, not per micro-batch, or the result depends on how sequences were split.

## Preference tuning, briefly

Preference data give a prompt with a chosen and a rejected response. The classical route trains a Bradley-Terry reward model and optimises against it with PPO plus a KL penalty to the SFT model. DPO skips the reward model, using the closed-form loss

$$\mathcal{L} = -\log \sigma\!\left(\beta\left[\log\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \log\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}\right]\right)$$

on sequence log-probabilities. Preference tuning improves style, helpfulness and refusal behaviour, but a learned reward is exploitable: optimise against it hard enough and the policy finds responses the reward model overrates.

## Reinforcement learning with verifiable rewards

For tasks with checkable outcomes, the reward can come from a program: 1 if the extracted final answer matches the reference, 0 otherwise; the fraction of unit tests passed in a sandbox; whether an instruction's constraints (word count, format, keywords) are met. The Tulu 3 report named this setting RLVR, and DeepSeek-R1-Zero showed that RL with rule-based accuracy and format rewards, starting from a base model, produces long chains of reasoning and large gains on maths benchmarks.

Verifiers are code, and the policy will find their bugs. Common problems: answers that are correct but formatted differently (fractions, LaTeX, units) graded wrong; answers extracted from the wrong place; code that special-cases visible tests; and outputs that crash or hang the grader. Normalise answers, use hidden tests, sandbox with timeouts and read samples of rewarded outputs regularly.

## Group-relative advantages

GRPO, introduced with DeepSeekMath, removes PPO's value network. For each prompt it samples a group of $G$ completions, scores them, and sets each completion's advantage to

$$A_i = \frac{r_i - \operatorname{mean}(r_{1..G})}{\operatorname{std}(r_{1..G}) + \epsilon},$$

applied to every token of completion $i$ inside a PPO-style clipped ratio objective, usually with a KL term to a reference model. The group mean acts as a per-prompt baseline. Three consequences follow.

- If every completion gets the same reward (all solved or all failed), every advantage is zero and the prompt contributes no gradient. Filter or resample prompts so most groups are mixed; a curriculum by pass rate keeps the batch informative.
- Dividing by the standard deviation upweights prompts with low variance (nearly always solved or nearly always failed). Some later work drops it, along with per-sequence length normalisation, which can bias the policy toward longer wrong answers and shorter right ones.
- Rollouts dominate cost. Generation runs on an inference engine and training on the trainer, with weights synchronised every step, so the system design of lessons 2 to 4 reappears on both sides.

## Evaluation and contamination

Reasoning models are evaluated by sampling, so fix the temperature, top-p, maximum length and prompt template, and report the mean over several samples or seeds. For pass@k, draw $n \ge k$ samples per problem, count $c$ correct, and use the unbiased estimator

$$\text{pass@}k = 1 - \binom{n-c}{k}\Big/\binom{n}{k}.$$

Contamination is the larger threat. Web-scale pretraining data and synthetic SFT data can contain benchmark questions and solutions. Check overlap of long n-grams (GPT-3 used 13-grams) between training sets and benchmarks, remove matches from training data, as the Tulu 3 team did, and prefer benchmarks published after the data cut-off. A jump on one benchmark with no movement on similar ones is a contamination smell.

<div class="ex">
<div class="lab">Worked example</div>

**one GRPO group and a pass@k estimate**

Eight completions for one maths prompt receive rewards [1, 0, 0, 1, 0, 0, 0, 1]. Mean 0.375, population standard deviation $\sqrt{0.375 \times 0.625} = 0.484$.

Correct completions get advantage $(1 - 0.375)/0.484 = +1.29$, incorrect ones $(0 - 0.375)/0.484 = -0.77$. The advantages sum to zero. With the sample standard deviation (0.518) they are +1.21 and -0.72.

Evaluation: 16 samples per problem with 4 correct gives pass@1 $= 1 - \binom{12}{1}/\binom{16}{1} = 0.25$, pass@4 $= 1 - 495/1820 = 0.728$ and pass@8 $= 1 - 495/12870 = 0.962$.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

mask SFT to assistant tokens, use preference tuning for style and RLVR for skills a program can check, harden the verifier as carefully as the model, and trust no score you have not decontaminated.

</div>
