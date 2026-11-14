---
id: "08.03"
course: "ai-engineering-course"
module: "08"
slug: "tool-usage-in-llms"
title: "Tool usage in LLMs"
keyIdeas: ["Tool use lets a model call functions it can't do itself: search, calculators, databases, APIs, code execution.", "You describe each tool with a name, a description and a JSON schema for arguments.", "The model decides whether to call a tool and outputs a structured tool call; your code executes it and returns the result as a new message.", "The model then reads the result and answers, or calls another tool. That loop is the core of agents.", "ReAct interleaves reasoning ('I need the population first') with actions and observations; Toolformer trained models to insert API calls themselves.", "Design rules: clear names and descriptions, few overlapping tools, validated arguments, errors returned as informative text, and confirmation for actions with side effects."]
question: "Describe the message flow when a model answers 'What's the weather in Paris?' using a get_weather tool. Who executes the tool?"
answer: "The app sends the user message plus the get_weather tool definition (name, description, JSON schema) to the model. The model returns a structured tool call get_weather(city='Paris') instead of text. The application (not the model) executes the function/API, then sends the result back as a tool-result message. The model reads it and writes the final answer, or calls another tool."
sources: [{"title": "Schick et al.: Toolformer (2023)", "url": "https://arxiv.org/abs/2302.04761", "note": "Models that teach themselves to call APIs."}, {"title": "Yao et al.: ReAct (2022)", "url": "https://arxiv.org/abs/2210.03629", "note": "Interleaving reasoning and actions."}, {"title": "Anthropic: Tool use with Claude", "url": "https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview", "note": "How tool definitions, tool calls and tool results flow."}]
---

## Why tools

An LLM can't look up today's weather, run code reliably, query your database or send an email. It can, however, decide that a tool is needed and write the arguments. Tool use connects that decision to real functions.

## The message flow

~~~dg-seq Tool calling: the model asks, your code acts.
User > App: What's the weather in Paris?
App > Model: user message + tool definitions
Model -> App: tool_call get_weather(city="Paris")
App > Weather API: execute
Weather API -> App: 14 C, cloudy
App > Model: tool_result 14 C, cloudy
Model -> User: It's 14 C and cloudy in Paris.
~~~

The model never executes anything. It produces a structured request; **your application** runs the function, then sends the result back as a tool-result message. Anthropic's tool use docs show exactly this exchange.

## Defining a tool

```
{
  "name": "get_weather",
  "description": "Current weather for a city. Use when the user asks about weather now.",
  "input_schema": {
    "type": "object",
    "properties": { "city": {"type": "string"} },
    "required": ["city"]
  }
}
```

The description is a prompt. It's how the model decides when to use the tool and how to fill the arguments, so write it for a reader who knows nothing about your system.

## Where the idea came from

- **ReAct** (Yao et al., 2022): prompt the model to alternate Thought, Action, Observation. Reasoning picks the next action; observations ground the reasoning in real data. It reduced hallucination on question answering compared with reasoning alone.
- **Toolformer** (Schick et al., 2023): a model learned on its own where API calls (calculator, search, translation) would help it predict text, by keeping only the calls that lowered its loss.
- Today, models are trained directly on function-calling formats, and APIs expose tool calls as structured output.

## Designing good tools

- Use clear, distinct names and descriptions; overlapping tools confuse selection.
- Keep argument schemas small and validate them before execution.
- Return errors as useful text ("city not found, try a country code") so the model can recover.
- Return only what's needed; huge tool outputs fill the context window.
- Require confirmation for tools with side effects (payments, deletes, sending messages).
- Treat tool outputs as untrusted: a web page fetched by a tool can contain prompt injection.

## Step through a tool call

The model never executes a tool. It emits a structured request, your code decides whether to run it, and the result goes back into the conversation. Pick a scenario and step through the messages.

~~~dg-raw
%%widget toolcall The messages exchanged in one tool call, step by step.
~~~

## A tool definition that works

```
{
  "name": "search_orders",
  "description": "Find a customer's orders. Use when the user asks about order status, delivery or returns. Returns at most 20 orders, newest first.",
  "input_schema": {
    "type": "object",
    "properties": {
      "customer_id": {"type": "string", "description": "Internal id, for example C-10293. Not the email."},
      "status": {"type": "string", "enum": ["any", "open", "shipped", "returned"], "default": "any"},
      "since": {"type": "string", "description": "ISO date, for example 2026-01-01"}
    },
    "required": ["customer_id"]
  }
}
```

What makes it good: the description says when to use it and what comes back; enums replace free text where the set is fixed; the id format is spelled out so the model doesn't pass an email; results are bounded.

## Tool results the model can use

- **Return only what is needed.** A 40 KB JSON blob burns context and buries the answer. Return the five fields the model needs, not the whole database row.
- **Make errors actionable.** "customer_id C-1029 not found; ids have 5 digits after C-" lets the model fix its call. A stack trace doesn't.
- **Use stable, human-readable ids** where the model will refer back to them.

## Safety is your code's job

Treat every tool argument as untrusted input: validate types, enforce limits, and check permissions as the end user, not as an all-powerful service account. Separate read tools from write tools, and require confirmation for anything that sends, deletes, pays or publishes. Text that comes back from a tool (a web page, an email) can contain instructions aimed at the model; that is prompt injection, and the defence is to limit what the model is allowed to do next, not to hope it ignores them.

<div class="take">
<div class="lab">Key takeaway</div>

the model chooses a tool and writes arguments; your code executes and returns the result; the model continues. Tool descriptions are prompts, and side-effecting tools need guardrails.

</div>
