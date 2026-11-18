---
id: "08.03"
course: "ai-engineering-course"
module: "08"
slug: "tool-usage-in-llms"
title: "Tool calling that holds up"
keyIdeas: ["The model never executes anything: it emits a structured call, your code validates and runs it, and the result goes back as a tool-result message matched by call id.", "tool_choice controls whether a call happens: auto (model decides), required/any (must call some tool), forced (must call this tool), none. Models can also emit several calls at once; return one result per call.", "Strict schema mode constrains decoding so arguments always match the schema. It guarantees shape, not correctness, so you still validate values and return actionable errors.", "Selection accuracy falls and cost rises as tool count grows; keep the default set small and load the rest on demand with tool search.", "Errors are part of the interface: flag them as errors, say what went wrong and what to try, cap retries. Streamed arguments are partial JSON until the call completes."]
question: "Your support agent has 60 tools. It often picks the wrong one, sometimes sends arguments that fail validation, and a timeout from the billing API makes it stop with an apology. What would you change, and why?"
answer: "Cut the always-loaded set to the handful of tools used on most turns and put the rest behind tool search with deferred loading, so the model chooses among fewer, clearer options and the prompt shrinks; rewrite overlapping names and descriptions so each says when to use it and what it returns. Turn on strict schema mode so arguments always parse and match the schema, and keep server-side validation for values the schema cannot express (does this order id exist, is the refund within limits). When validation or the billing call fails, return a tool result flagged as an error that names the field, the problem and the next step ('billing timed out, safe to retry once in 5 s'), and cap retries at two or three before escalating. Measure selection accuracy and argument error rate on logged cases before and after."
rubric: ["Reduces the default tool set and uses on-demand loading (tool search, deferred tools, or routing) rather than sending all 60 every call.", "Improves tool names and descriptions so tools do not overlap and each states when to use it.", "Uses strict or constrained argument decoding and still validates values server-side.", "Returns errors as flagged, specific, actionable tool results instead of raising or returning generic failures.", "Bounds retries and escalates, and measures selection and argument accuracy before and after."]
sources: [{"title": "OpenAI: Function calling guide", "url": "https://platform.openai.com/docs/guides/function-calling", "note": "tool_choice, parallel calls, strict mode requirements, tool search, streamed arguments."}, {"title": "Anthropic: Tool use with Claude", "url": "https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview", "note": "Tool definitions, tool_choice, parallel tool use, is_error results, strict tool use."}, {"title": "Anthropic: Introducing advanced tool use (2025)", "url": "https://www.anthropic.com/engineering/advanced-tool-use", "note": "Tool definition token costs, tool search with deferred loading, measured accuracy gains."}, {"title": "Anthropic: Writing effective tools for agents", "url": "https://www.anthropic.com/engineering/writing-tools-for-agents", "note": "Designing tool names, outputs and error messages, evaluated with the agent."}]
practice: ["validate-tool-arguments", "dispatch-tool-call", "authorize-tool-call", "parse-react-step", "mcp-tool-descriptor"]
tools: ["OpenAI API", "Claude API", "Gemini API"]
---

## The message flow

An LLM cannot look up today's weather, query your database or send an email. It can decide that a tool is needed and write the arguments. Tool calling connects that decision to real functions.

~~~dg-seq Tool calling: the model asks, your code acts.
User > App: What's the weather in Paris?
App > Model: user message + tool definitions
Model -> App: tool_call get_weather(city="Paris")
App > Weather API: execute
Weather API -> App: 14 C, cloudy
App > Model: tool_result 14 C, cloudy
Model -> User: It's 14 C and cloudy in Paris.
~~~

The model never executes anything. It produces a structured request with an id; **your application** decides whether to run it, runs it, and sends the result back as a tool-result message carrying the same id. The idea goes back to ReAct (reason, act, observe) and Toolformer; today models are trained on function-calling formats and APIs return calls as structured output. Step through the messages:

~~~dg-raw
%%widget toolcall The messages exchanged in one tool call, step by step.
~~~

## A tool definition is a prompt

```json
{
  "name": "search_orders",
  "description": "Find a customer's orders. Use when the user asks about order status, delivery or returns. Returns at most 20 orders, newest first.",
  "input_schema": {
    "type": "object",
    "properties": {
      "customer_id": {"type": "string", "description": "Internal id, for example C-10293. Not the email."},
      "status": {"type": "string", "enum": ["any", "open", "shipped", "returned"]},
      "since": {"type": "string", "description": "ISO date, for example 2026-01-01"}
    },
    "required": ["customer_id"]
  }
}
```

The description says when to use the tool and what comes back; an enum replaces free text where the set is fixed; the id format is spelled out so the model does not pass an email; results are bounded. Write it for a reader who knows nothing about your system, because that is who reads it.

## Controlling whether a tool is called

Every major API has the same four settings under slightly different names:

| Setting | OpenAI | Anthropic | Gemini | Use it for |
| --- | --- | --- | --- | --- |
| Model decides | `"auto"` | `{"type": "auto"}` | `AUTO` | Default for chat and agents |
| Must call some tool | `"required"` | `{"type": "any"}` | `ANY` | A step that must act, such as a router |
| Must call this tool | `{"type": "function", "name": ...}` | `{"type": "tool", "name": ...}` | `ANY` + one allowed name | Extraction into a fixed shape |
| No tools | `"none"` | `{"type": "none"}` | `NONE` | Final summary turn |

Two traps. A required setting left on in an agent loop means the model can never answer in text, so the loop never ends unless you switch back to auto or provide a "finish" tool. And forcing is not free: on Anthropic's API a forced call starts directly with the tool call, skipping any thinking, and some models and thinking modes reject forced choice outright. When you only want to narrow the options, OpenAI's allowed-tools list restricts calls to a subset without changing the tool list itself, which keeps the cached prompt prefix intact.

## Parallel calls

Asked "compare the weather in Paris and Tokyo", a model can emit two calls in one turn. Run them concurrently, then return **one result per call, all in the next message**, each matched by its call id. Dropping one, or sending them across separate turns, is a common source of API errors. Parallel calls are on by default; turn them off (`parallel_tool_calls: false`, or `disable_parallel_tool_use` inside Anthropic's tool_choice) when calls have side effects that must happen in order, or when the second call depends on the first call's result.

## Strict schema mode

By default arguments are best effort: usually valid, occasionally a missing field, a wrong type or truncated JSON. Strict mode (`strict: true` on the tool definition, on OpenAI and Anthropic) compiles the schema into a grammar and masks the model's sampling so only schema-valid output can be produced. OpenAI recommends always enabling it. Its requirements shape your schemas: `additionalProperties: false` on every object, and every property listed as required, with optional fields expressed as a union with `null`. Some JSON Schema features are unsupported, so check the provider's list. How constrained decoding works is covered in [Structured outputs](lesson:aieng:12.01).

Strict mode guarantees the **shape**, not the **value**. `{"customer_id": "C-99999"}` is schema-valid and may not exist. You still validate values in code.

## When the arguments are wrong

Validate every call before running it: schema, then business rules (does the id exist, is the amount within the order total, does this user own this record). On failure, do not raise inside your loop and do not silently coerce. Return a tool result flagged as an error (`is_error: true` on Anthropic, `isError: true` in MCP) that names the field and the fix:

```json
{"type": "tool_result", "tool_use_id": "toolu_02", "is_error": true,
 "content": "customer_id 'jane@example.com' is not an id. Ids look like C-10293. Call find_customer with the email first."}
```

Anthropic's docs note Claude typically retries two or three times with corrections before giving up. Make that explicit in code: count consecutive failures per tool, and after two or three escalate to a human or stop with a clear message. If the same argument error keeps recurring in logs, fix the description or add an input example; that is cheaper than a retry on every call.

## Tool count and tool search

Every tool definition is input tokens on every call, and more tools means more chances to pick the wrong one. OpenAI's guide suggests, as a soft rule, fewer than 20 functions available at the start of a turn. Anthropic's advanced tool use post measured a five-server MCP setup with 58 tools at about 55,000 tokens of definitions before the conversation started.

The fix is **tool search**: send all definitions, mark most as deferred (`defer_loading: true` on both OpenAI and Anthropic), and give the model a search tool. It sees the search tool plus your few most-used tools, searches when it needs something, and only the matching definitions are loaded. In Anthropic's internal MCP evaluations with large tool libraries, this raised Claude Opus 4 from 49% to 74% and Opus 4.5 from 79.5% to 88.1%. The cost is an extra search step, so it pays when definitions exceed roughly 10,000 tokens or ten-plus tools, and not for a small set used every turn.

<div class="ex">
<div class="lab">Worked example</div>

**what 58 always-loaded tools cost an agent**

Take the 58-tool setup at 55,000 tokens of definitions, and a 20-step agent run (definitions are resent on every step). With an illustrative \$3 per million input tokens and no caching:

- All tools loaded: 55,000 × 20 = 1,100,000 tokens, \$3.30 per run in definitions alone.
- Tool search: about 500 tokens for the search tool plus about 3,000 for the three to five tools found, so 3,500 × 20 = 70,000 tokens, \$0.21 per run.

That is 94% less spent on definitions, before counting the accuracy gain. Prompt caching narrows the gap for the always-loaded version, since definitions sit at the front of the cached prefix ([Context engineering](lesson:aieng:09.01)), but it does nothing for selection accuracy.

</div>

## Results the model can act on

- **Return what is needed, bounded.** A 40 KB JSON row burns context and buries the answer. Return the five fields that matter, paginate lists, and say when results were truncated.
- **Make errors specific.** "Rate limit exceeded, retry after 60 seconds" or "no orders since 2026-01-01; try status=any" lets the model recover. A stack trace or "failed" does not.
- **Use stable, readable ids** the model can refer back to in later calls.

## Streaming arguments

When you stream a response, a tool call's arguments arrive as fragments (`response.function_call_arguments.delta` on OpenAI, `input_json_delta` on Anthropic). Accumulate them per call id and parse only when the call is complete. Use fragments for UI only, such as showing a long document as it is written. Anthropic's fine-grained tool streaming skips server-side buffering for lower latency and can deliver invalid JSON, and a response that hits `max_tokens` can stop mid-argument. Guard the parse, and treat a truncated call as an error to retry, never as something to execute.

## Where this breaks

- **Side effects without a gate.** Separate read tools from write tools, require confirmation for anything that sends, deletes, pays or publishes, and check permissions as the end user, not as a service account.
- **Trusting tool output.** A fetched page or email can carry instructions aimed at the model. Limit what the model may do next rather than hoping it ignores them ([Prompt injection](lesson:aieng:12.03), [Securing agents and MCP servers](lesson:aieng:09.07)).
- **Overlapping tools.** `search_orders` and `find_orders` with similar descriptions guarantee selection errors no prompt will fix. Merge them.

<div class="take">
<div class="lab">Key takeaway</div>

The model chooses a tool and writes arguments; your code validates, executes and returns one result per call. Make that reliable with a small default tool set plus tool search, strict schemas plus value checks, error results that say how to recover, bounded retries, and arguments parsed only once their stream is complete.

</div>
