---
id: "09.03"
course: "ai-engineering-course"
module: "09"
slug: "model-context-protocol"
title: "Model Context Protocol"
keyIdeas: ["MCP is an open JSON-RPC protocol connecting AI applications to tools and data, introduced by Anthropic in November 2024. It turns M apps x N services custom integrations into M + N.", "A host (the AI app) runs one client per server. Servers expose tools (model-invoked), resources (app-attached data) and prompts (user-picked templates).", "Transports: stdio for local servers, Streamable HTTP (one POST endpoint, JSON or an SSE stream per request) for remote ones, with OAuth authorization.", "The 2026-07-28 revision made MCP stateless: no initialize handshake or sessions, version and capabilities sent in _meta on every request, server/discover for up-front negotiation, and server-to-client requests such as elicitation carried as input_required results.", "Tools can declare an outputSchema and return structuredContent; elicitation asks the user for input (form or URL mode); the MCP Registry publishes server metadata, not trust."]
question: "Explain the M x N integration problem MCP solves, name MCP's three server primitives with an example of each, and describe how a client calls a tool on a remote server under the 2026-07-28 revision."
answer: "Without a standard, each of M AI apps needs a custom connector for each of N services, M x N integrations. MCP defines one protocol, so each service ships one server and each app one client: M + N. Servers expose tools (model-invoked actions, such as create_issue), resources (data the application attaches to context, such as a file or a table schema) and prompts (templates the user picks, such as 'summarise this PR'). Under 2026-07-28 there is no initialize handshake or session. The client POSTs each JSON-RPC request to the server's single Streamable HTTP endpoint with an OAuth access token, and puts the protocol version and its capabilities in the request's _meta. It may call server/discover first to pick a version. It sends tools/list, gives the definitions to the model, and when the model asks, sends tools/call. The result has resultType complete, with content, optional structuredContent matching the tool's outputSchema, and isError for failures the model can fix; or resultType input_required, asking for user input before the client retries."
rubric: ["Explains M x N versus M + N.", "Names tools, resources and prompts with who controls each and an example.", "Describes Streamable HTTP: one endpoint, POST per request, JSON or SSE response, OAuth.", "Mentions the stateless 2026-07-28 model: no initialize or session, version and capabilities in _meta, optional server/discover.", "Describes the tools/list then tools/call flow and the result shape (content, structuredContent, isError or input_required)."]
sources: [{"title": "MCP specification (2026-07-28)", "url": "https://modelcontextprotocol.io/specification/2026-07-28", "note": "Current revision: stateless requests, transports, primitives, authorization."}, {"title": "MCP specification: Tools (2026-07-28)", "url": "https://modelcontextprotocol.io/specification/2026-07-28/server/tools", "note": "tools/list, tools/call, outputSchema and structuredContent, tool execution errors."}, {"title": "MCP Registry: About", "url": "https://modelcontextprotocol.io/registry/about", "note": "What the official registry stores and who it is for."}, {"title": "Anthropic: Introducing the Model Context Protocol", "url": "https://www.anthropic.com/news/model-context-protocol", "note": "Why it was created (November 2024)."}]
practice: ["mcp-tool-descriptor", "parse-jsonrpc-message", "mcp-dispatch-tool-call", "mcp-read-resource", "mcp-server-capstone"]
tools: ["MCP"]
---

## The integration problem

Every AI application wants the same things: files, GitHub, Slack, databases, calendars. Before MCP, each app wrote its own connector for each service, with its own quirks. Anthropic introduced the Model Context Protocol in November 2024 as an open standard: wrap a service once as an MCP server and it works with every MCP-capable app.

~~~dg-raw
%%widget mcpcount Integrations needed with custom connectors versus a shared protocol.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**five apps, twelve services**

A company runs 5 AI applications (a chat assistant, an IDE agent, a support bot, a data notebook, an internal search) and wants each to reach 12 services. Custom connectors: $5 \times 12 = 60$ integrations to build and keep working through every API change. With MCP: 5 clients plus 12 servers, 17 pieces. Adding a sixth app costs one client instead of 12 connectors; adding a thirteenth service costs one server instead of 5. The saving grows with both counts, which is why most AI apps and agent SDKs now speak MCP.

</div>

## Architecture

~~~dg-flow One host, several servers, one client per server.
*Host (AI app) -(MCP client)-> GitHub server
Host (AI app) -(MCP client)-> Database server
Host (AI app) -(MCP client)-> Local files server
~~~

- **Host**: the AI application the user interacts with (a chat app, an IDE, an agent).
- **Client**: a connector inside the host, one per server.
- **Server**: a program that exposes capabilities, local or remote.

Messages are JSON-RPC 2.0. Two transports are defined. **stdio**: the host launches a local server as a subprocess and talks over stdin and stdout. **Streamable HTTP**: a remote server exposes one endpoint (say `https://example.com/mcp`); the client sends each request as its own POST, and the server answers with a single JSON object or an SSE stream scoped to that request, carrying progress notifications and then the result. Remote servers use OAuth for authorization.

The current revision, **2026-07-28**, made the protocol stateless. Earlier versions opened with an `initialize` handshake and kept a session id in an `Mcp-Session-Id` header. Both are gone. Every request now carries its protocol version and the client's capabilities in `_meta`; servers must implement `server/discover` so a client can learn supported versions and capabilities up front. Any server state that must span calls is passed as an explicit handle in tool arguments. Statelessness lets remote servers sit behind ordinary load balancers, and a broken stream is simply retried as a new request.

## What servers expose

| Primitive | Controlled by | What it is | Example |
| --- | --- | --- | --- |
| Tools | Model | Functions the model can call | create_issue, run_query |
| Resources | Application | Data the app can read into context | a file, a table schema, a log |
| Prompts | User | Reusable templates the user selects | "Summarise this pull request" |

Clients offer features back to servers. The main one now is **elicitation**: a server asks the user for input mid-task. In 2026-07-28, **sampling** (a server borrowing the host's model) and **roots** (telling servers which directories they may use) are deprecated, along with protocol-level logging; servers should call model APIs directly and take paths as tool arguments.

## A tool call on the wire

~~~dg-seq Discovering and calling a tool (2026-07-28).
Host > Server: server/discover (optional)
Host > Server: tools/list
Server -> Host: tools + inputSchema, outputSchema
Host > Server: tools/call get_issue {number: 812}
Server -> Host: resultType complete, content + structuredContent
~~~

The host turns the `tools/list` result into tool definitions for the model ([Tool calling that holds up](lesson:aieng:08.03)); when the model asks for `get_issue`, the client sends `tools/call` and passes the result back as the tool result:

```json
-> {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
    "params": {"name": "get_issue", "arguments": {"number": 812},
      "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {"elicitation": {}}}}}
<- {"jsonrpc": "2.0", "id": 2, "result": {"resultType": "complete",
     "content": [{"type": "text", "text": "{\"number\": 812, \"title\": \"Login fails on Safari\", \"state\": \"open\"}"}],
     "structuredContent": {"number": 812, "title": "Login fails on Safari", "state": "open"},
     "isError": false}}
```

List results also carry `ttlMs` and `cacheScope` so clients can cache tool lists, and servers should return tools in a deterministic order, which keeps the host's prompt cache stable ([Context engineering](lesson:aieng:09.01)).

## Structured output and errors

A tool may declare an **outputSchema**. If it does, the server must return `structuredContent` that conforms to it, and the client should validate it. For older clients the server should also return the same JSON as a text block. Structured content lets the host hand typed data to code (render a table, chain into the next call) instead of parsing prose. It is unrelated to schema-constrained generation by the model ([Structured outputs](lesson:aieng:12.01)). Results can also include **resource links**, pointers the host can fetch later instead of inlining a large payload.

Errors come in two kinds. **Protocol errors** (unknown tool, malformed request) are JSON-RPC errors the model can rarely fix. **Tool execution errors**, including input validation failures, belong in the result with `isError: true` and an actionable message, so the model can correct its arguments and retry.

## Elicitation

Sometimes a server needs something only the user can supply: which of three accounts, a confirmation, a sign-in. Under 2026-07-28 this uses the multi round-trip request pattern: instead of a server-initiated request, the server answers `tools/call` with `resultType: "input_required"` and a set of input requests. The client shows them, then retries the original call with a new request id, the user's `inputResponses`, and any opaque `requestState` the server returned.

There are two modes. **Form mode** asks for structured data against a small JSON schema; the user can review, edit, decline or cancel. **URL mode** sends the user to a web page, and it is mandatory for anything sensitive: servers must not collect passwords, API keys, tokens or payment details through form mode, so those never pass through the client or the model. Clients must show which server is asking and, for URL mode, the target domain.

## The registry

The **MCP Registry** (in preview) is the official metadata catalogue for public servers. A publisher submits a `server.json` with a namespaced name (for example `io.github.user/server-name`, with namespace ownership verified through GitHub or DNS), where to get the server (an npm, PyPI or Docker package, or a remote URL) and how to run it. It stores metadata, not code, and is meant to be read by downstream aggregators and marketplaces, which add curation and ratings; private and internal servers are out of scope. A listing proves who published a server, not that it is safe to run.

## Security, briefly

MCP makes powerful tools easy to connect, so tool descriptions, tool results and annotations from servers you do not control are untrusted input. Install servers you trust, grant least privilege, keep a human able to deny tool calls, and confirm writes. OAuth 2.1 authorization for remote servers (including Client ID Metadata Documents, which 2026-07-28 prefers over dynamic client registration), token passthrough, tool poisoning and rug pulls, and sandboxing are covered in [Securing agents and MCP servers](lesson:aieng:09.07).

## Where this breaks

- **Version skew.** Servers written for 2025-06-18 or 2025-11-25 expect `initialize` and sessions; a 2026-07-28 client must fall back (server/discover is the probe), and results from older servers lack `resultType`, which clients treat as complete. Test against the versions you actually connect to.
- **Too many servers.** Five servers can put dozens of tools and tens of thousands of tokens of definitions into every call. Load tools on demand with tool search.
- **Building on deprecated features.** Sampling, roots and logging still work but are scheduled for removal under the spec's twelve-month deprecation policy. Don't start new servers on them.

<div class="take">
<div class="lab">Key takeaway</div>

MCP is a JSON-RPC protocol between AI apps and servers that expose tools, resources and prompts, turning M x N integrations into M + N. Since 2026-07-28 every request is self-contained: version and capabilities ride in `_meta`, tools can return schema-checked structured content, user input arrives through input_required results, and the registry tells you who published a server, not whether to trust it.

</div>
