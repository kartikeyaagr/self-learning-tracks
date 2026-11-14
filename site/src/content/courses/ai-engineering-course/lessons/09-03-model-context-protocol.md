---
id: "09.03"
course: "ai-engineering-course"
module: "09"
slug: "model-context-protocol"
title: "Model Context Protocol"
keyIdeas: ["MCP is an open protocol for connecting AI applications to tools and data, introduced by Anthropic in November 2024.", "Without it, every app needs a custom integration for every data source: M apps x N tools. With MCP it becomes M + N.", "Architecture: a host (the AI app) runs one MCP client per server; servers expose capabilities over JSON-RPC.", "Server primitives: tools (actions the model can call), resources (data the app can read into context), prompts (reusable templates the user can pick).", "Transports: stdio for local servers, streamable HTTP for remote ones; remote servers use OAuth for authorization.", "Security: tool descriptions and outputs are untrusted input (prompt injection, tool poisoning); keep humans in the loop for sensitive actions and grant least privilege."]
question: "Explain the M x N integration problem MCP solves, and name MCP's three server-side primitives with an example of each."
answer: "Without a standard, each of M AI apps needs a custom integration for each of N services (M x N). MCP defines one protocol, so each service ships one server and each app one client implementation (M + N). Server primitives: tools (model-invoked actions, e.g. create_issue on GitHub), resources (app-controlled data read into context, e.g. a file or DB schema), and prompts (user-selected templates, e.g. a 'summarise this PR' workflow)."
sources: [{"title": "Model Context Protocol: Introduction", "url": "https://modelcontextprotocol.io/introduction", "note": "Official overview: hosts, clients, servers."}, {"title": "MCP specification", "url": "https://modelcontextprotocol.io/specification/2025-06-18", "note": "Protocol details: JSON-RPC, primitives, transports, authorization."}, {"title": "Anthropic: Introducing the Model Context Protocol", "url": "https://www.anthropic.com/news/model-context-protocol", "note": "Why it was created (November 2024)."}]
tools: ["MCP"]
---

## The integration problem

Every AI application wants access to the same things: files, GitHub, Slack, databases, calendars. Before MCP, each app wrote its own connector for each service. With M apps and N services that's M x N integrations, each with its own quirks. Anthropic introduced the Model Context Protocol in November 2024 as an open standard so that a service is wrapped once as an MCP server and works with every MCP-capable app: M + N.

## Architecture

~~~dg-flow One host, several servers, one client per server.
*Host (AI app) -(MCP client)-> GitHub server
Host (AI app) -(MCP client)-> Database server
Host (AI app) -(MCP client)-> Local files server
~~~

- **Host**: the AI application the user interacts with (a chat app, an IDE, an agent).
- **Client**: a connection inside the host, one per server.
- **Server**: a program that exposes capabilities, local or remote.

Messages are JSON-RPC 2.0. Local servers usually run as a subprocess over **stdio**; remote servers use **streamable HTTP**, with OAuth for authorization.

## What servers expose

| Primitive | Controlled by | What it is | Example |
| --- | --- | --- | --- |
| Tools | Model | Functions the model can call | create_issue, run_query |
| Resources | Application | Data the app can read into context | a file, a table schema |
| Prompts | User | Reusable templates the user selects | "review this PR" |

Clients can also offer capabilities to servers, for example **sampling** (the server asks the host's model to generate text) and **elicitation** (the server asks the user for input).

## A tool call through MCP

~~~dg-seq Discovering and calling a tool.
Host > Server: initialize (capabilities)
Host > Server: tools/list
Server -> Host: [create_issue: schema...]
Host > Server: tools/call create_issue {...}
Server -> Host: result: issue #42 created
~~~

The host puts the tool list in the model's context (lesson 08.03); when the model emits a tool call, the host routes it to the right server through its client.

## Security

MCP makes it easy to connect powerful tools, which raises the stakes:

- **Prompt injection**: a resource or tool result (an email, a web page, an issue comment) can contain instructions aimed at the model.
- **Tool poisoning**: a malicious server's tool descriptions can carry hidden instructions.
- **Over-broad permissions**: a server with full account access can do damage if the model is tricked.
Mitigations: install servers you trust, grant least privilege, show users what will be called, require confirmation for writes, and treat everything that comes back as data, not instructions.

## The integration count

Without a shared protocol, every AI application writes its own connector for every tool: N × M integrations. With one protocol, each application implements a client once and each tool implements a server once: N + M.

~~~dg-raw
%%widget mcpcount Integrations needed with custom connectors versus a shared protocol.
~~~

## What travels on the wire

MCP messages are JSON-RPC 2.0. After an initialisation handshake where client and server agree on a protocol version and capabilities, a tool call looks like this:

```
-> {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
<- {"jsonrpc": "2.0", "id": 1, "result": {"tools": [
     {"name": "get_issue", "description": "Fetch one GitHub issue",
      "inputSchema": {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]}}]}}

-> {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
    "params": {"name": "get_issue", "arguments": {"number": 812}}}
<- {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "#812 Login fails on Safari ..."}], "isError": false}}
```

The host application turns the tools/list result into tool definitions for the model, and when the model asks for get_issue, the client sends tools/call and passes the content back as the tool result.

Two transports are defined: **stdio** for local servers launched as a subprocess, and **Streamable HTTP** for remote servers, with OAuth for authorisation.

## Beyond tools

| Primitive | Controlled by | Example |
| --- | --- | --- |
| Tools | The model decides when to call | create_ticket, run_query |
| Resources | The application decides what to attach | A file, a database schema, a log |
| Prompts | The user picks from a menu | "Summarise this pull request" template |
| Sampling (client feature) | Server asks the host's model for a completion | A server that needs an LLM without its own API key |

## Security in practice

- **Only install servers you trust.** A server's tool descriptions go straight into the model's context and can contain hidden instructions ("tool poisoning").
- **Least privilege.** A server with a token that can delete repositories gives that power to whatever text the model reads.
- **Watch for the lethal combination**: access to private data, exposure to untrusted content, and a way to send data out. Any agent with all three can be steered by injected text into leaking data. Remove at least one.

<div class="take">
<div class="lab">Key takeaway</div>

MCP is a standard JSON-RPC protocol between AI apps (hosts and clients) and servers that expose tools, resources and prompts. It turns M x N integrations into M + N, and makes tool security a first-class concern.

</div>
