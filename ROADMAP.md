# Go Agent SDK Roadmap

This is the roadmap for the Go Agent SDK, split into ***two tracks***. One is ***personal***, the other is open to ***contributors***. I am still figuring out exactly what contributor workflows look like, but the idea is simple: I build the ***core features*** as part of my own ***learning***, and smaller pieces like tests, docs, and evals are where others can jump in.

## Why I am building this

I took a break from work and wanted to learn a language outside of Python. Something ***low level***. I was torn between ***Rust and Go***, and ***Go won***.

After finishing the basics, skimming docs, and reading enough code, I started seeing how functions and data flow through Go repos. That is when I realized I could actually ***build something real*** in it.

AI engineering is something I have been doing for a while. I worked at ***CAMEL AI*** and spent time studying frameworks like ***Pydantic AI*** and ***LangChain***, so I understand the internal gears of how agents work. Tool calling, structured output, streaming, provider abstractions. I have seen it all from the inside.

Then I noticed something. The whole ***cloud ecosystem runs on Go***. Backend services, infrastructure, deployment. But when it comes to agent frameworks, ***Python dominates completely***. Almost every agent endpoint is built in Python, then wrapped and shipped alongside Go services. That felt like a ***gap no one was filling***.

So I decided to build it ***from the ground up***. Learning streaming from SSE specs, tool calling from existing frameworks, structured output across multiple providers, all while writing it in Go. This repo is a ***public way of keeping myself accountable***. If I say I will build it, I have to build it.

## Two tracks

**Track 1: Core features till v1.** Built by me. This is part of my learning experience and I want to own the architectural decisions.

**Track 2: Tests, docs, evals, and post-v1 polish.** Contributors are welcome here. I will define the interfaces and contracts, and others can help fill in the gaps.

## Roadmap

### Phase 0: Hardening what exists

Before anything fancy, the foundation needs to be solid.

1. **Provider-specific cleanup from streaming.** Some issues were left behind when streaming shipped across OpenAI, Anthropic, and Gemini. I need to go back and tighten those.

2. **Native structured output.** Right now structured output works through a fake tool call. That is functional, but I also want native provider-level structured output where available.

3. **Expand ResponseFormat in llm/types.go.** The current type is too minimal. It needs to support json_schema and provider-specific shapes.

4. **Parallel tool execution.** When the LLM returns multiple tool calls, they currently run one by one. I want to see how the framework performs when tools run concurrently, especially for long-running tasks.

5. **Concurrency safety and robust testing.** The agent stores conversation history internally. I need to understand what happens when Run is called from multiple goroutines, and add tests that prove the behavior.

6. **Better error handling.** Tool errors and generic failures need clearer, more actionable responses instead of passing raw strings back to the model.

### Phase 1: Runtime maturity

7. **Dependency injection for tools.** Right now tools are stateless functions. I need a way for tools to access databases, HTTP clients, and user context without making everything global.

8. **Improved observability.** The callback system exists but could grow into something richer. Metrics, tracing, maybe middleware-style hooks.

### Phase 2: Graph orchestration (the v1 feature)

9. **Graph engine.** This is the big one. Replace the recursive loop with a proper graph model where agents are composable nodes. Nodes for model calls, tool calls, branching logic, validation, and endpoints. Edges connect them. Cycles are allowed for agent loops.

10. **Human in the loop.** Hooks that let the graph pause for human approval before continuing.

11. **Multi-agent patterns.** Examples and support for supervisors, delegation, and agents calling other agents as tools.

## What v1 means

Graph orchestration is the headline feature, but v1 itself means the core runtime is reliable enough to deserve it. That means stable provider abstraction, stable tool system, stable streaming, stable structured output, and enough tests and docs that someone else can actually use this thing.

## Contributor welcome

While the major features above are mine to build, there is plenty of room for contributors:

- Tests and test tooling
- Documentation and examples
- Benchmarks
- Provider-specific polish
- Evaluation frameworks

If you are interested, open an issue or draft a PR. I will review it.

## Learning in public

I am documenting my learning journey through each major release. If you want to understand how the SDK was built from the ground up, these posts walk through the internals:

- [How I built my own Langchain from scratch in Go](https://parthshr370.github.io/blogs/how-i-built-my-own-langchain-from-scratch-in-go/) - Part 1: message structs, history as an append-only array, and how marshalling bridges Go types and JSON
- [How I built a tool calling engine for my Go agent SDK](https://parthshr370.github.io/blogs/how-i-built-tool-calling-engine-for-my-agent-sdk/) - Part 2: registration, schema generation via reflection, execution, and how tool results flow back to the LLM

More posts will follow as structured output, streaming, and graph orchestration ship.

---

## Feature checklist

Track what is promised and when it ships.

- [ ] Provider-specific streaming cleanup
- [ ] Native structured output (OpenAI / Gemini)
- [ ] Expand ResponseFormat type
- [ ] Parallel tool execution
- [ ] Concurrency safety and tests
- [ ] Better error handling for tools
- [ ] Dependency injection for tools
- [ ] Improved observability and middleware hooks
- [ ] Graph orchestration engine
- [ ] Human in the loop support
- [ ] Multi-agent patterns and examples
