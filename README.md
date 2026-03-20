# Go Agent SDK

<img width="750" height="298" alt="Screenshot 2026-02-01 at 17-22-37 golang mascot - Google Search" src="https://github.com/user-attachments/assets/74b9a2f1-0c7d-4941-be64-7bf999d6c3b3" />

A minimal Go SDK for building AI agents from first principles. Zero external dependencies — just `net/http`, `encoding/json`, and `reflect`.

## Features

- **Multi-provider**: Swap between OpenAI, Anthropic, Gemini, or any OpenAI-compatible endpoint by changing one line
- **Streaming**: Real-time token output via SSE — works with all three providers
- **Type-safe tools**: Register plain Go functions as tools — JSON Schema generated from structs automatically
- **Conversation memory**: Multi-turn history managed for you
- **Callback system**: Observer to see raw JSON at every step, plus streaming token callbacks
- **No dependencies**: Pure standard library, Go 1.24+

## Quick Start

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"go-agent-sdk/agent"
	"go-agent-sdk/llm/openai"
	// "go-agent-sdk/llm/anthropic"
	// "go-agent-sdk/llm/gemini"
)

func main() {
	// Pick your provider (uncomment one):
	provider := openai.NewOpenRouter(os.Getenv("OPENROUTER_API_KEY"), "z-ai/glm-5")
	// provider := openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-5.4-mini-2026-03-17")
	// provider := anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-6")
	// provider := gemini.New(os.Getenv("GEMINI_API_KEY"), "gemini-3-flash-preview")

	a := agent.New(provider,
		agent.WithSystemPrompts("You are a helpful assistant."),
	)

	reply, err := a.Run(context.Background(), "Explain Go in one sentence.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(reply)
}
```

## Adding Tools

Tools are regular Go functions. The SDK inspects your struct's `json` and `description` tags to generate the schema the LLM sees.

```go
type WeatherArgs struct {
	City string `json:"city" description:"The city name"`
}

func GetWeather(args WeatherArgs) string {
	return fmt.Sprintf("It's sunny in %s!", args.City)
}

// Register it on any agent:
a.RegisterTool("get_weather", "Get current weather", GetWeather)

reply, err := a.Run(ctx, "What is the weather in London?")
// The agent calls GetWeather automatically and incorporates the result.
```

## Streaming

`RunStream()` works like `Run()` but tokens print as they arrive. Same return type, same tool call handling -- the streaming is visible through the callback.

```go
a := agent.New(provider,
	agent.WithCallback(&agent.DebugCallback{}), // OnStreamToken prints each token
)

reply, err := a.RunStream(ctx, "Explain goroutines in 2 sentences.")
// Tokens appear in real time, reply has the complete text at the end.
```

All three providers (OpenAI, Anthropic, Gemini) support streaming. Tool call rounds run automatically between stream rounds -- the agent accumulates the stream, detects tool calls, executes them, and streams the next response.

## Providers

Every provider implements `llm.ChatProvider`. The agent depends on the interface, not on any concrete client.

| Provider | Constructor | Example Model |
|----------|------------|---------------|
| OpenAI | `openai.New(key, model)` | `gpt-5.4-mini-2026-03-17` |
| OpenRouter | `openai.NewOpenRouter(key, model)` | `z-ai/glm-5` |
| Anthropic | `anthropic.New(key, model)` | `claude-sonnet-4-6` |
| Gemini | `gemini.New(key, model)` | `gemini-3-flash-preview` |

Any OpenAI-compatible service works with `openai.New` + `WithBaseURL`:

```go
// Groq, DeepSeek, Cerebras, Together, Ollama, etc.
provider := openai.New(apiKey, "llama-3.3-70b-versatile", openai.WithBaseURL(openai.GroqBaseURL))
```

Built-in base URLs: Groq, Cerebras, DeepSeek, Fireworks, Together, Mistral, Moonshot, DashScope, ZAI, Anyscale. See `llm/openai/client.go` for the full list.

## Debug Logging

Pass `DebugCallback` to see the full JSON at every step:

```go
a := agent.New(provider,
	agent.WithCallback(&agent.DebugCallback{}),
)
```

## Project Structure

```
llm/           Provider abstraction and common types
├── openai/    OpenAI + OpenRouter + all compatible endpoints
├── anthropic/ Anthropic Claude (Messages API translation)
├── gemini/    Google Gemini (generateContent translation)
agent/         Orchestrator: Run() loop, RunStream(), callbacks
tools/         Reflection-based tool registration and execution
```

## License

MIT License — see [LICENSE](LICENSE) for details.
