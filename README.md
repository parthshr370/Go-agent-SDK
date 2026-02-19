# Go Agent SDK

<img width="750" height="298" alt="Screenshot 2026-02-01 at 17-22-37 golang mascot - Google Search" src="https://github.com/user-attachments/assets/74b9a2f1-0c7d-4941-be64-7bf999d6c3b3" />

A minimal Go SDK for building AI agents from first principles. Zero external dependencies — just `net/http`, `encoding/json`, and `reflect`.

## Features

- **Multi-provider**: Swap between OpenAI, Anthropic, Gemini, or any OpenAI-compatible endpoint (OpenRouter, Ollama, Azure) by changing one line
- **Type-safe tools**: Register plain Go functions as tools — JSON Schema is generated automatically from your structs
- **Conversation memory**: Multi-turn history managed for you
- **Callback system**: Optional observer to see the raw JSON at every step (requests, responses, tool calls, results)
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
	provider := openai.NewOpenRouter(os.Getenv("OPENROUTER_API_KEY"), "google/gemini-3-flash-preview")
	// provider := openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-4o")
	// provider := anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-20250514")
	// provider := gemini.New(os.Getenv("GEMINI_API_KEY"), "gemini-2.5-flash")

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

## Provider Setup

Every provider implements `llm.ChatProvider` (two methods: `CreateChat` and `ModelName`). The agent depends on the interface, not on any concrete client.

**Native providers** (each has its own translation layer):

```go
// OpenAI / OpenRouter
provider := openai.New(apiKey, "gpt-4o")
provider := openai.NewOpenRouter(apiKey, "google/gemini-3-flash-preview")

// Anthropic
provider := anthropic.New(apiKey, "claude-sonnet-4-20250514")

// Gemini
provider := gemini.New(apiKey, "gemini-2.5-flash")
```

**OpenAI-compatible services** — many providers speak the same wire format. Use `openai.New` with `WithBaseURL` and your provider's API key:

```go
// Groq (fast inference)
provider := openai.New(apiKey, "llama-3.3-70b-versatile", openai.WithBaseURL(openai.GroqBaseURL))

// DeepSeek
provider := openai.New(apiKey, "deepseek-chat", openai.WithBaseURL(openai.DeepSeekBaseURL))

// Together AI
provider := openai.New(apiKey, "meta-llama/Llama-3-70b-chat-hf", openai.WithBaseURL(openai.TogetherBaseURL))

// Local Ollama
provider := openai.New("", "llama3", openai.WithBaseURL("http://localhost:11434/v1"))
```

Some commonly used base URL constants:

| Constant | URL |
|----------|-----|
| `openai.DefaultBaseURL` | `https://api.openai.com/v1` |
| `openai.OpenRouterBaseURL` | `https://openrouter.ai/api/v1` |
| `openai.CerebrasBaseURL` | `https://api.cerebras.ai/v1` |
| `openai.ZAIBaseURL` | `https://api.z.ai/v1` |
| `openai.DeepSeekBaseURL` | `https://api.deepseek.com/v1` |

There are more (Groq, Fireworks, Together, Mistral, Moonshot, DashScope, Anyscale) — see [`llm/openai/client.go`](llm/openai/client.go) for the full list. Any URL can also be passed directly as a string to `WithBaseURL`.

## Debug Logging

Pass `DebugCallback` to see the full JSON at every step:

```go
a := agent.New(provider,
	agent.WithCallback(&agent.DebugCallback{}),
)
```

## Project Structure

```
llm/
├── provider.go          # ChatProvider interface (the contract)
├── types.go             # Common request/response types (OpenAI-shaped)
├── messages.go          # Message constructors
├── openai/client.go     # OpenAI + OpenRouter provider
├── anthropic/client.go  # Anthropic provider (full translation layer)
└── gemini/client.go     # Gemini provider (full translation layer)
agent/
├── agent.go             # Run() loop, depends on ChatProvider
└── callback.go          # Observer pattern
tools/
├── registry.go          # Tool registration
├── execution.go         # Reflection-based tool execution
└── jsonschema/schema.go # Struct-to-JSON-Schema generator
```

## License

MIT License — see [LICENSE](LICENSE) for details.
