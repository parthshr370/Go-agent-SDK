# Go Agent SDK

A minimal, high-performance Go SDK for building AI agents from first principles.

This project is a lightweight, type-safe framework for building LLM-powered agents in Go. It avoids heavy dependencies, using only `net/http`, `encoding/json`, and `reflect` to provide a clean, idiomatic Go experience.

## Features

- **Minimalistic Core**: Built using pure Go standard library (no official SDKs required).
- **Type-Safe Tool System**: Automatically generate JSON Schemas from Go structs using reflection.
- **Functional Options Pattern**: Clean and extensible agent configuration.
- **Multi-Turn Conversations**: Automatic history management for seamless back-and-forth interactions.
- **Robust Tool Calling Loop**: Handles parallel tool calls, errors, and recursive LLM responses.
- **OpenRouter Integration**: Ready to use with any model supported by OpenRouter (OpenAI, Gemini, Claude, etc.).

## Architecture

The SDK is organized into three distinct layers:

1.  **LLM Provider (`llm/`)**: Handles the raw API communication, type-safe requests/responses, and message factories.
2.  **Tool System (`tools/`)**: Manages tool registration, JSON Schema generation, and the reflection-based execution pipeline.
3.  **Agent Orchestrator (`agent/`)**: The main interaction loop that coordinates between the user, the LLM, and the tool registry.

## Quick Start

### Installation

```bash
go get github.com/yourusername/go-agent-sdk
```

### Basic Usage

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"go-agent-sdk/agent"
	"go-agent-sdk/llm"
)

func main() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	client := llm.NewClient(apiKey)

	// Create a new agent
	chatAgent := agent.New(client, "google/gemini-3-flash-preview",
		agent.WithSystemPrompts("You are a helpful assistant."),
	)

	// Run the agent
	ctx := context.Background()
	reply, err := chatAgent.Run(ctx, "Explain Go in one sentence.")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Agent: %s\n", reply)
}
```

### Adding Tools

Tools are just regular Go functions. The framework uses reflection to map LLM arguments to your struct fields.

```go
type WeatherArgs struct {
	City string `json:"city" description:"The city name"`
}

func GetWeather(args WeatherArgs) string {
	return fmt.Sprintf("It's sunny in %s!", args.City)
}

func main() {
    // ... setup agent ...
    agent.RegisterTool("get_weather", "Get current weather", GetWeather)
    
    reply, err := agent.Run(ctx, "What is the weather in London?")
    // Agent will automatically call GetWeather and incorporate the result!
}
```

## Project Structure

- `agent/`: Core `Agent` orchestrator and interaction loop.
- `llm/`: API client, message types, and OpenRouter integration.
- `tools/`: Tool registry and reflection-based execution.
- `tools/jsonschema/`: Automatic JSON Schema generation from Go structs.
- `examples/`: Standalone examples for various use cases.

## License

MIT License - see [LICENSE](LICENSE) for details.
