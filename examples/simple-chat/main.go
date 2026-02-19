package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"go-agent-sdk/agent"
	"go-agent-sdk/llm/openai"
	// "go-agent-sdk/llm/anthropic"
	// "go-agent-sdk/llm/gemini"
)

// Simple chat example — the most basic SDK usage.
// Creates an agent and sends a single message without any tools.

func main() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("Set OPENROUTER_API_KEY environment variable")
	}

	// Pick your provider (uncomment one). See README for the full list.
	provider := openai.NewOpenRouter(apiKey, "google/gemini-3-flash-preview")
	// provider := openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-4o")
	// provider := anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-20250514")
	// provider := gemini.New(os.Getenv("GEMINI_API_KEY"), "gemini-2.5-flash")
	// provider := openai.New(os.Getenv("GROQ_API_KEY"), "llama-3.3-70b-versatile", openai.WithBaseURL(openai.GroqBaseURL))

	// Create agent with a system prompt and retry configuration.
	myAgent := agent.New(provider,
		agent.WithSystemPrompts("You are a helpful assistant. Keep responses concise."),
		agent.WithMaxRetries(3),
		agent.WithCallback(&agent.DebugCallback{}), // remove this if you do not want detailed json output
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	reply, err := myAgent.Run(ctx, "Explain goroutines in Go in 2 sentences.")
	if err != nil {
		log.Fatalf("Failed: %v", err)
	}
	fmt.Printf("Agent: %s\n", reply)
}
