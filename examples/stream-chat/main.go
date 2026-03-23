package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"go-agent-sdk/agent"
	"go-agent-sdk/llm/openai"
	// "go-agent-sdk/llm/openai"
	// "go-agent-sdk/llm/anthropic"
)

// Streaming chat example -- same as simple-chat but tokens print
// as they arrive instead of all at once. The DebugCallback's
// OnStreamToken method handles the real-time printing.

func main() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("Set GEMINI_API_KEY environment variable")
	}

	// Pick your provider (uncomment one). All three support streaming.
	//provider := gemini.New(apiKey, "gemini-3-flash-preview")
	provider := openai.NewOpenRouter(os.Getenv("OPENROUTER_API_KEY"), "z-ai/glm-5")
	// provider := openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-5.4-mini-2026-03-17")
	// provider := anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-6")

	myAgent := agent.New(provider,
		agent.WithSystemPrompts("You are a helpful assistant. Keep responses detailed."),
		agent.WithCallback(&agent.DebugCallback{}),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// RunStream works like Run but tokens appear in real time via OnStreamToken.
	// The full text is still returned at the end for conversation history.
	reply, err := myAgent.RunStream(ctx, "How to make tiramasu but as a kanye west song.")
	if err != nil {
		log.Fatalf("Failed: %v", err)
	}

	fmt.Printf("\n\nFull reply: %s\n", reply)
}
