package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"go-agent-sdk/agent"
	"go-agent-sdk/llm"
)

// Simple chat example - demonstrates the most basic SDK usage.
// This creates an agent and sends a single message without any tools.

func main() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("Set OPENROUTER_API_KEY environment variable")
	}

	client := llm.NewClient(apiKey)

	// Create agent with a system prompt and retry configuration.
	myAgent := agent.New(client, "google/gemini-3-flash-preview",
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
