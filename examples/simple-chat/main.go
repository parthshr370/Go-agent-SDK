package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"go-agent-sdk/agent"
	"go-agent-sdk/llm/gemini"
	// "go-agent-sdk/llm/anthropic"
	// "go-agent-sdk/llm/gemini"
)

// Simple chat example — the most basic SDK usage.
// Creates an agent and sends a single message without any tools.

func main() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("Set GEMINI_API_KEY= environment variable")
	}

	// Pick your provider (uncomment one). See README for the full list.
	//provider := openai.NewOpenRouter(apiKey, "z-ai/glm-5")
	// provider := openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-5.4-mini-2026-03-17")
	//provider := anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-6")
	provider := gemini.New(os.Getenv("GEMINI_API_KEY"), "gemini-3-flash-preview")
	// provider := openai.New(os.Getenv("GROQ_API_KEY"), "llama-3.3-70b-versatile", openai.WithBaseURL(openai.GroqBaseURL))

	// Create agent with a system prompt and retry configuration.
	myAgent := agent.New(provider,
		agent.WithSystemPrompts("You are a helpful assistant. Keep responses concise."),
		agent.WithMaxRetries(3),
		agent.WithCallback(&agent.DebugCallback{}), // remove this if you do not want detailed json output
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	reply, err := myAgent.Run(ctx, "Explain tiramasu recipe in detailss make it a kanye west song.")
	if err != nil {
		log.Fatalf("Failed: %v", err)
	}
	fmt.Printf("Agent: %s\n", reply)
}
