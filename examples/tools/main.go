package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"go-agent-sdk/agent"
	"go-agent-sdk/llm"
)

// Tool calling example - shows how to register and use tools.
// The SDK auto-generates JSON Schema from your struct definitions.

// LookupArgs defines what the LLM needs to provide.
// The description tag becomes part of the schema sent to the LLM.
type LookupArgs struct {
	Topic string `json:"topic" description:"The topic to look up"`
}

// LookupFact is the actual tool function.
// It takes one struct argument and returns a string.
func LookupFact(args LookupArgs) string {
	facts := map[string]string{
		"go":     "Go was created at Google in 2009 by Robert Griesemer, Rob Pike, and Ken Thompson.",
		"rust":   "Rust was first released in 2010 and emphasizes memory safety without garbage collection.",
		"python": "Python was created by Guido van Rossum and first released in 1991.",
	}

	topic := strings.ToLower(args.Topic)
	if fact, ok := facts[topic]; ok {
		return fact
	}
	return fmt.Sprintf("No facts available for: %s", args.Topic)
}

func main() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("Set OPENROUTER_API_KEY environment variable")
	}

	client := llm.NewClient(apiKey)

	myAgent := agent.New(client, "google/gemini-3-flash-preview",
		agent.WithSystemPrompts("You are a helpful assistant with access to a fact database. Use the lookup tool when asked about programming languages."),
		agent.WithCallback(&agent.DebugCallback{}), // Uncomment this when you need to add verbose json logging on what is happening internally
	)

	// Register the tool - the framework introspects the function signature
	// and generates the JSON Schema automatically.
	myAgent.RegisterTool("lookup_fact", "Look up facts about programming languages", LookupFact)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// The LLM decides whether to use the tool based on the question.
	reply, err := myAgent.Run(ctx, "Tell me about the history of Go programming language.")
	if err != nil {
		log.Fatalf("Failed: %v", err)
	}
	fmt.Printf("Agent: %s\n", reply)
}
