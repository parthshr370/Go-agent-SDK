package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"go-agent-sdk/agent"
	"go-agent-sdk/llm/openai"
	// "go-agent-sdk/llm/anthropic"
	// "go-agent-sdk/llm/gemini"
)

// WeatherArgs defines what the LLM needs to provide to get weather info.
type WeatherArgs struct {
	City string `json:"city" description:"The city to get weather for"`
}

// GetWeather is a mock weather tool - in a real app this would call a weather API.
func GetWeather(args WeatherArgs) string {
	weather := map[string]string{
		"paris":    "Sunny, 22C",
		"london":   "Overcast, 14C",
		"tokyo":    "Clear skies, 28C",
		"new york": "Partly cloudy, 18C",
		"mumbai":   "Humid, 33C",
	}

	city := strings.ToLower(args.City)
	if w, ok := weather[city]; ok {
		return fmt.Sprintf("Weather in %s: %s", args.City, w)
	}
	return fmt.Sprintf("Weather in %s: No data available", args.City)
}

// CalculatorArgs defines parameters for basic math operations.
type CalculatorArgs struct {
	Operation string  `json:"operation" description:"The math operation: add, subtract, multiply, divide"`
	A         float64 `json:"a" description:"First number"`
	B         float64 `json:"b" description:"Second number"`
}

// Calculate performs basic math operations.
func Calculate(args CalculatorArgs) string {
	var result float64
	switch strings.ToLower(args.Operation) {
	case "add":
		result = args.A + args.B
	case "subtract":
		result = args.A - args.B
	case "multiply":
		result = args.A * args.B
	case "divide":
		if args.B == 0 {
			return "Error: division by zero"
		}
		result = args.A / args.B
	default:
		return fmt.Sprintf("Unknown operation: %s", args.Operation)
	}
	return fmt.Sprintf("%.2f %s %.2f = %.2f", args.A, args.Operation, args.B, result)
}

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
	// provider := openai.New(os.Getenv("DEEPSEEK_API_KEY"), "deepseek-chat", openai.WithBaseURL(openai.DeepSeekBaseURL))

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	fmt.Println("=== Go Agent SDK Demo ===")
	fmt.Println()

	// Example 1: Simple chat without tools.
	fmt.Println("--- Example 1: Simple Chat ---")

	chatAgent := agent.New(provider,
		agent.WithSystemPrompts("You are a helpful assistant. Keep responses brief."),
	)

	reply, err := chatAgent.Run(ctx, "What is Go (the programming language) in one sentence?")
	if err != nil {
		log.Fatalf("Simple chat failed: %v", err)
	}
	fmt.Printf("Agent: %s\n\n", reply)

	// Example 2: Register tools so the LLM can call them when needed.
	fmt.Println("--- Example 2: Tool Calling ---")

	toolAgent := agent.New(provider,
		agent.WithSystemPrompts("You are a helpful assistant with access to weather data and a calculator. Use them when needed."),
	)

	toolAgent.RegisterTool("get_weather", "Get current weather for a city", GetWeather)
	toolAgent.RegisterTool("calculator", "Perform basic math operations", Calculate)

	reply, err = toolAgent.Run(ctx, "What's the weather like in Paris right now?")
	if err != nil {
		log.Fatalf("Tool call failed: %v", err)
	}
	fmt.Printf("Agent: %s\n\n", reply)

	// Example 3: Multi-turn conversation - the agent remembers previous messages.
	fmt.Println("--- Example 3: Multi-Turn Conversation ---")

	conversationAgent := agent.New(provider,
		agent.WithSystemPrompts("You are a helpful assistant. Keep responses brief."),
	)

	reply, err = conversationAgent.Run(ctx, "My name is Parth. Remember that.")
	if err != nil {
		log.Fatalf("Turn 1 failed: %v", err)
	}
	fmt.Printf("Turn 1 - Agent: %s\n\n", reply)

	reply, err = conversationAgent.Run(ctx, "What's my name?")
	if err != nil {
		log.Fatalf("Turn 2 failed: %v", err)
	}
	fmt.Printf("Turn 2 - Agent: %s\n\n", reply)

	// Example 4: Multiple tool selection - the LLM chooses which tool to use.
	fmt.Println("--- Example 4: Multiple Tool Selection ---")

	reply, err = toolAgent.Run(ctx, "What is 1337 multiplied by 42?")
	if err != nil {
		log.Fatalf("Calculator failed: %v", err)
	}
	fmt.Printf("Agent: %s\n\n", reply)

	fmt.Println("=== Demo Complete ===")
}
