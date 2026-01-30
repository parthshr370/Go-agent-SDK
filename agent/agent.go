package agent

import (
	"context"
	"fmt"
	"go-agent-sdk/llm"
	"go-agent-sdk/tools"
)

// Agent is the orchestrator that manages the conversation with an LLM.
// It handles message history, tool registration, and the main interaction loop.
//
// An Agent maintains state between calls - it remembers the conversation
// so you can have multi-turn interactions without resending everything.
type Agent struct {
	client       *llm.Client     // Connection to OpenRouter
	SystemPrompt string          // Instructions for the LLM's behavior
	MaxRetries   int             // How many times to retry on failure
	Model        string          // Which model to use (e.g., "google/gemini-3-flash-preview")
	History      []llm.Message   // The conversation so far
	tools        *tools.Registry // Registered tools the LLM can call
}

// Option is a function that configures an Agent.
// This is the functional options pattern - it lets us have clean APIs
// with sensible defaults while still allowing customization.
type Option func(*Agent)

// New creates an Agent with the given client and model.
// Additional options can be passed to customize behavior.
//
// Example - create an agent with a system prompt:
//
//	agent := agent.New(client, "google/gemini-3-flash-preview",
//	    agent.WithSystemPrompts("You are a helpful assistant"),
//	    agent.WithMaxRetries(3),
//	)
//
// The variadic opts parameter (...Option) means you can pass zero or more options.
// They're applied in order, so later options can override earlier ones.
func New(client *llm.Client, model string, opts ...Option) *Agent {
	// Start with sensible defaults
	a := &Agent{
		client:     client,
		Model:      model,
		MaxRetries: 1,
		History:    make([]llm.Message, 0),
		tools:      tools.NewRegistry(),
	}

	// Apply each option to customize the agent
	// The _ ignores the index, we only care about the option function itself
	for _, opt := range opts {
		opt(a) // opt is a function that modifies the agent
	}

	// If a system prompt was provided, add it as the first message
	if a.SystemPrompt != "" {
		a.History = append(a.History, llm.NewSystemMessage(a.SystemPrompt))
	}

	return a
}

// WithSystemPrompts sets the system prompt for the agent.
// The system prompt guides the LLM's behavior and personality.
// It's automatically added as the first message in the history.
func WithSystemPrompts(prompt string) Option {
	return func(a *Agent) {
		a.SystemPrompt = prompt
	}
}

// WithMaxRetries sets how many times to retry failed requests.
// This is useful for handling temporary network issues or rate limits.
func WithMaxRetries(n int) Option {
	return func(a *Agent) {
		a.MaxRetries = n
	}
}

// RegisterTool adds a function that the LLM can call.
// The function must take a single struct argument with JSON tags
// and return a string (or something convertible to string).
//
// The struct's fields define what parameters the LLM should provide.
// For example:
//
//	type WeatherArgs struct {
//	    City string `json:"city" description:"The city name"`
//	}
//
//	func GetWeather(args WeatherArgs) string { ... }
//
//	agent.RegisterTool("get_weather", "Get current weather", GetWeather)
func (a *Agent) RegisterTool(name, description string, fn any) error {
	return a.tools.Register(name, description, fn)
}

// Run sends a message to the LLM and returns the response.
// It handles the full conversation flow including history management and tool execution.
//
// The flow has two branches based on the LLM's finish_reason:
//
// 1. Normal text response (finish_reason == "stop"):
//   - Append user message to history
//   - Send to LLM
//   - Get text response
//   - Add assistant message to history
//   - Return the text
//
// 2. Tool calling (finish_reason == "tool_calls"):
//   - Append user message to history
//   - Send to LLM (with tools available)
//   - LLM responds with tool_calls array instead of text
//   - Add assistant message containing the tool_calls to history (CRITICAL!)
//   - Execute each requested tool using our registry
//   - Add tool results to history with proper tool_call_id linkage
//   - Recurse: Call Run again with empty message so LLM sees results
//   - LLM generates final text response incorporating tool results
//   - Return final answer
//
// The recursion is key here - after executing tools, we call Run again
// with an empty user message. This lets the LLM "see" the tool results
// in the conversation history and generate a coherent response.
//
// Example tool calling flow:
//
//	User: "What's the weather in Paris?"
//	→ LLM: ToolCall{ID: "call_123", Name: "get_weather", Args: "{\"city\": \"Paris\"}"}
//	→ Execute: get_weather(city="Paris") → "Sunny, 22°C"
//	→ Add ToolResult{tool_call_id: "call_123", content: "Sunny, 22°C"} to history
//	→ Recurse
//	→ LLM: "It's sunny and 22°C in Paris right now!"
//
// Example:
//
//	reply, err := agent.Run(ctx, "What is the weather in Paris?")
func (a *Agent) Run(ctx context.Context, usrMsg string) (string, error) {

	// Only add user message if it's not empty.
	// Empty messages happen when we recurse after tool execution.
	if usrMsg != "" {
		userMessage := llm.NewUserMessage(usrMsg)
		a.History = append(a.History, userMessage)
	}

	// Build the chat request including all available tools.
	// Tools must be included in EVERY request - OpenRouter validates
	// the tool schema on each call, even when the LLM is responding
	// to previous tool results.
	req := llm.ChatRequest{
		Model:       a.Model,
		Messages:    a.History,
		Tools:       a.tools.GetAllTools(),
		Temperature: 0.7, // Hardcoded for now - could make this configurable
	}

	resp, err := a.client.CreateChat(ctx, req)
	if err != nil {
		return "", fmt.Errorf("LLM call failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("LLM returned no choices")
	}

	choice := resp.Choices[0]
	finishReason := choice.FinishReason

	// Branch 1: LLM wants to call tools
	if finishReason == "tool_calls" {
		// CRITICAL: Must add the assistant's tool_calls message to history FIRST.
		// The LLM needs to see its own request in the conversation context
		// when we recurse. Without this, the tool_call_ids won't make sense.
		assistantMsg := llm.NewToolCallMessage(choice.Message.ToolCalls)
		a.History = append(a.History, assistantMsg)

		// Execute each tool the LLM requested.
		// The LLM can request multiple tools in parallel (though we execute sequentially).
		for _, call := range choice.Message.ToolCalls {
			result, err := a.tools.Execute(call.Function.Name, call.Function.Arguments)

			var toolMsg llm.Message
			if err != nil {
				// Tool execution failed - tell the LLM so it can try again or explain
				toolMsg = llm.NewToolError(call.ID, err)
			} else {
				// Success - send the result back with the matching tool_call_id
				toolMsg = llm.NewToolResult(call.ID, result)
			}
			a.History = append(a.History, toolMsg)
		}

		// Recurse with empty message so the LLM sees the tool results.
		// The LLM will now generate a text response incorporating these results.
		return a.Run(ctx, "")
	}

	// Branch 2: Normal text response (finish_reason == "stop")
	if finishReason == "stop" {
		assistantContent := choice.Message.Content
		assistantMessage := llm.NewAssistantMessage(assistantContent)
		a.History = append(a.History, assistantMessage)
		return assistantContent, nil
	}

	// Handle other finish reasons (should be rare but good to catch)
	return "", fmt.Errorf("unexpected finish_reason: %s", finishReason)
}
