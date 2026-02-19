package agent

import (
	"context"
	"fmt"
	"go-agent-sdk/llm"
	"go-agent-sdk/tools"
	"time"
)

// Agent is the orchestrator that manages the conversation with an LLM.
// It handles message history, tool registration, and the main interaction loop.
//
// An Agent maintains state between calls - it remembers the conversation
// so you can have multi-turn interactions without resending everything.
//
// The agent depends on llm.ChatProvider (an interface), not on any concrete
// client. This lets you swap providers (OpenAI, Anthropic, Gemini, OpenRouter)
// without changing agent code.
type Agent struct {
	provider     llm.ChatProvider // Any LLM backend that implements ChatProvider
	SystemPrompt string           // Instructions for the LLM's behavior
	MaxRetries   int              // How many times to retry on failure
	History      []llm.Message    // The conversation so far
	tools        *tools.Registry  // Registered tools the LLM can call
	callback     Callback         // optional observer, fires at key moments during Run(). nil means silent.
}

// Option is a function that configures an Agent.
// This is the functional options pattern - it lets us have clean APIs
// with sensible defaults while still allowing customization.
type Option func(*Agent)

// New creates an Agent with the given provider.
// The provider implements llm.ChatProvider and determines which LLM backend
// is used (OpenAI, Anthropic, Gemini, OpenRouter, etc.).
// Additional options can be passed to customize behavior.
//
// Example - create an agent with OpenAI:
//
//	provider := openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-4o")
//	agent := agent.New(provider,
//	    agent.WithSystemPrompts("You are a helpful assistant"),
//	    agent.WithMaxRetries(3),
//	)
//
// Example - create an agent with OpenRouter:
//
//	provider := openai.NewOpenRouter(os.Getenv("OPENROUTER_API_KEY"), "google/gemini-3-flash-preview")
//	agent := agent.New(provider,
//	    agent.WithSystemPrompts("You are a helpful assistant"),
//	)
//
// The variadic opts parameter (...Option) means you can pass zero or more options.
// They're applied in order, so later options can override earlier ones.
func New(provider llm.ChatProvider, opts ...Option) *Agent {
	// Start with sensible defaults
	a := &Agent{
		provider:   provider,
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

// WithCallback attaches an observer to the agent's internal execution.
// When set, the agent calls the callback methods at key moments during Run() -
// before/after LLM calls and before/after tool executions.
// This is how you see the raw JSON flowing through the system.
//
// Pass nil or just don't use this option to keep the agent silent.
//
// Example - see everything:
//
//	a := agent.New(provider,
//	    agent.WithCallback(&agent.DebugCallback{}),
//	)
func WithCallback(cb Callback) Option {
	return func(a *Agent) {
		a.callback = cb
	}
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
//	LLM decides to call get_weather with {"city": "Paris"}
//	We execute get_weather - returns "Sunny, 22C"
//	We add the tool result to history, linked by tool_call_id
//	We recurse - call Run("") so the LLM sees the result
//	LLM sees the tool result and responds: "It's sunny and 22C in Paris!"
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
	// Tools must be included in EVERY request - most LLM providers validate
	// the tool schema on each call, even when the LLM is responding
	// to previous tool results.
	req := llm.ChatRequest{
		Model:       a.provider.ModelName(),
		Messages:    a.History,
		Tools:       a.tools.GetAllTools(),
		Temperature: 0.7, // Hardcoded for now - could make this configurable
	}

	// let the callback see the full request before we send it
	if a.callback != nil {
		a.callback.OnLLMRequest(req)
	}

	// track how long the LLM takes to respond
	start := time.Now()
	resp, err := a.provider.CreateChat(ctx, req)
	latency := time.Since(start)

	if err != nil {
		return "", fmt.Errorf("LLM call failed: %w", err)
	}

	// let the callback see the full response and how long it took
	if a.callback != nil {
		a.callback.OnLLMResponse(*resp, latency)
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

			// let the callback see which tool is about to run and what args the LLM sent
			if a.callback != nil {
				a.callback.OnToolCall(call.Function.Name, call.Function.Arguments)
			}

			// run the tool and track how long it takes
			toolStart := time.Now()
			result, err := a.tools.Execute(call.Function.Name, call.Function.Arguments)
			toolLatency := time.Since(toolStart)

			// let the callback see the outcome - result or error
			if a.callback != nil {
				a.callback.OnToolResult(call.Function.Name, result, err, toolLatency)
			}

			var toolMsg llm.Message
			if err != nil {
				// Tool execution failed - tell the LLM so it can try again or explain
				toolMsg = llm.NewToolError(call.ID, call.Function.Name, err)
			} else {
				// Success - send the result back with the matching tool_call_id
				toolMsg = llm.NewToolResult(call.ID, call.Function.Name, result)
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
