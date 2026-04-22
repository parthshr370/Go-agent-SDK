package agent

import (
	"context"
	"fmt"
	"go-agent-sdk/llm"
	"go-agent-sdk/tools"
	"io"
	"time"
)

// agentBase holds all configuration and state that does not depend on the
// output type T. Option operates on this so it can stay non-generic --
// Go cannot infer type parameters for functions that only appear in the
// return type, so keeping Option as func(*agentBase) means all the With*
// helpers work with both New and NewTyped[T] without explicit type params.
type agentBase struct {
	provider     llm.ChatProvider // Any LLM backend that implements ChatProvider
	SystemPrompt string           // Instructions for the LLM's behavior
	MaxRetries   int              // How many times to retry on failure (also bounds output retries)
	History      []llm.Message    // The conversation so far
	tools        *tools.Registry  // Registered tools the LLM can call
	callback     Callback         // optional observer, fires at key moments during Run(). nil means silent.
}

// Agent is the orchestrator that manages the conversation with an LLM.
// It handles message history, tool registration, and the main interaction loop.
//
// An Agent maintains state between calls - it remembers the conversation
// so you can have multi-turn interactions without resending everything.
//
// The agent depends on llm.ChatProvider (an interface), not on any concrete
// client. This lets you swap providers (OpenAI, Anthropic, Gemini, OpenRouter)
// without changing agent code.
//
// T is the return type of Run and RunStream.
// Use New for plain text (T = string). Use NewTyped[T] for structured output.
type Agent[T any] struct {
	agentBase
	output outputSpec[T] // describes how to get T back from the LLM
}

// Option is a function that configures an Agent.
// This is the functional options pattern - it lets us have clean APIs
// with sensible defaults while still allowing customization.
type Option func(*agentBase)

// New creates an Agent with the given provider.
// The provider implements llm.ChatProvider and determines which LLM backend
// is used (OpenAI, Anthropic, Gemini, OpenRouter, etc.).
// Additional options can be passed to customize behavior.
//
// Example - create an agent with OpenAI:
//
//	provider := openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-5.4-mini-2026-03-17")
//	agent := agent.New(provider,
//	    agent.WithSystemPrompts("You are a helpful assistant"),
//	    agent.WithMaxRetries(3),
//	)
//
// Example - create an agent with OpenRouter:
//
//	provider := openai.NewOpenRouter(os.Getenv("OPENROUTER_API_KEY"), "z-ai/glm-5")
//	agent := agent.New(provider,
//	    agent.WithSystemPrompts("You are a helpful assistant"),
//	)
//
// The variadic opts parameter (...Option) means you can pass zero or more options.
// They're applied in order, so later options can override earlier ones.
func New(provider llm.ChatProvider, opts ...Option) *Agent[string] {
	return NewTyped[string](provider, opts...)
}

// NewTyped creates an Agent that returns T from Run and RunStream.
// At construction time it inspects T to decide the output mode:
//   - T is string → text mode, same behaviour as New, no schema needed
//   - T is a struct → tool mode, JSON Schema built from T's fields via
//     the same jsonschema package used for tool parameters
//
// Example:
//
//	type MovieInfo struct {
//	    Title  string `json:"title"`
//	    Year   int    `json:"year"`
//	}
//	a := agent.NewTyped[MovieInfo](provider, agent.WithSystemPrompts("..."))
//	movie, err := a.Run(ctx, "Tell me about Inception.")
func NewTyped[T any](provider llm.ChatProvider, opts ...Option) *Agent[T] {
	// Start with sensible defaults
	a := &Agent[T]{
		agentBase: agentBase{
			provider:   provider,
			MaxRetries: 1,
			History:    make([]llm.Message, 0),
			tools:      tools.NewRegistry(),
		},
		output: buildOutputSpec[T](),
	}

	// Apply each option to customize the agent
	// The _ ignores the index, we only care about the option function itself
	for _, opt := range opts {
		opt(&a.agentBase) // opt is a function that modifies the agent
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
	return func(a *agentBase) {
		a.SystemPrompt = prompt
	}
}

// WithMaxRetries sets how many times to retry failed requests.
// This is useful for handling temporary network issues or rate limits.
// The same budget is used for structured-output validation retries.
func WithMaxRetries(n int) Option {
	return func(a *agentBase) {
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
func (a *Agent[T]) RegisterTool(name, description string, fn any) error {
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
	return func(a *agentBase) {
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
func (a *Agent[T]) Run(ctx context.Context, usrMsg string) (T, error) {
	return a.run(ctx, usrMsg, 0)
}

// run is the internal implementation. outputRetry tracks how many times we have
// asked the LLM to fix its structured output -- it is separate from regular tool
// recursion and is bounded by MaxRetries so the loop cannot spin forever.
//
// Tool execution recursion always resets outputRetry to 0 because it is a fresh
// LLM round, not a retry of the same output. Output validation failure increments
// outputRetry and recurses. Once outputRetry >= MaxRetries the call fails fast.
func (a *Agent[T]) run(ctx context.Context, usrMsg string, outputRetry int) (T, error) {
	var zero T

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
	//
	// In tool mode the final_result output tool is also appended here so the
	// LLM knows to call it when it has a complete structured answer.
	req := llm.ChatRequest{
		Model:       a.provider.ModelName(),
		Messages:    a.History,
		Tools:       a.tools.GetAllTools(),
		Temperature: 0.7, // Hardcoded for now - could make this configurable
	}
	if a.output.mode == outputModeTool {
		req.Tools = append(req.Tools, a.output.buildOutputTool())
		// Force the model to call final_result when no other tools are registered.
		// With additional user tools, "auto" is safer -- it lets the model call its
		// regular tools first and then call final_result when it has a complete answer.
		if len(a.tools.GetAllTools()) == 0 {
			req.ToolChoice = map[string]any{
				"type":     "function",
				"function": map[string]any{"name": "final_result"},
			}
		}
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
		return zero, fmt.Errorf("LLM call failed: %w", err)
	}

	// let the callback see the full response and how long it took
	if a.callback != nil {
		a.callback.OnLLMResponse(*resp, latency)
	}

	if len(resp.Choices) == 0 {
		return zero, fmt.Errorf("LLM returned no choices")
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

		// Structured output intercept: check for final_result before executing
		// anything. If it's present its arguments ARE the structured output --
		// we unmarshal them into T and return instead of running a real function.
		for _, call := range choice.Message.ToolCalls {
			if call.Function.Name == "final_result" {
				result, err := validateAndUnmarshal[T](call.Function.Arguments)
				if err != nil {
					// Arguments were malformed JSON. Check retry budget before asking
					// the LLM to fix the structure and call final_result again.
					if outputRetry >= a.MaxRetries {
						return zero, fmt.Errorf("structured output failed after %d retries: %w", a.MaxRetries, err)
					}
					a.History = append(a.History, retryMessage(err))
					return a.run(ctx, "", outputRetry+1)
				}
				return result, nil
			}
		}

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
		// Reset outputRetry to 0 -- this is a new LLM round, not an output retry.
		return a.run(ctx, "", 0)
	}

	// Branch 2: Normal text response (finish_reason == "stop")
	if finishReason == "stop" {
		assistantContent := choice.Message.Content
		assistantMessage := llm.NewAssistantMessage(assistantContent)
		a.History = append(a.History, assistantMessage)

		// Text mode (T = string): return the content directly, same as before.
		if a.output.mode == outputModeText {
			result, ok := any(assistantContent).(T)
			if !ok {
				return zero, fmt.Errorf("text mode requires string output type")
			}
			return result, nil
		}

		// Tool mode but LLM returned plain text instead of calling final_result.
		// Try to unmarshal it as T in case it returned valid JSON directly.
		// If that also fails, ask it to use the tool and retry -- bounded by MaxRetries.
		if a.output.mode == outputModeTool {
			result, err := validateAndUnmarshal[T](assistantContent)
			if err != nil {
				if outputRetry >= a.MaxRetries {
					return zero, fmt.Errorf("structured output failed after %d retries: %w", a.MaxRetries, err)
				}
				a.History = append(a.History, llm.NewUserMessage(
					"Please call the final_result tool with your answer. Do not return plain text.",
				))
				return a.run(ctx, "", outputRetry+1)
			}
			return result, nil
		}
	}

	// Handle other finish reasons (should be rare but good to catch)
	return zero, fmt.Errorf("unexpected finish_reason: %s", finishReason)
}

// RunStream sends a message to the LLM and returns the response, streaming
// tokens as they arrive. It has the same return type as Run() -- the streaming
// is visible to the caller through the Callback.OnStreamToken method.
//
// The flow is the same as Run() with two differences:
//  1. Uses CreateChatStream() instead of CreateChat() to get a StreamReader
//  2. Reads events one at a time via Recv(), firing OnStreamToken for each text chunk
//
// Tool call rounds work identically to Run(): accumulate the full stream,
// check if the LLM requested tools, execute them, loop again. The outer
// loop uses iteration (not recursion like Run) because we're already
// managing state with the Accumulator.
//
// Structured-output retry parity: if final_result arguments are malformed,
// RunStream retries the same way Run does -- bounded by MaxRetries, with
// a retry message appended to history before the next iteration.
//
// The provider must implement llm.StreamProvider. If it doesn't (e.g.
// Anthropic, Gemini which don't have streaming yet), RunStream returns
// an error immediately.
//
// Example:
//
//	a := agent.New(provider, agent.WithCallback(&agent.DebugCallback{}))
//	reply, err := a.RunStream(ctx, "What is the weather in Paris?")
//	// tokens printed in real time via OnStreamToken, reply has the full text
func (a *Agent[T]) RunStream(ctx context.Context, usrMsg string) (T, error) {
	var zero T
	outputRetries := 0 // bounded retry counter for structured-output validation

	// Add user message to history once, before the loop.
	if usrMsg != "" {
		a.History = append(a.History, llm.NewUserMessage(usrMsg))
	}

	// Check that the provider supports streaming.
	sp, ok := a.provider.(llm.StreamProvider)
	if !ok {
		return zero, fmt.Errorf("provider %T does not support streaming", a.provider)
	}

	// Outer loop: one iteration per LLM call. If the LLM requests tools,
	// we execute them and go around again. If it returns text, we return.
	// The label lets final_result retry continue the outer loop directly.
outer:
	for {
		// Build a fresh request with the latest history (which now includes
		// any tool results from the previous iteration).
		req := llm.ChatRequest{
			Model:       a.provider.ModelName(),
			Messages:    a.History,
			Tools:       a.tools.GetAllTools(),
			Temperature: 0.7,
		}
		if a.output.mode == outputModeTool {
			req.Tools = append(req.Tools, a.output.buildOutputTool())
			// Mirror Run() tool_choice forcing: when no user tools are registered,
			// the model must call final_result. With extra tools, use auto.
			if len(a.tools.GetAllTools()) == 0 {
				req.ToolChoice = map[string]any{
					"type":     "function",
					"function": map[string]any{"name": "final_result"},
				}
			}
		}

		if a.callback != nil {
			a.callback.OnLLMRequest(req)
		}

		stream, err := sp.CreateChatStream(ctx, req)
		if err != nil {
			return zero, fmt.Errorf("LLM stream call failed: %w", err)
		}

		// Inner loop: read events from the stream one at a time.
		// Each Recv() blocks until the next SSE chunk arrives.
		var acc llm.Accumulator
		for {
			event, err := stream.Recv()
			if err == io.EOF {
				break
			}
			if err != nil {
				stream.Close()
				return zero, fmt.Errorf("stream read failed: %w", err)
			}

			acc.Add(event)

			// Fire callback on each text token so the caller sees them in real time.
			if event.Type == llm.EventText && a.callback != nil {
				a.callback.OnStreamToken(event.Text)
			}
		}
		stream.Close()

		// Stream is done. The Accumulator has the complete response.
		// Add it to history (same Message type as the non-streaming path).
		a.History = append(a.History, acc.Message())

		// Check if the LLM requested tool calls.
		if acc.HasToolCalls() {
			// Structured output intercept: same logic as run().
			// final_result arguments are the output -- unmarshal and return.
			// On bad JSON, retry up to MaxRetries times (mirrors run() behavior).
			for _, call := range acc.ToolCalls() {
				if call.Function.Name == "final_result" {
					result, err := validateAndUnmarshal[T](call.Function.Arguments)
					if err != nil {
						if outputRetries >= a.MaxRetries {
							return zero, fmt.Errorf("structured output failed after %d retries: %w", a.MaxRetries, err)
						}
						outputRetries++
						a.History = append(a.History, retryMessage(err))
						continue outer
					}
					return result, nil
				}
			}

			// Execute each tool, same as run().
			for _, call := range acc.ToolCalls() {
				if a.callback != nil {
					a.callback.OnToolCall(call.Function.Name, call.Function.Arguments)
				}

				toolStart := time.Now()
				result, toolErr := a.tools.Execute(call.Function.Name, call.Function.Arguments)
				toolLatency := time.Since(toolStart)

				if a.callback != nil {
					a.callback.OnToolResult(call.Function.Name, result, toolErr, toolLatency)
				}

				var toolMsg llm.Message
				if toolErr != nil {
					toolMsg = llm.NewToolError(call.ID, call.Function.Name, toolErr)
				} else {
					toolMsg = llm.NewToolResult(call.ID, call.Function.Name, result)
				}
				a.History = append(a.History, toolMsg)
			}

			// Go around the outer loop again -- the LLM will see the tool
			// results in history and (hopefully) respond with text this time.
			continue
		}

		// No tool calls -- text response.
		text := acc.Text()

		// Text mode (T = string): return as-is.
		if a.output.mode == outputModeText {
			result, ok := any(text).(T)
			if !ok {
				return zero, fmt.Errorf("text mode requires string output type")
			}
			return result, nil
		}

		// Tool mode but LLM returned plain text -- try unmarshal, else retry.
		// Bounded by MaxRetries, same contract as run().
		if a.output.mode == outputModeTool {
			result, err := validateAndUnmarshal[T](text)
			if err != nil {
				if outputRetries >= a.MaxRetries {
					return zero, fmt.Errorf("structured output failed after %d retries: %w", a.MaxRetries, err)
				}
				outputRetries++
				a.History = append(a.History, llm.NewUserMessage(
					"Please call the final_result tool with your answer. Do not return plain text.",
				))
				continue outer
			}
			return result, nil
		}

		return zero, fmt.Errorf("unexpected: no output produced")
	}
}
