package agent

import (
	"encoding/json"
	"fmt"
	"go-agent-sdk/llm"
	"time"
)

// Callback lets you observe what happens inside the agent during Run().
// Implement this interface to see the raw data at each step - the JSON
// going to the LLM, what comes back, which tools get called, and what
// they return.
//
// The agent checks if a callback is set (not nil) before calling any
// of these methods. If you don't set one, nothing happens - zero overhead.
//
// There are 4 moments the agent reports on:
//   - OnLLMRequest: right before we send the request to the LLM provider
//   - OnLLMResponse: right after we get the response back
//   - OnToolCall: when the LLM asks us to run a tool, before we run it
//   - OnToolResult: after the tool finishes, with the result or error
//
// You don't need to build your own - use DebugCallback for raw JSON output.
// If you want custom behavior (metrics, file logging, etc), implement
// this interface on your own struct.
type Callback interface {
	OnLLMRequest(req llm.ChatRequest)
	OnLLMResponse(resp llm.ChatResponse, latency time.Duration)
	OnToolCall(name string, args string)
	OnToolResult(name string, result string, err error, latency time.Duration)
}

// DebugCallback is a built-in Callback that prints the raw JSON at every step.
// It uses json.MarshalIndent so the output is human-readable in your terminal.
//
// This is the quickest way to see what's happening inside the agent.
// Just pass it when creating the agent:
//
//	a := agent.New(provider, agent.WithCallback(&agent.DebugCallback{}))
//
// When the agent runs, you'll see the full ChatRequest JSON before each LLM call,
// the full ChatResponse JSON after, and every tool call with its arguments and result.
type DebugCallback struct{}

// OnLLMRequest prints the full ChatRequest as indented JSON.
// This shows you exactly what we're sending to the LLM provider - the model,
// all messages in the conversation history, all registered tools, and
// the temperature setting.
func (d *DebugCallback) OnLLMRequest(req llm.ChatRequest) {
	data, _ := json.MarshalIndent(req, "", "  ")
	fmt.Printf("[DEBUG] LLM Request:\n%s\n\n", string(data))
}

// OnLLMResponse prints the full ChatResponse as indented JSON.
// This shows what the LLM provider sent back - the finish_reason (did it want
// to call tools or is it a final answer?), the message content or tool_calls
// array, and token usage for cost tracking.
func (d *DebugCallback) OnLLMResponse(resp llm.ChatResponse, latency time.Duration) {
	data, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Printf("[DEBUG] LLM Response [%s]:\n%s\n\n", latency, string(data))
}

// OnToolCall prints which tool the LLM wants to call and with what arguments.
// The args string is raw JSON straight from the LLM - you can see exactly
// what it generated, including any mistakes (wrong field names, extra fields, etc).
func (d *DebugCallback) OnToolCall(name string, args string) {
	fmt.Printf("[DEBUG] Tool Call: %s\n   Args: %s\n\n", name, args)
}

// OnToolResult prints what the tool returned after execution.
// If the tool errored, you'll see the error. Otherwise you'll see the result
// string and how long the tool took to run.
func (d *DebugCallback) OnToolResult(name string, result string, err error, latency time.Duration) {
	if err != nil {
		fmt.Printf("[DEBUG] Tool Error: %s - %v [%s]\n\n", name, err, latency)
	} else {
		fmt.Printf("[DEBUG] Tool Result: %s - %s [%s]\n\n", name, result, latency)
	}
}
