package llm

import "fmt"

// NewSystemMessage creates a system message to set up the LLM's behavior.
// This is typically the first message in the conversation and sets the context
// for how the assistant should respond.
func NewSystemMessage(content string) Message {
	return Message{
		Role:    "system",
		Content: content,
	}
}

// NewUserMessage creates a message from the user.
// Use this to send user queries to the LLM.
func NewUserMessage(content string) Message {
	return Message{
		Role:    "user",
		Content: content,
	}
}

// NewAssistantMessage creates a message from the LLM.
// Use this when you receive a response from the API and want to add it
// to the conversation history.
func NewAssistantMessage(content string) Message {
	return Message{
		Role:    "assistant",
		Content: content,
	}
}

// NewToolCallMessage creates an assistant message containing tool calls.
// The LLM sends these when it decides to use tools instead of responding directly.
// You probably won't create these yourself - you receive them from the API.
// Content must be empty when ToolCalls are present (OpenAI requirement).
func NewToolCallMessage(calls []ToolCall) Message {
	return Message{
		Role:      "assistant",
		ToolCalls: calls,
		// Content must be empty for tool calls in strict OpenAI standards
	}
}

// NewToolResult creates a message containing the result of a tool execution.
// After the LLM requests a tool call, you execute the tool and send back
// the result using this function. The ToolCallID is crucial - it must match
// the ID from the ToolCall that requested this execution, otherwise the LLM
// won't know which tool result corresponds to which request.
//
// The name parameter is the function name (e.g. "get_weather"). Some providers
// (Gemini) require it in the tool result to link calls and responses.
func NewToolResult(toolCallID string, name string, output string) Message {
	return Message{
		Role:       "tool",
		ToolCallID: toolCallID,
		Name:       name,
		Content:    output,
	}
}

// NewToolError creates a message indicating a tool failed to execute.
// Use this when Execute returns an error - it formats the error nicely
// and tells the LLM to fix its arguments.
func NewToolError(toolCallID string, name string, err error) Message {
	return Message{
		Role:       "tool",
		ToolCallID: toolCallID,
		Name:       name,
		Content:    fmt.Sprintf("Error executing tool: %v. Please fix your arguments.", err),
	}
}
