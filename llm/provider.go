package llm

import "context"

// ChatProvider is the interface that all LLM providers must implement.
// The agent depends on this interface, not on any concrete client.
// This is what lets you swap between OpenAI, Anthropic, Gemini, or
// any OpenAI-compatible endpoint (OpenRouter, Ollama, Azure, etc.)
// without changing the agent code.
//
// Each provider is a self-contained adapter that translates between
// the common types in this package (ChatRequest, ChatResponse) and
// the provider's native API format.
//
// For providers that speak the OpenAI format natively (OpenAI, OpenRouter,
// Ollama), the translation is trivial — marshal and unmarshal directly.
// For others (Anthropic, Gemini), the provider maps request/response
// fields to the native format and back.
type ChatProvider interface {
	// CreateChat sends a chat completion request and returns the response.
	// Every provider translates the common ChatRequest into its native
	// format, calls its API, and translates the native response back
	// into a common ChatResponse.
	//
	// Context should be used for timeouts and cancellation.
	CreateChat(ctx context.Context, req ChatRequest) (*ChatResponse, error)

	// ModelName returns the model identifier this provider was configured with.
	// The agent uses this to set the Model field on ChatRequest, so individual
	// providers don't need to worry about it — the agent handles it.
	ModelName() string
}
