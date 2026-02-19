package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"go-agent-sdk/llm"
)

// Base URLs for known OpenAI-compatible services.
// Pass any of these to [WithBaseURL] — they all speak the same chat
// completions format, so our common types work directly with no translation.
const (
	DefaultBaseURL    = "https://api.openai.com/v1"
	OpenRouterBaseURL = "https://openrouter.ai/api/v1"

	// Inference providers
	GroqBaseURL      = "https://api.groq.com/openai/v1"
	CerebrasBaseURL  = "https://api.cerebras.ai/v1"
	FireworksBaseURL = "https://api.fireworks.ai/inference/v1"
	TogetherBaseURL  = "https://api.together.xyz/v1"
	AnyscaleBaseURL  = "https://api.endpoints.anyscale.com/v1"

	// Model-specific providers
	DeepSeekBaseURL  = "https://api.deepseek.com/v1"
	MistralBaseURL   = "https://api.mistral.ai/v1"
	MoonshotBaseURL  = "https://api.moonshot.ai/v1"
	DashScopeBaseURL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	ZAIBaseURL       = "https://api.z.ai/v1"
)

// Client implements llm.ChatProvider for OpenAI and any OpenAI-compatible
// API (OpenRouter, Azure OpenAI, Ollama, vLLM, etc.).
//
// Since our common types (llm.ChatRequest, llm.ChatResponse) already follow
// the OpenAI chat completions format, this provider is a thin HTTP wrapper —
// it marshals the request directly and unmarshals the response directly.
// No field translation needed.
//
// Use the functional options (WithBaseURL, WithHTTPClient) to configure
// the client for different endpoints.
type Client struct {
	apiKey     string
	model      string
	baseURL    string
	httpClient *http.Client
}

// Option is a function that configures a Client.
// These are called "functional options" — they let you customize the client
// without a sprawling constructor or config struct.
//
// Example:
//
//	client := openai.New(key, "gpt-4o",
//	    openai.WithBaseURL(openai.OpenRouterBaseURL),
//	    openai.WithHTTPClient(customClient),
//	)
type Option func(*Client)

// WithBaseURL overrides the default API base URL.
// Use this to point at OpenRouter, Azure, Ollama, or any OpenAI-compatible endpoint.
//
// Common values:
//   - openai.DefaultBaseURL (https://api.openai.com/v1) — default
//   - openai.OpenRouterBaseURL (https://openrouter.ai/api/v1)
//   - "http://localhost:11434/v1" for Ollama
//   - "https://{resource}.openai.azure.com/openai/deployments/{model}" for Azure
func WithBaseURL(url string) Option {
	return func(c *Client) {
		c.baseURL = url
	}
}

// WithHTTPClient overrides the default HTTP client.
// Use this to configure timeouts, proxies, TLS settings, or connection pooling.
//
// Example — set a transport-level timeout:
//
//	client := openai.New(key, "gpt-4o",
//	    openai.WithHTTPClient(&http.Client{
//	        Timeout: 60 * time.Second,
//	    }),
//	)
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		c.httpClient = hc
	}
}

// New creates an OpenAI-compatible provider.
// By default it points at api.openai.com. Use WithBaseURL to change the endpoint.
//
// Examples:
//
//	// Direct OpenAI
//	provider := openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-4o")
//
//	// OpenRouter (any model)
//	provider := openai.New(os.Getenv("OPENROUTER_API_KEY"), "google/gemini-3-flash-preview",
//	    openai.WithBaseURL(openai.OpenRouterBaseURL),
//	)
//
//	// Local Ollama
//	provider := openai.New("", "llama3",
//	    openai.WithBaseURL("http://localhost:11434/v1"),
//	)
func New(apiKey string, model string, opts ...Option) *Client {
	c := &Client{
		apiKey:     apiKey,
		model:      model,
		baseURL:    DefaultBaseURL,
		httpClient: &http.Client{},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// NewOpenRouter is a convenience constructor for OpenRouter.
// Equivalent to New(apiKey, model, WithBaseURL(OpenRouterBaseURL)).
func NewOpenRouter(apiKey string, model string, opts ...Option) *Client {
	// Prepend the base URL option so user opts can still override it
	allOpts := append([]Option{WithBaseURL(OpenRouterBaseURL)}, opts...)
	return New(apiKey, model, allOpts...)
}

// ModelName returns the model identifier this client was configured with.
func (c *Client) ModelName() string {
	return c.model
}

// CreateChat sends a chat completion request to the OpenAI-compatible endpoint.
//
// Since our common types already match the OpenAI wire format, this method
// is straightforward:
//  1. Marshal the ChatRequest to JSON (it's already the right shape)
//  2. POST to {baseURL}/chat/completions with Bearer auth
//  3. Read the response body (including on errors, for better diagnostics)
//  4. Unmarshal into ChatResponse (it's already the right shape)
//
// No field translation is needed — this is the advantage of using OpenAI's
// format as the common protocol.
func (c *Client) CreateChat(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	// basic marshal with error handling
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("openai: failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/chat/completions", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("openai: failed to create HTTP request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read the full body so we can include it in error messages.
	// The old client discarded error bodies, which made debugging painful.
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai: unexpected status %d: %s", resp.StatusCode, string(body))
	}

	var chatResp llm.ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("openai: failed to decode response: %w", err)
	}

	return &chatResp, nil
}
