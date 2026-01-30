package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// Client manages the HTTP connection to OpenRouter.
// It stores your API key and provides a reusable HTTP client.
// You can create one Client and reuse it for all your requests.
type Client struct {
	APIKey     string       // Your OpenRouter API key
	BaseURL    string       // Usually https://openrouter.ai/api/v1
	HTTPClient *http.Client // Reusable HTTP client (can be customized for timeouts)
}

// NewClient creates a Client with your API key.
// It sets up sensible defaults:
//   - BaseURL: OpenRouter's API endpoint
//   - HTTPClient: Standard Go HTTP client
//
// Example:
//
//	client := llm.NewClient(os.Getenv("OPENROUTER_API_KEY"))
func NewClient(apikey string) *Client {
	return &Client{
		APIKey:     apikey,
		BaseURL:    "https://openrouter.ai/api/v1",
		HTTPClient: &http.Client{},
	}
}

// CreateChat sends a request to the LLM and returns the response.
// This is the main method you'll use - it handles all the HTTP plumbing
// so you can focus on the conversation.
//
// The flow:
//  1. Marshal your ChatRequest to JSON
//  2. Create an HTTP POST request to /chat/completions
//  3. Set authentication headers (Bearer token with your API key)
//  4. Send the request with context support (for timeouts/cancellation)
//  5. Check status code (anything other than 200 is an error)
//  6. Decode the JSON response into a ChatResponse struct
//
// Context is important here - you can use it to set timeouts:
//
//	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
//	defer cancel()
//	resp, err := client.CreateChat(ctx, req)
func (c *Client) CreateChat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {

	// Convert the request struct to JSON bytes that we can send over HTTP
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal chat request to JSON: %w", err)
	}

	// Build the HTTP request with context support
	// Context lets us cancel the request if it takes too long
	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.BaseURL+"/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)

	}

	// Set required headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)

	resp, err := c.HTTPClient.Do(httpReq)

	if err != nil {
		return nil, fmt.Errorf("failed to send HTTP request to OpenRouter: %w", err)

	}

	// Always close the response body to avoid resource leaks
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)

	}

	var chatResp ChatResponse
	// Decode the JSON response body into our ChatResponse struct
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}
	return &chatResp, nil
}
