// Package gemini implements llm.ChatProvider for the Google Gemini REST API.
//
// This provider translates between the common types (llm.ChatRequest,
// llm.ChatResponse) and Gemini's generateContent format. The key differences:
//
//   - System prompts go in a separate top-level "system_instruction" field
//   - Auth uses x-goog-api-key header (not Bearer token)
//   - Messages are "contents" with "parts" arrays, not role/content objects
//   - Roles: "assistant" becomes "model", "tool" becomes "user" with functionResponse
//   - Tool calls are "functionCall" parts, tool results are "functionResponse" parts
//   - Tool call args are a JSON object, not a JSON string (unlike OpenAI)
//   - Finish reason is ALWAYS "STOP" even for tool calls — we detect tool calls
//     by inspecting response parts for functionCall, not by finish reason
//   - Config (temperature, maxTokens, etc.) goes in a nested "generationConfig" object
package gemini

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go-agent-sdk/llm"
	"io"
	"net/http"
)

// geminiRequest is the top-level body for POST /v1beta/models/{model}:generateContent.
// System prompt is a top-level field (not in messages), messages become "contents"
// with only "user"/"model" roles, and generation params nest under "generationConfig".
type geminiRequest struct {
	Contents          []geminiContent    `json:"contents"`
	SystemInstruction *systemInstruction `json:"systemInstruction,omitempty"`
	Tools             []geminiTool       `json:"tools,omitempty"`
	GenerationConfig  *generationConfig  `json:"generationConfig,omitempty"`
}

// systemInstruction holds the system prompt as a top-level field.
// Gemini requires role to be "user" here (not "system").
type systemInstruction struct {
	Role  string  `json:"role"`
	Parts []gPart `json:"parts"`
}

// geminiContent is a single turn in the conversation.
// Only two roles exist: "user" and "model".
// Tool results (functionResponse) go inside role="user" content.
// Tool calls (functionCall) go inside role="model" content.
type geminiContent struct {
	Role  string  `json:"role"`
	Parts []gPart `json:"parts"`
}

// gPart is the union type for content parts. One content can mix text,
// functionCall, and functionResponse parts in the same array.
type gPart struct {
	Text             string             `json:"text,omitempty"`
	FunctionCall     *gFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *gFunctionResponse `json:"functionResponse,omitempty"`
}

// gFunctionCall is a tool invocation from the model.
// Args is a JSON object (map), not a string like OpenAI uses.
type gFunctionCall struct {
	Name string `json:"name"`
	Args any    `json:"args,omitempty"`
}

// gFunctionResponse is a tool result we send back to the model.
// Response must be an object (not a plain string). We wrap string results
// in {"return_value": "..."} and errors in {"error": "..."}.
type gFunctionResponse struct {
	Name     string `json:"name"`
	Response any    `json:"response"`
	ID       string `json:"id,omitempty"`
}

// geminiTool wraps function declarations.
// All functions go in a single functionDeclarations array.
type geminiTool struct {
	FunctionDeclarations []gFunctionDeclaration `json:"functionDeclarations"`
}

// gFunctionDeclaration describes a tool available to the model.
// Flatter than OpenAI's format: no "type":"function" wrapper, just name +
// description + parameters at the top level.
type gFunctionDeclaration struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters,omitempty"`
}

// generationConfig holds model configuration parameters.
// These are nested under a single object, not top-level like OpenAI.
type generationConfig struct {
	Temperature     float64  `json:"temperature,omitempty"`
	TopP            float64  `json:"topP,omitempty"`
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

// geminiResponse is the top-level response from generateContent.
// Candidates are like OpenAI's "choices" but each has a content with parts.
// The big gotcha: finishReason is "STOP" even for tool calls, so we can't
// rely on it to detect tool use — we check the parts instead.
type geminiResponse struct {
	Candidates    []geminiCandidate `json:"candidates"`
	UsageMetadata *geminiUsage      `json:"usageMetadata,omitempty"`
	ModelVersion  string            `json:"modelVersion,omitempty"`
}

// geminiCandidate is one possible completion (usually just one).
type geminiCandidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason"` // "STOP", "MAX_TOKENS", "SAFETY", etc.
	Index        int           `json:"index"`
}

// geminiUsage tracks token consumption.
// We map promptTokenCount to PromptTokens and totalTokenCount to TotalTokens.
// CompletionTokens includes both candidatesTokenCount and thoughtsTokenCount
// because thinking tokens (from Gemini 2.5+ models) are billed as output.
type geminiUsage struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
	ThoughtsTokenCount   int `json:"thoughtsTokenCount"`
}

const (
	// DefaultBaseURL for the Gemini REST API.
	// CreateChat appends /v1beta/models/{model}:generateContent.
	DefaultBaseURL = "https://generativelanguage.googleapis.com"
)

type Client struct {
	apiKey     string
	model      string
	baseURL    string
	httpClient *http.Client
}

type Option func(*Client)

// WithBaseURL overrides the default API base URL.
// Useful for proxies or Vertex AI endpoints.
func WithBaseURL(url string) Option {
	return func(c *Client) {
		c.baseURL = url
	}
}

// WithHTTPClient overrides the default HTTP client.
// Use this for custom timeouts, proxies, or TLS settings.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		c.httpClient = hc
	}
}

// New creates a Gemini provider.
//
// Example:
//
//	provider := gemini.New(os.Getenv("GEMINI_API_KEY"), "gemini-2.5-flash")
//	agent := agent.New(provider)
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

// ModelName returns the model identifier this client was configured with.
func (c *Client) ModelName() string {
	return c.model
}

// generateCallID creates a random ID for linking tool calls to tool results.
// Gemini doesn't reliably return IDs on functionCall, so we generate our own.
// The agent passes these through ToolCall.ID, then ToolCallID, then back here.
func generateCallID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "call_" + hex.EncodeToString(b)
}

// mapRequest translates our common llm.ChatRequest into Gemini's native format.
func mapRequest(req llm.ChatRequest) geminiRequest {

	var sysInst *systemInstruction
	var contents []geminiContent

	for _, msg := range req.Messages {
		switch msg.Role {

		case "system":
			// System prompt goes in the top-level systemInstruction field.
			// Multiple system messages get concatenated as separate parts.
			if sysInst == nil {
				sysInst = &systemInstruction{Role: "user"}
			}
			sysInst.Parts = append(sysInst.Parts, gPart{Text: msg.Content})

		case "user":
			contents = append(contents, geminiContent{
				Role:  "user",
				Parts: []gPart{{Text: msg.Content}},
			})

		case "assistant":
			if len(msg.ToolCalls) > 0 {
				// Assistant with tool calls becomes model content with functionCall parts.
				var parts []gPart

				if msg.Content != "" {
					parts = append(parts, gPart{Text: msg.Content})
				}

				for _, call := range msg.ToolCalls {
					// OpenAI Arguments is a JSON string, Gemini wants a JSON object.
					var argsObj any
					if err := json.Unmarshal([]byte(call.Function.Arguments), &argsObj); err != nil {
						argsObj = map[string]any{}
					}
					parts = append(parts, gPart{
						FunctionCall: &gFunctionCall{
							Name: call.Function.Name,
							Args: argsObj,
						},
					})
				}

				contents = append(contents, geminiContent{
					Role:  "model",
					Parts: parts,
				})
			} else {
				// Plain text assistant message becomes model content.
				contents = append(contents, geminiContent{
					Role:  "model",
					Parts: []gPart{{Text: msg.Content}},
				})
			}

		case "tool":
			// Tool results go in role="user" with functionResponse parts.
			// Gemini requires the response to be an object, not a plain string,
			// so we wrap it in {"return_value": "..."}.
			respObj := map[string]any{"return_value": msg.Content}

			contents = append(contents, geminiContent{
				Role: "user",
				Parts: []gPart{
					{
						FunctionResponse: &gFunctionResponse{
							Name:     msg.Name,
							Response: respObj,
							ID:       msg.ToolCallID,
						},
					},
				},
			})
		}
	}

	// Convert tools: unwrap OpenAI's {"type":"function","function":{...}} wrapper
	// into Gemini's flat functionDeclarations format.
	var tools []geminiTool
	if len(req.Tools) > 0 {
		var decls []gFunctionDeclaration
		for _, t := range req.Tools {
			decls = append(decls, gFunctionDeclaration{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  t.Function.Parameters,
			})
		}
		tools = append(tools, geminiTool{FunctionDeclarations: decls})
	}

	// Build generation config from request fields.
	var genConfig *generationConfig
	if req.Temperature != 0 || req.TopP != 0 || req.MaxTokens != 0 || len(req.Stop) > 0 {
		genConfig = &generationConfig{
			Temperature:     req.Temperature,
			TopP:            req.TopP,
			MaxOutputTokens: req.MaxTokens,
			StopSequences:   req.Stop,
		}
	}

	return geminiRequest{
		Contents:          contents,
		SystemInstruction: sysInst,
		Tools:             tools,
		GenerationConfig:  genConfig,
	}
}

// mapResponse translates Gemini's native response into our common llm.ChatResponse.
//
// The critical difference from OpenAI/Anthropic: Gemini returns finishReason="STOP"
// for BOTH text responses and tool calls. We detect tool calls by checking whether
// any part contains a functionCall, and set finish_reason accordingly so the agent's
// Run() loop branches correctly.
func mapResponse(resp geminiResponse) *llm.ChatResponse {

	if len(resp.Candidates) == 0 {
		return &llm.ChatResponse{
			Choices: []llm.Choice{},
		}
	}

	candidate := resp.Candidates[0]

	// Walk parts, collecting text and tool calls separately.
	var textContent string
	var toolCalls []llm.ToolCall

	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			textContent += part.Text
		}

		if part.FunctionCall != nil {
			// Gemini Args is a JSON object, our common format wants a JSON string.
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				argsJSON = []byte("{}")
			}

			toolCalls = append(toolCalls, llm.ToolCall{
				ID:   generateCallID(),
				Type: "function",
				Function: llm.FunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	// Determine finish_reason for the agent's Run() loop.
	// This is the key translation: Gemini says "STOP" for everything,
	// but our agent needs "tool_calls" when the model wants to call tools.
	var finishReason string
	if len(toolCalls) > 0 {
		// Model returned functionCall parts — the agent should execute tools.
		finishReason = "tool_calls"
	} else {
		// No tool calls — map Gemini's native finish reason.
		switch candidate.FinishReason {
		case "STOP":
			finishReason = "stop"
		case "MAX_TOKENS":
			finishReason = "length"
		case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT":
			finishReason = "content_filter"
		default:
			finishReason = candidate.FinishReason
		}
	}

	var usage llm.Usage
	if resp.UsageMetadata != nil {
		usage = llm.Usage{
			PromptTokens:     resp.UsageMetadata.PromptTokenCount,
			CompletionTokens: resp.UsageMetadata.CandidatesTokenCount + resp.UsageMetadata.ThoughtsTokenCount,
			TotalTokens:      resp.UsageMetadata.TotalTokenCount,
		}
	}

	return &llm.ChatResponse{
		Model: resp.ModelVersion,
		Choices: []llm.Choice{
			{
				Index: 0,
				Message: llm.Message{
					Role:      "assistant",
					Content:   textContent,
					ToolCalls: toolCalls,
				},
				FinishReason: finishReason,
			},
		},
		Usage: usage,
	}
}

// CreateChat sends a chat completion request to Gemini's generateContent endpoint.
// It implements the llm.ChatProvider interface.
func (c *Client) CreateChat(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {

	nativeReq := mapRequest(req)

	jsonData, err := json.Marshal(nativeReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: failed to marshal request: %w", err)
	}

	// Gemini puts the model name in the URL path, not in the request body.
	url := fmt.Sprintf("%s/v1beta/models/%s:generateContent", c.baseURL, c.model)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("gemini: failed to create HTTP request: %w", err)
	}

	// Gemini uses x-goog-api-key header for auth (not Bearer token).
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", c.apiKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("gemini: failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini: unexpected status %d: %s", resp.StatusCode, string(body))
	}

	var nativeResp geminiResponse
	if err := json.Unmarshal(body, &nativeResp); err != nil {
		return nil, fmt.Errorf("gemini: failed to decode response: %w", err)
	}

	return mapResponse(nativeResp), nil
}
