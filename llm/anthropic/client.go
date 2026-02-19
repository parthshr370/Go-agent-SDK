// Package anthropic implements llm.ChatProvider for the Anthropic Messages API.
//
// This provider translates between the common types (llm.ChatRequest,
// llm.ChatResponse) and Anthropic's native format. The key differences:
//
//   - System prompts are a top-level "system" parameter, not a message
//   - Auth uses "x-api-key" header, not Bearer token
//   - Tool calls are "tool_use" content blocks, not a separate tool_calls field
//   - Tool results are "tool_result" content blocks in user messages
//   - Finish reasons differ: "end_turn" means "stop", "tool_use" means "tool_calls"
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go-agent-sdk/llm"
	"io"
	"net/http"
)

// anthropicRequest is the top-level body for POST /v1/messages.
// System prompt is a top-level string (not a message), max_tokens is required,
// tools don't have the "type":"function" wrapper, and messages use content
// block arrays instead of plain strings.
type anthropicRequest struct {
	Model       string             `json:"model"`
	MaxTokens   int                `json:"max_tokens"`
	System      string             `json:"system,omitempty"`
	Messages    []anthropicMessage `json:"messages"`
	Tools       []anthropicTool    `json:"tools,omitempty"`
	Temperature float64            `json:"temperature,omitempty"`
	TopP        float64            `json:"top_p,omitempty"`
	StopSeqs    []string           `json:"stop_sequences,omitempty"`
}

// anthropicMessage is a single message in the conversation.
//
// Unlike our common llm.Message where Content is a plain string,
// Anthropic uses a content block array. This is because a single
// assistant message can contain BOTH text AND tool_use blocks,
// and a single user message can contain BOTH text AND tool_result blocks.
//
// Content is json.RawMessage so we can marshal it as either:
//   - a plain string (for simple user/assistant text messages)
//   - an array of contentBlock (for tool_use, tool_result, or mixed content)
type anthropicMessage struct {
	Role    string          `json:"role"`    // "user" or "assistant" (never "system" or "tool")
	Content json.RawMessage `json:"content"` // string OR []contentBlock
}

// contentBlock is the union type for Anthropic's content array.
// A single message can contain multiple blocks of different types.
//
// Which fields are populated depends on the "type" field:
//
//	type="text"        : Text is set
//	type="tool_use"    : ID, Name, Input are set (assistant asking to call a tool)
//	type="tool_result" : ToolUseID, Content are set (us returning a tool's output)
//
// We use omitempty on everything except Type so the JSON stays clean —
// a text block won't have empty "id" or "name" fields cluttering it up.
type contentBlock struct {
	Type string `json:"type"` // "text", "tool_use", or "tool_result"

	// Fields for type="text"
	Text string `json:"text,omitempty"`

	// Fields for type="tool_use" (assistant requesting a tool call)
	ID    string `json:"id,omitempty"`
	Name  string `json:"name,omitempty"`
	Input any    `json:"input,omitempty"` // JSON object, not a string (unlike OpenAI)

	// Fields for type="tool_result" (us sending back tool output)
	ToolUseID string `json:"tool_use_id,omitempty"`
	// Content here is the tool's output. We use any because Anthropic accepts
	// either a plain string or an array of content blocks for rich results.
	// For our purposes we always send a plain string.
	ResultContent any `json:"content,omitempty"`

	// IsError signals to Claude that the tool execution failed.
	// When true, Claude knows to handle the error gracefully rather than
	// treating the content as a successful result.
	IsError bool `json:"is_error,omitempty"`
}

// anthropicTool describes a tool available to Claude.
//
// Anthropic's format is flatter than OpenAI's. Compare:
//
//	OpenAI:    {"type": "function", "function": {"name": "x", "parameters": {...}}}
//	Anthropic: {"name": "x", "input_schema": {...}}
//
// No "type":"function" wrapper. The schema key is "input_schema" not "parameters".
type anthropicTool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"input_schema"` // JSON Schema object
}

// anthropicResponse is the top-level response from POST /v1/messages.
// No "choices" array here — Anthropic returns one response directly with
// content blocks. Uses "stop_reason" instead of "finish_reason" and
// different usage field names (input_tokens vs prompt_tokens).
type anthropicResponse struct {
	ID         string          `json:"id"`
	Type       string          `json:"type"`        // always "message"
	Role       string          `json:"role"`        // always "assistant"
	Content    []responseBlock `json:"content"`     // text and/or tool_use blocks
	Model      string          `json:"model"`       // which model served this
	StopReason string          `json:"stop_reason"` // "end_turn", "tool_use", "max_tokens", "stop_sequence"
	StopSeq    *string         `json:"stop_sequence"`
	Usage      anthropicUsage  `json:"usage"`
}

// responseBlock is a content block in the response.
// Same union pattern as contentBlock, but only "text" and "tool_use" appear
// in responses (the API never sends back "tool_result" — that's only in requests).
//
//	type="text"     : Text is populated
//	type="tool_use" : ID, Name, Input are populated
type responseBlock struct {
	Type  string `json:"type"`            // "text" or "tool_use"
	Text  string `json:"text,omitempty"`  // for type="text"
	ID    string `json:"id,omitempty"`    // for type="tool_use"
	Name  string `json:"name,omitempty"`  // for type="tool_use"
	Input any    `json:"input,omitempty"` // for type="tool_use" — JSON object (map), NOT a string
}

// anthropicUsage tracks token consumption.
// We map input_tokens to PromptTokens and output_tokens to CompletionTokens.
// There's no total field so we compute it ourselves. Anthropic also returns
// cache-related fields which we ignore since our common Usage type doesn't
// have slots for them yet.
type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

const (
	// Base URL only — CreateChat appends "/v1/messages".
	// If this included the path, WithBaseURL("https://my-proxy.com") would break
	// because the path would be missing from the override.
	DefaultBaseURL = "https://api.anthropic.com"
)

type Client struct {
	apiKey     string
	model      string
	baseURL    string
	httpClient *http.Client
}

type Option func(*Client)

func WithBaseUrl(url string) Option {
	return func(c *Client) {
		c.baseURL = url
	}
}

func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		c.httpClient = hc
	}
}

func (c *Client) ModelName() string {
	return c.model
}
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

// mapRequest translates our common llm.ChatRequest into Anthropic's native format.
// Private because only CreateChat calls this — native types never leak out.
func mapRequest(req llm.ChatRequest) anthropicRequest {

	var systemPrompt string
	var messages []anthropicMessage

	for _, msg := range req.Messages {
		switch msg.Role {

		case "system":
			// Anthropic wants system prompt as a top-level field, not a message.
			if systemPrompt != "" {
				systemPrompt += "\n"
			}
			systemPrompt += msg.Content

		case "user":
			contentJSON, _ := json.Marshal(msg.Content)
			messages = append(messages, anthropicMessage{
				Role:    "user",
				Content: contentJSON,
			})

		case "assistant":
			if len(msg.ToolCalls) > 0 {
				// Assistant with tool calls: text + tool_use blocks in one content array.
				var blocks []contentBlock

				if msg.Content != "" {
					blocks = append(blocks, contentBlock{
						Type: "text",
						Text: msg.Content,
					})
				}

				for _, call := range msg.ToolCalls {
					// OpenAI Arguments is a JSON string, Anthropic Input is a JSON object.
					var inputObj any
					if err := json.Unmarshal([]byte(call.Function.Arguments), &inputObj); err != nil {
						inputObj = map[string]any{}
					}
					blocks = append(blocks, contentBlock{
						Type:  "tool_use",
						ID:    call.ID,
						Name:  call.Function.Name,
						Input: inputObj,
					})
				}

				contentJSON, _ := json.Marshal(blocks)
				messages = append(messages, anthropicMessage{
					Role:    "assistant",
					Content: contentJSON,
				})
			} else {
				// Plain text assistant message.
				contentJSON, _ := json.Marshal(msg.Content)
				messages = append(messages, anthropicMessage{
					Role:    "assistant",
					Content: contentJSON,
				})
			}

		case "tool":
			// OpenAI has role="tool". Anthropic has no "tool" role — tool results
			// go inside a role="user" message as a tool_result content block.
			blocks := []contentBlock{
				{
					Type:          "tool_result",
					ToolUseID:     msg.ToolCallID,
					ResultContent: msg.Content,
				},
			}
			contentJSON, _ := json.Marshal(blocks)
			messages = append(messages, anthropicMessage{
				Role:    "user",
				Content: contentJSON,
			})
		}
	}

	// Convert tools: unwrap OpenAI's {"type":"function","function":{...}} wrapper.
	var tools []anthropicTool
	for _, t := range req.Tools {
		tools = append(tools, anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		})
	}

	// Anthropic requires max_tokens. Default to 4096 if not set.
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}

	return anthropicRequest{
		Model:       req.Model,
		MaxTokens:   maxTokens,
		System:      systemPrompt,
		Messages:    messages,
		Tools:       tools,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		StopSeqs:    req.Stop,
	}
}

// mapResponse translates Anthropic's native response into our common llm.ChatResponse.
// The reverse of mapRequest: Anthropic's shape goes in, OpenAI-shaped common types come out.
func mapResponse(resp anthropicResponse) *llm.ChatResponse {

	// Walk content blocks, collecting text and tool calls separately.
	var textContent string
	var toolCalls []llm.ToolCall

	for _, block := range resp.Content {
		switch block.Type {

		case "text":
			// There can be multiple text blocks. Concatenate them.
			textContent += block.Text

		case "tool_use":
			// Reverse of what mapRequest did: Anthropic Input is a JSON object,
			// but our common ToolCall.Function.Arguments needs a JSON string.
			argsJSON, _ := json.Marshal(block.Input)

			toolCalls = append(toolCalls, llm.ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: llm.FunctionCall{
					Name:      block.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	// Normalize stop_reason to our common finish_reason values.
	// These are the only strings Run() checks, so they must match exactly.
	var finishReason string
	switch resp.StopReason {
	case "end_turn":
		finishReason = "stop"
	case "tool_use":
		finishReason = "tool_calls"
	case "max_tokens":
		finishReason = "length"
	default:
		finishReason = resp.StopReason
	}

	// Build the common response. Anthropic returns one response directly,
	// but our common format wraps it in a Choices array (OpenAI convention).
	return &llm.ChatResponse{
		ID:    resp.ID,
		Model: resp.Model,
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
		Usage: llm.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}
}

// CreateChat sends a chat completion request to Anthropic's Messages API.
// It implements the llm.ChatProvider interface.
func (c *Client) CreateChat(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {

	// Translate common format to Anthropic's native format.
	nativeReq := mapRequest(req)

	jsonData, err := json.Marshal(nativeReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/messages", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("anthropic: failed to create HTTP request: %w", err)
	}

	// Anthropic uses x-api-key header, not Bearer token.
	// Also requires an anthropic-version header on every request.
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", c.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("anthropic: failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("anthropic: unexpected status %d: %s", resp.StatusCode, string(body))
	}

	// Unmarshal into Anthropic's native response type, not our common type.
	// The JSON shape is different — no "choices" array, different field names.
	var nativeResp anthropicResponse
	if err := json.Unmarshal(body, &nativeResp); err != nil {
		return nil, fmt.Errorf("anthropic: failed to decode response: %w", err)
	}

	// Translate native response back to common format.
	return mapResponse(nativeResp), nil
}
