package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go-agent-sdk/llm"
	"io"
	"net/http"
)

// Wire structs for OpenAI's streaming ChatCompletionChunk JSON.
// These are unexported because they're internal plumbing -- the ParseFunc
// translates them into common llm.StreamEvent before anything outside
// this package sees them.
//
// Each struct matches one level of nesting in the JSON:
//   streamChunk -> streamChoice -> streamDelta -> streamToolCall -> streamFunction

type streamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []streamChoice `json:"choices"`
}

type streamChoice struct {
	Index        int         `json:"index"`
	FinishReason *string     `json:"finish_reason"`
	Delta        streamDelta `json:"delta"`
}

type streamDelta struct {
	Role      *string          `json:"role"`
	Content   *string          `json:"content"`
	ToolCalls []streamToolCall `json:"tool_calls"`
}

type streamToolCall struct {
	Index    int            `json:"index"`
	ID       *string        `json:"id"`
	Type     *string        `json:"type"`
	Function streamFunction `json:"function"`
}

type streamFunction struct {
	Name      *string `json:"name"`
	Arguments *string `json:"arguments"`
}

// parseStreamChunk translates one raw JSON chunk from the SSE stream into
// a common [llm.StreamEvent]. It checks finish_reason, then content, then
// tool_calls -- whichever is populated first wins. Chunks with no useful
// data (common with reasoning models) return an empty event.
func parseStreamChunk(data string) (llm.StreamEvent, error) {
	var chunk streamChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return llm.StreamEvent{}, fmt.Errorf("openai stream: failed to parse chunk: %w", err)
	}
	if len(chunk.Choices) == 0 {
		return llm.StreamEvent{}, nil
	}

	choice := chunk.Choices[0]
	// Check finish_reason first -- catches end of stream
	if choice.FinishReason != nil {
		return llm.StreamEvent{
			Type:         llm.EventDone,
			FinishReason: *choice.FinishReason,
		}, nil
	}
	// Check for text content
	if choice.Delta.Content != nil && *choice.Delta.Content != "" {
		return llm.StreamEvent{
			Type: llm.EventText,
			Text: *choice.Delta.Content,
		}, nil
	}
	// Check for tool calls
	if len(choice.Delta.ToolCalls) > 0 {
		tc := choice.Delta.ToolCalls[0]
		// First chunk for this tool call has an ID -- that's the start
		if tc.ID != nil {
			name := ""
			if tc.Function.Name != nil {
				name = *tc.Function.Name
			}
			return llm.StreamEvent{
				Type:          llm.EventToolCallStart,
				ToolCallIndex: tc.Index,
				ToolCallID:    *tc.ID,
				ToolCallName:  name,
			}, nil
		}
		// Subsequent chunks just have argument fragments
		if tc.Function.Arguments != nil {
			return llm.StreamEvent{
				Type:          llm.EventToolCallDelta,
				ToolCallIndex: tc.Index,
				ToolCallArgs:  *tc.Function.Arguments,
			}, nil
		}
	}
	// Chunk had no useful data (reasoning models send these)
	return llm.StreamEvent{}, nil
}

// CreateChatStream sends a streaming chat completion request. It's the same
// as [Client.CreateChat] except it sets stream=true, doesn't read the full
// body, and returns a [llm.StreamReader] with the connection still open.
// The caller reads events with Recv() and must call Close() when done.
//
// Error responses (4xx, 5xx) aren't streamed -- they come back as normal
// JSON, so we read and close the body in that case before returning.
func (c *Client) CreateChatStream(ctx context.Context, req llm.ChatRequest) (*llm.StreamReader, error) {

	req.Stream = true

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

	// Error responses aren't streamed -- read the body for the error message
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("openai: unexpected status %d: %s", resp.StatusCode, string(body))
	}
	// Success -- wrap the open body in a StreamReader and return it.
	// The caller reads with Recv() and closes with defer stream.Close().
	return llm.NewStreamReader(resp.Body, parseStreamChunk), nil
}
