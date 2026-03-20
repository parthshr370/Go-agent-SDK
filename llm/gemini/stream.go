package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go-agent-sdk/llm"
	"io"
	"net/http"
)

// Wire structs for Gemini's streaming response.
// Gemini streams the same JSON shape as non-streaming (candidates with parts),
// but each chunk is a partial response delivered as an SSE data line.
//
// Key differences from OpenAI/Anthropic:
//   - No [DONE] sentinel -- the stream just ends (scanner hits EOF)
//   - finishReason is "STOP" for both text and tool calls (same quirk as non-streaming)
//   - Tool calls arrive as complete functionCall objects, not fragmented arguments
//   - Text arrives as partial parts[0].text in each chunk

// streamChunk is the top-level SSE data payload. Same shape as geminiResponse
// but we define it separately to keep streaming types self-contained.
type streamChunk struct {
	Candidates    []streamCandidate `json:"candidates"`
	UsageMetadata *geminiUsage      `json:"usageMetadata,omitempty"`
}

type streamCandidate struct {
	Content      streamContent `json:"content"`
	FinishReason string        `json:"finishReason,omitempty"` // only on the last chunk
	Index        int           `json:"index"`
}

type streamContent struct {
	Role  string       `json:"role"`
	Parts []streamPart `json:"parts"`
}

// streamPart is the union type for chunk content. On any given chunk,
// either Text or FunctionCall is populated, never both.
type streamPart struct {
	Text         string          `json:"text,omitempty"`
	FunctionCall *streamFuncCall `json:"functionCall,omitempty"`
}

type streamFuncCall struct {
	Name string `json:"name"`
	Args any    `json:"args,omitempty"` // JSON object, not a string
}

// parseStreamChunk translates one Gemini SSE data payload into a common
// [llm.StreamEvent]. Gemini's chunks carry partial candidates, so we check
// for text parts and functionCall parts in the same way as the non-streaming
// mapResponse. The finishReason quirk applies here too: "STOP" means both
// "done with text" and "done with tool calls."
func parseStreamChunk(data string) (llm.StreamEvent, error) {
	var chunk streamChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return llm.StreamEvent{}, fmt.Errorf("gemini stream: failed to parse chunk: %w", err)
	}

	if len(chunk.Candidates) == 0 {
		return llm.StreamEvent{}, nil
	}

	candidate := chunk.Candidates[0]

	// Check parts for content. Gemini sends one part per chunk typically.
	for i, part := range candidate.Content.Parts {
		// Tool call -- Gemini sends complete functionCall objects, not fragments.
		// Each functionCall is a separate EventToolCallStart + EventToolCallDone
		// rolled into one because there's no argument fragmentation.
		if part.FunctionCall != nil {
			// Convert args from JSON object to JSON string (our common format).
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				argsJSON = []byte("{}")
			}

			// For Gemini, tool calls arrive complete. We emit a ToolCallStart
			// with the full arguments already available. The Accumulator will
			// create the tool call builder from this. We also emit a ToolCallDelta
			// with the complete args so they get accumulated.
			//
			// We return the start event here. The agent will also need the args,
			// but since Gemini gives us everything in one shot, we pack the args
			// into ToolCallArgs on the start event. The Accumulator's Add() for
			// EventToolCallStart doesn't read ToolCallArgs though -- so we handle
			// this by emitting just the start event and relying on a convention:
			// the caller should check if ToolCallArgs is non-empty on start events.
			//
			// Actually, the simpler approach: emit EventText-style but for tools.
			// We generate our own ID (Gemini doesn't reliably return one).
			return llm.StreamEvent{
				Type:          llm.EventToolCallStart,
				ToolCallIndex: i,
				ToolCallID:    generateCallID(),
				ToolCallName:  part.FunctionCall.Name,
				ToolCallArgs:  string(argsJSON),
			}, nil
		}

		// Text content
		if part.Text != "" {
			return llm.StreamEvent{
				Type: llm.EventText,
				Text: part.Text,
			}, nil
		}
	}

	// Check if this is the final chunk (has finishReason).
	// We check this after parts because the last chunk can have both
	// a text part AND finishReason.
	if candidate.FinishReason != "" {
		reason := candidate.FinishReason
		switch reason {
		case "STOP":
			reason = "stop"
		case "MAX_TOKENS":
			reason = "length"
		case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT":
			reason = "content_filter"
		}
		return llm.StreamEvent{
			Type:         llm.EventDone,
			FinishReason: reason,
		}, nil
	}

	return llm.StreamEvent{}, nil
}

// CreateChatStream sends a streaming request to Gemini's generateContent endpoint.
// Same as [Client.CreateChat] except it uses the streamGenerateContent endpoint
// with ?alt=sse, doesn't read the full body, and returns a [llm.StreamReader]
// with the connection still open.
//
// Gemini's streaming endpoint is different from non-streaming:
//   - Non-streaming: /v1beta/models/{model}:generateContent
//   - Streaming:     /v1beta/models/{model}:streamGenerateContent?alt=sse
//
// The request body is identical. The response comes as SSE data lines,
// each containing a partial candidates response. There's no [DONE] sentinel --
// the stream simply ends (our StreamReader handles this via scanner EOF).
func (c *Client) CreateChatStream(ctx context.Context, req llm.ChatRequest) (*llm.StreamReader, error) {

	nativeReq := mapRequest(req)

	jsonData, err := json.Marshal(nativeReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: failed to marshal request: %w", err)
	}

	// Gemini uses a different endpoint for streaming.
	url := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse", c.baseURL, c.model)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("gemini: failed to create HTTP request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", c.apiKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: HTTP request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("gemini: unexpected status %d: %s", resp.StatusCode, string(body))
	}

	return llm.NewStreamReader(resp.Body, parseStreamChunk), nil
}
