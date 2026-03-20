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

// Wire structs for Anthropic's streaming SSE events.
// Anthropic uses typed SSE (event: line before data: line) and a "type"
// discriminator inside the JSON. The event types we care about:
//
//	content_block_start  -- new content block (text or tool_use) begins
//	content_block_delta  -- fragment for the current block (text_delta or input_json_delta)
//	content_block_stop   -- current block is done
//	message_delta        -- carries stop_reason on the final event
//	message_stop         -- stream is done
//
// We skip message_start and ping events.

// streamEvent is the top-level envelope for every SSE data payload.
// The Type field tells us which other fields are populated.
type streamEvent struct {
	Type         string        `json:"type"`
	Index        int           `json:"index,omitempty"`
	ContentBlock *streamBlock  `json:"content_block,omitempty"` // for content_block_start
	Delta        *streamDelta  `json:"delta,omitempty"`         // for content_block_delta and message_delta
	Message      *streamMsgRef `json:"message,omitempty"`       // for message_start (we mostly skip this)
}

// streamBlock describes a content block starting. For text blocks, Type is "text".
// For tool calls, Type is "tool_use" and ID + Name are populated.
type streamBlock struct {
	Type string `json:"type"` // "text" or "tool_use"
	ID   string `json:"id,omitempty"`
	Name string `json:"name,omitempty"`
}

// streamDelta carries the actual fragment data. Which fields are populated
// depends on the delta's Type:
//
//	text_delta       -- Text is set (a text content fragment)
//	input_json_delta -- PartialJSON is set (a tool argument fragment)
//
// For message_delta events, StopReason is set instead.
type streamDelta struct {
	Type        string `json:"type,omitempty"`
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	StopReason  string `json:"stop_reason,omitempty"`
}

// streamMsgRef is the message object inside message_start events.
// We only use it to detect the event type; we don't extract data from it.
type streamMsgRef struct {
	ID    string `json:"id,omitempty"`
	Model string `json:"model,omitempty"`
}

// parseStreamEvent translates one Anthropic SSE data payload into a common
// [llm.StreamEvent]. Anthropic's discriminator is the top-level "type" field,
// not finish_reason like OpenAI. We also track the content block index for
// tool call accumulation.
func parseStreamEvent(data string) (llm.StreamEvent, error) {
	var evt streamEvent
	if err := json.Unmarshal([]byte(data), &evt); err != nil {
		return llm.StreamEvent{}, fmt.Errorf("anthropic stream: failed to parse event: %w", err)
	}

	switch evt.Type {

	case "content_block_start":
		if evt.ContentBlock == nil {
			return llm.StreamEvent{}, nil
		}
		if evt.ContentBlock.Type == "tool_use" {
			return llm.StreamEvent{
				Type:          llm.EventToolCallStart,
				ToolCallIndex: evt.Index,
				ToolCallID:    evt.ContentBlock.ID,
				ToolCallName:  evt.ContentBlock.Name,
			}, nil
		}
		// text block start -- no data to emit yet, text comes in deltas
		return llm.StreamEvent{}, nil

	case "content_block_delta":
		if evt.Delta == nil {
			return llm.StreamEvent{}, nil
		}
		switch evt.Delta.Type {
		case "text_delta":
			if evt.Delta.Text != "" {
				return llm.StreamEvent{
					Type: llm.EventText,
					Text: evt.Delta.Text,
				}, nil
			}
		case "input_json_delta":
			if evt.Delta.PartialJSON != "" {
				return llm.StreamEvent{
					Type:          llm.EventToolCallDelta,
					ToolCallIndex: evt.Index,
					ToolCallArgs:  evt.Delta.PartialJSON,
				}, nil
			}
		}
		return llm.StreamEvent{}, nil

	case "message_delta":
		// Carries stop_reason. Normalize to our common values.
		if evt.Delta != nil && evt.Delta.StopReason != "" {
			reason := evt.Delta.StopReason
			switch reason {
			case "end_turn":
				reason = "stop"
			case "tool_use":
				reason = "tool_calls"
			case "max_tokens":
				reason = "length"
			}
			return llm.StreamEvent{
				Type:         llm.EventDone,
				FinishReason: reason,
			}, nil
		}
		return llm.StreamEvent{}, nil

	case "message_stop":
		// Final event, stream is done. Recv() will get io.EOF from [DONE]
		// or scanner ending, so we just return an empty event here.
		return llm.StreamEvent{}, nil

	default:
		// message_start, ping, error events -- skip
		return llm.StreamEvent{}, nil
	}
}

// CreateChatStream sends a streaming request to Anthropic's Messages API.
// Same as [Client.CreateChat] except it sets stream=true, doesn't read the
// full body, and returns a [llm.StreamReader] with the connection still open.
//
// Anthropic's streaming uses the same /v1/messages endpoint with the same
// request body, just with "stream": true added. The response comes as
// typed SSE (event: line before each data: line). Our StreamReader's Recv()
// already skips event: lines and only processes data: lines.
func (c *Client) CreateChatStream(ctx context.Context, req llm.ChatRequest) (*llm.StreamReader, error) {

	// Translate to Anthropic's native request format, same as CreateChat.
	nativeReq := mapRequest(req)

	// Anthropic uses a separate "stream" field in the request body.
	// We marshal to a map so we can inject it without changing the struct.
	jsonData, err := json.Marshal(nativeReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: failed to marshal request: %w", err)
	}

	// Inject "stream": true into the JSON.
	var reqMap map[string]any
	if err := json.Unmarshal(jsonData, &reqMap); err != nil {
		return nil, fmt.Errorf("anthropic: failed to inject stream field: %w", err)
	}
	reqMap["stream"] = true
	jsonData, err = json.Marshal(reqMap)
	if err != nil {
		return nil, fmt.Errorf("anthropic: failed to re-marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/messages", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("anthropic: failed to create HTTP request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", c.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: HTTP request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("anthropic: unexpected status %d: %s", resp.StatusCode, string(body))
	}

	return llm.NewStreamReader(resp.Body, parseStreamEvent), nil
}
