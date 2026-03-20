package llm

import (
	"bufio"
	"fmt"
	"io"
	"strings"
)

// EventType tells you what kind of streaming event just arrived.
// Check this before reading any other field on StreamEvent,
// because only the fields relevant to this type will be populated.
type EventType int

const (
	// EventText means a text content fragment arrived. Read StreamEvent.Text.
	EventText EventType = iota

	// EventToolCallStart means the model began a new tool call.
	// Read StreamEvent.ToolCallID, ToolCallName, and ToolCallIndex.
	EventToolCallStart

	// EventToolCallDelta means a tool call argument fragment arrived.
	// Read StreamEvent.ToolCallIndex and ToolCallArgs.
	EventToolCallDelta

	// EventToolCallDone means the tool call arguments are complete.
	// Read StreamEvent.ToolCallIndex.
	EventToolCallDone

	// EventDone means the stream finished normally.
	// Read StreamEvent.FinishReason for why ("stop", "length", "tool_calls").
	EventDone

	// EventError means something went wrong mid-stream.
	// The error itself comes back from Recv(), not from this event.
	EventError
)

// StreamEvent is a single event from a streaming LLM response.
// Most fields are empty on any given event -- check Type first to know
// which fields have values. We use one struct instead of separate types
// per event because it's simpler and sufficient for our needs.
type StreamEvent struct {
	Type EventType

	// Set when Type == EventText
	Text string

	// Set when Type == EventToolCallStart
	ToolCallID   string
	ToolCallName string

	// Set when Type == EventToolCallStart or EventToolCallDelta or EventToolCallDone
	ToolCallIndex int

	// Set when Type == EventToolCallDelta
	ToolCallArgs string

	// Set when Type == EventDone
	FinishReason string
}

// ParseFunc takes a raw SSE data payload (the part after "data: ")
// and translates it into a StreamEvent. Each provider implements its
// own ParseFunc because the JSON format differs between OpenAI,
// Anthropic, and Gemini.
type ParseFunc func(data string) (StreamEvent, error)

// StreamReader reads streaming events from an LLM provider one at a time.
// It wraps an open HTTP response body and reads SSE lines from it.
//
// The caller is responsible for closing it when done:
//
//	stream, err := provider.CreateChatStream(ctx, req)
//	if err != nil { ... }
//	defer stream.Close()
//	for {
//	    event, err := stream.Recv()
//	    if err == io.EOF { break }
//	    if err != nil { ... }
//	    fmt.Print(event.Text)
//	}
type StreamReader struct {
	body    io.ReadCloser  // the open HTTP response body (live connection)
	scanner *bufio.Scanner // reads lines from body one at a time
	parse   ParseFunc      // provider-specific chunk parser
	done    bool           // true after stream has ended
}

// NewStreamReader creates a StreamReader from an open HTTP response body.
// The parse function handles provider-specific JSON translation.
// Providers call this inside their CreateChatStream implementation.
func NewStreamReader(body io.ReadCloser, parse ParseFunc) *StreamReader {
	return &StreamReader{
		body:    body,
		scanner: bufio.NewScanner(body),
		parse:   parse,
	}
}

// Recv blocks until the next streaming event arrives, then returns it.
// Returns io.EOF when the stream ends normally.
// Returns a non-EOF error if the connection drops or parsing fails.
//
// Each call reads SSE lines from the HTTP body, skipping blank lines
// and comments, until it finds a data payload to parse.
func (r *StreamReader) Recv() (StreamEvent, error) {
	if r.done {
		return StreamEvent{}, io.EOF
	}

	for r.scanner.Scan() {
		line := r.scanner.Text()

		// blank lines are SSE event separators, skip them
		if line == "" {
			continue
		}

		// lines starting with : are SSE comments (keepalive pings), skip them
		if strings.HasPrefix(line, ":") {
			continue
		}

		// lines starting with event: set the SSE event type.
		// OpenAI doesn't use these, Anthropic does.
		// For now we skip them and just parse data lines.
		if strings.HasPrefix(line, "event:") {
			continue
		}

		// we only care about data lines
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		// [DONE] is OpenAI's end-of-stream signal
		if data == "[DONE]" {
			r.done = true
			return StreamEvent{Type: EventDone}, io.EOF
		}

		// hand the raw JSON to the provider-specific parser
		return r.parse(data)
	}

	// scanner.Scan() returned false -- either clean EOF or an error
	r.done = true
	if err := r.scanner.Err(); err != nil {
		return StreamEvent{}, fmt.Errorf("stream read error: %w", err)
	}
	return StreamEvent{}, io.EOF
}

// Close shuts down the stream by closing the underlying HTTP response body.
// This releases the TCP connection back to the pool. Best used with defer
// right after creating the stream.
func (r *StreamReader) Close() error {
	return r.body.Close()
}

// Accumulator collects streaming events and builds up complete data
// from fragments. Feed it events from Recv() as they arrive. When the
// stream ends, call Text() or ToolCalls() to get the assembled result.
//
// The agent uses this to reconstruct the full Message for conversation
// history after streaming completes.
type Accumulator struct {
	text      strings.Builder
	toolCalls map[int]*toolCallBuilder
}

// toolCallBuilder accumulates fragments for a single tool call.
type toolCallBuilder struct {
	id   string
	name string
	args strings.Builder
}

// Add feeds a streaming event into the accumulator.
// Call this for every event you get from Recv().
func (a *Accumulator) Add(event StreamEvent) {
	switch event.Type {
	case EventText:
		a.text.WriteString(event.Text)

	case EventToolCallStart:
		if a.toolCalls == nil {
			a.toolCalls = make(map[int]*toolCallBuilder)
		}
		tc := &toolCallBuilder{
			id:   event.ToolCallID,
			name: event.ToolCallName,
		}
		// Some providers (Gemini) send complete tool call args in one shot
		// on the start event rather than as separate deltas.
		if event.ToolCallArgs != "" {
			tc.args.WriteString(event.ToolCallArgs)
		}
		a.toolCalls[event.ToolCallIndex] = tc

	case EventToolCallDelta:
		if tc, ok := a.toolCalls[event.ToolCallIndex]; ok {
			tc.args.WriteString(event.ToolCallArgs)
		}
	}
}

// Text returns the complete accumulated text content.
func (a *Accumulator) Text() string {
	return a.text.String()
}

// HasToolCalls reports whether any tool calls were accumulated.
func (a *Accumulator) HasToolCalls() bool {
	return len(a.toolCalls) > 0
}

// ToolCalls returns the accumulated tool calls as the same ToolCall type
// used by ChatResponse, so they plug directly into the existing agent loop.
func (a *Accumulator) ToolCalls() []ToolCall {
	if len(a.toolCalls) == 0 {
		return nil
	}

	calls := make([]ToolCall, 0, len(a.toolCalls))
	for _, tc := range a.toolCalls {
		calls = append(calls, ToolCall{
			ID:   tc.id,
			Type: "function",
			Function: FunctionCall{
				Name:      tc.name,
				Arguments: tc.args.String(),
			},
		})
	}
	return calls
}

// Message builds a complete Message from the accumulated stream data.
// This is what gets appended to conversation history after streaming ends.
func (a *Accumulator) Message() Message {
	msg := Message{
		Role:    "assistant",
		Content: a.text.String(),
	}
	if a.HasToolCalls() {
		msg.ToolCalls = a.ToolCalls()
	}
	return msg
}
