package llm

// ChatRequest is what we send to the OpenRouter API.
// It contains everything the LLM needs to generate a response.
//
// The `omitempty` tag on optional fields means they won't be sent
// if they're zero values (empty string, 0, false, nil, etc.).
// This keeps the JSON payload clean and small.
type ChatRequest struct {
	// Required fields - these must always be present
	Model    string    `json:"model"`    // The model ID, like "google/gemini-3-flash-preview"
	Messages []Message `json:"messages"` // The conversation history

	// Optional Configuration
	Temperature      float64         `json:"temperature,omitempty"`       // 0.0 to 2.0, controls randomness
	TopP             float64         `json:"top_p,omitempty"`             // Nucleus sampling parameter
	Stream           bool            `json:"stream,omitempty"`            // If true, get tokens as they're generated
	Stop             []string        `json:"stop,omitempty"`              // Stop generation at these strings
	MaxTokens        int             `json:"max_tokens,omitempty"`        // Maximum tokens to generate
	PresencePenalty  float64         `json:"presence_penalty,omitempty"`  // Penalize repeating topics
	FrequencyPenalty float64         `json:"frequency_penalty,omitempty"` // Penalize repeating words
	LogitBias        map[string]int  `json:"logit_bias,omitempty"`        // Influence token likelihood
	User             string          `json:"user,omitempty"`              // End-user ID for tracking
	ResponseFormat   *ResponseFormat `json:"response_format,omitempty"`   // Force JSON output
	Seed             int             `json:"seed,omitempty"`              // For deterministic outputs

	// Tool Calling Configuration
	// Tools tells the LLM what functions it can call.
	// The LLM doesn't actually run them - it just tells us to.
	Tools []Tool `json:"tools,omitempty"`
	// ToolChoice controls when the LLM can use tools:
	//   "auto" - LLM decides when to use tools
	//   "none" - Never use tools
	//   specific object - Force a specific tool
	ToolChoice interface{} `json:"tool_choice,omitempty"`
}

// Message is a single exchange in the conversation.
// The Role field determines what kind of message this is:
//
//	"system"    - Setup instructions for the LLM's behavior
//	"user"      - What the human is asking
//	"assistant" - What the LLM responded (can contain ToolCalls)
//	"tool"      - The result of executing a tool
//
// Content is the actual text. Note that Content is empty (null in JSON)
// when the assistant is making tool calls - the ToolCalls field holds
// that information instead.
type Message struct {
	Role       string     `json:"role"`    // "user", "assistant", "system", or "tool"
	Content    string     `json:"content"` // The text content (empty for tool call messages)
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`   // Present when assistant wants to call tools
	ToolCallID string     `json:"tool_call_id,omitempty"` // Required for "tool" role messages
}

// Tool describes a function the LLM can call.
// We send these in the request so the LLM knows what's available.
type Tool struct {
	Type     string              `json:"type"`     // Always "function" for now
	Function FunctionDescription `json:"function"` // The actual function definition
}

// FunctionDescription provides metadata about a callable function.
// This is what the LLM sees when deciding which tool to use.
type FunctionDescription struct {
	Name        string      `json:"name"`                  // Unique identifier for the function
	Description string      `json:"description,omitempty"` // What the function does
	Parameters  interface{} `json:"parameters"`            // JSON Schema describing the arguments
}

// ToolCall is the LLM's request to execute a specific tool.
// When the LLM decides it needs to use a function, it sends one of these
// in the response's Message.ToolCalls field.
type ToolCall struct {
	ID       string       `json:"id"`       // Unique ID for this tool call (we need to echo it back)
	Type     string       `json:"type"`     // Always "function"
	Function FunctionCall `json:"function"` // Which function and with what arguments
}

// FunctionCall contains the specific function name and arguments.
// IMPORTANT: Arguments is a JSON STRING, not a JSON object.
// This is OpenAI's format - the LLM sends arguments as a serialized JSON string
// that we need to parse before calling the actual Go function.
type FunctionCall struct {
	Name      string `json:"name"`      // The function name
	Arguments string `json:"arguments"` // JSON string like "{\"city\": \"Paris\"}"
}

// ChatResponse is what we get back from the OpenRouter API.
// It contains the LLM's reply along with metadata about the generation.
type ChatResponse struct {
	ID                string   `json:"id"`                           // Unique request ID
	Object            string   `json:"object"`                       // Usually "chat.completion"
	Created           int64    `json:"created"`                      // Unix timestamp
	Model             string   `json:"model"`                        // Which model actually served this
	SystemFingerprint string   `json:"system_fingerprint,omitempty"` // Internal routing info
	Choices           []Choice `json:"choices"`                      // The actual response(s)
	Usage             Usage    `json:"usage"`                        // Token counts
}

// Choice represents one possible completion from the LLM.
// Usually we only get one (index 0), but if you request multiple completions,
// you get multiple choices.
type Choice struct {
	Index        int         `json:"index"`         // Which choice this is (0-based)
	Message      Message     `json:"message"`       // The actual message content
	FinishReason string      `json:"finish_reason"` // Why the generation stopped
	Logprobs     interface{} `json:"logprobs,omitempty"`
}

// Usage tracks how many tokens were used in this request.
// Tokens are what the LLM actually processes - roughly 3-4 characters per token.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`     // Tokens in your messages
	CompletionTokens int `json:"completion_tokens"` // Tokens in the response
	TotalTokens      int `json:"total_tokens"`      // Total for billing
}

// ResponseFormat forces the LLM to output valid JSON.
// Set Type to "json_object" to get structured output.
type ResponseFormat struct {
	Type string `json:"type"` // "text" or "json_object"
}
