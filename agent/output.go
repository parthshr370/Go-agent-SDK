package agent

import (
	"encoding/json"
	"fmt"
	"go-agent-sdk/llm"
	"go-agent-sdk/tools/jsonschema"
	"reflect"
)

// outputMode controls how the agent extracts a typed T from the LLM response.
//
// Text mode is the existing behavior -- return the raw string.
// Tool mode injects a fake "final_result" tool whose schema matches T.
// The LLM calls it with structured arguments; we intercept and unmarshal
// those arguments into T instead of executing a real function.
// Native mode uses response_format json_schema (OpenAI/Gemini only) -- future.
type outputMode string

const (
	outputModeText   outputMode = "text"
	outputModeTool   outputMode = "tool"
	outputModeNative outputMode = "native"
)

// outputSpec holds everything the agent needs to produce a T from the LLM.
// Built once at NewTyped[T] construction time, reused on every Run call.
type outputSpec[T any] struct {
	mode   outputMode
	name   string         // Go type name -- used as the tool name
	schema map[string]any // JSON Schema generated from T's fields
}

// buildOutputSpec inspects T at construction time and returns a configured outputSpec.
//
// When T is string, the spec uses text mode -- no schema, no extra tooling,
// existing behavior unchanged.
//
// For all other types (structs), the spec uses tool mode and generates a
// JSON Schema from T's fields using the same jsonschema package used for tools.
// The same reflection machinery that builds schemas for tool parameters
// works identically for output types.
func buildOutputSpec[T any]() outputSpec[T] {
	t := reflect.TypeOf((*T)(nil)).Elem()

	if t.Kind() == reflect.String {
		return outputSpec[T]{mode: outputModeText}
	}

	schema := jsonschema.GenerateSchema(t)
	return outputSpec[T]{
		mode:   outputModeTool,
		name:   t.Name(),
		schema: schema,
	}
}

// buildOutputTool creates the fake "final_result" tool from the output schema.
//
// This tool is injected into every LLM request when mode is "tool".
// The LLM sees it as a callable function with T's fields as parameters.
// When the LLM calls it, we intercept -- instead of executing a real Go
// function, the arguments themselves are the structured output.
//
// The description is deliberately instructional so the LLM knows to call
// this tool when it has a complete answer rather than returning plain text.
func (s outputSpec[T]) buildOutputTool() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.FunctionDescription{
			Name:        "final_result",
			Description: "Call this tool to return your final answer. You MUST call this tool to deliver your response -- do not return plain text.",
			Parameters:  s.schema,
		},
	}
}

// validateAndUnmarshal parses rawJSON into T.
//
// When T is string, raw is returned directly -- no JSON parsing needed,
// since the LLM's text output IS the result.
//
// For structs, standard json.Unmarshal is used. Returns a descriptive error
// if parsing fails so the agent can send it back to the LLM as a retry message.
func validateAndUnmarshal[T any](raw string) (T, error) {
	var out T

	// Type switch on the zero value to detect whether T is string at runtime.
	// This works because Go generics are monomorphized -- each instantiation
	// has its own concrete type for the switch to match against.
	switch any(out).(type) {
	case string:
		// T is string -- cast raw directly, no JSON parsing
		result, _ := any(raw).(T)
		return result, nil
	default:
		if err := json.Unmarshal([]byte(raw), &out); err != nil {
			return out, fmt.Errorf("output did not match expected structure: %w", err)
		}
		return out, nil
	}
}

// retryMessage builds the message sent back to the LLM when its output
// failed validation. Tells the LLM what went wrong so it can fix its output.
func retryMessage(err error) llm.Message {
	return llm.NewUserMessage(fmt.Sprintf(
		"Your output was invalid: %v. Please call the final_result tool again with valid JSON that matches the required schema exactly.",
		err,
	))
}
