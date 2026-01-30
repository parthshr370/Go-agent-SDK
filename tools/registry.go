package tools

import (
	"fmt"
	"go-agent-sdk/llm"
	"go-agent-sdk/tools/jsonschema"
	"reflect"
)

// ToolDefinition wraps a Go function so the Agent can understand and execute it.
// Each ToolDefinition holds everything needed to describe itself to the LLM and
// to be called with the right arguments later.
type ToolDefinition struct {
	Name        string
	Description string

	// Func is the actual function stored as a reflect.Value.
	// We need this because Go doesn't let us store functions directly
	// in maps with arbitrary signatures - reflection is the escape hatch.
	Func reflect.Value

	// ArgsType tells us what struct the function expects.
	// If the function is GetWeather(args WeatherArgs),
	// ArgsType holds the type information for WeatherArgs.
	// We need this to create new instances when the LLM calls the tool.
	ArgsType reflect.Type

	// Schema is the JSON Schema describing the function's parameters.
	// This gets sent to the LLM so it knows what arguments to provide.
	// It's a map[string]any (Go's version of a flexible dict) because
	// JSON Schema has nested objects.
	Schema map[string]any
}

// Registry stores all the tool definitions the Agent can use.
// Think of it as a toolbox where each tool has a name tag.
type Registry struct {
	definitions map[string]ToolDefinition
}

// NewRegistry creates an empty Registry ready for tools to be added.
func NewRegistry() *Registry {
	return &Registry{
		definitions: make(map[string]ToolDefinition),
	}
}

// Register adds a function to the Registry so the Agent can use it.
// The function must take exactly one argument (a struct with JSON tags)
// and return a string (or something that can be converted to string).
//
// What happens here:
//  1. We validate that 'function' is actually a function (not a string or int)
//  2. We check it has exactly one argument (the LLM can't call functions with 0 or 2+ args)
//  3. We extract the argument's type so we can recreate it later
//  4. We generate a JSON Schema describing that argument type (for the LLM)
//  5. We store everything using reflection so we can call it dynamically
//
// Example:
//
//	type WeatherArgs struct {
//	    City string `json:"city"`
//	}
//
//	func GetWeather(args WeatherArgs) string {
//	    return fmt.Sprintf("Sunny in %s", args.City)
//	}
//
//	registry.Register("get_weather", "Get current weather", GetWeather)
func (r *Registry) Register(name string, description string, function any) error {

	fnType := reflect.TypeOf(function)

	if fnType.Kind() != reflect.Func {
		return fmt.Errorf("this is not a valid function please try again")
	}

	if fnType.NumIn() != 1 {
		return fmt.Errorf("function must have exactly 1 argument")
	}

	argType := fnType.In(0)

	// Generate schema using our helper
	schema := jsonschema.GenerateSchema(argType)

	// Store the tool definition
	r.definitions[name] = ToolDefinition{
		Name:        name,
		Description: description,
		Func:        reflect.ValueOf(function),
		ArgsType:    argType,
		Schema:      schema,
	}

	return nil
}

// GetAllTools converts internal tool definitions to the API format required by OpenRouter.
// The Registry stores tools as a map for fast lookup by name, but the API expects
// a list (slice) of tools. This function performs that transformation.
//
// Why we need this:
//   - Our internal storage uses map[string]ToolDefinition for O(1) lookups by name
//   - OpenAI/OpenRouter expects []llm.Tool in the request body
//   - We convert each ToolDefinition to llm.Tool, setting Type to "function"
//
// The conversion:
//   - Name → Function.Name
//   - Description → Function.Description
//   - Schema → Function.Parameters (this is the JSON Schema describing args)
//   - Type is always "function" (OpenAI's terminology for callable tools)
//
// If no tools are registered, returns an empty slice (not nil) to avoid
// JSON marshaling issues where null might cause API errors.
func (r *Registry) GetAllTools() []llm.Tool {

	// Initialize empty slice (not nil) - important for JSON marshaling
	// A nil slice would marshal to "null", empty slice to "[]"
	// OpenRouter expects either a valid array or no field at all
	result := make([]llm.Tool, 0)

	// Iterate over all registered tool definitions
	// We use _ for the key (tool name) since we already have it in the definition
	for _, def := range r.definitions {

		// Convert internal ToolDefinition to API llm.Tool format
		apiTool := llm.Tool{
			Type: "function", // Always "function" for executable tools
			Function: llm.FunctionDescription{
				Name:        def.Name,
				Description: def.Description,
				Parameters:  def.Schema, // The JSON Schema describing what args the LLM should provide
			},
		}
		result = append(result, apiTool)
	}
	return result
}
