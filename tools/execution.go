package tools

import (
	"encoding/json"
	"fmt"
	"reflect"
)

// Execute runs a tool that the LLM requested.
//
// This is where the magic happens - we take a tool name and JSON arguments
// from the LLM, and somehow call the right Go function with the right arguments.
// We use reflection because we don't know at compile time which function
// will be called or what its argument type is.
//
// The pipeline:
//  1. Look up the tool by name in our registry
//  2. Create an empty instance of the tool's argument struct using reflect.New()
//     (this gives us something like *WeatherArgs{City: ""})
//  3. Unmarshal the LLM's JSON into that empty struct
//     (now we have *WeatherArgs{City: "Paris"})
//  4. Call the actual function using reflect.Value.Call()
//     (this runs GetWeather(args) under the hood)
//  5. Extract the result and convert it to a string
//
// The tricky part is that Call() needs the actual value, not the pointer,
// so we use argsInstance.Elem() to dereference it.
//
// If the function returns a plain string, we use that directly.
// If it returns interface{}, we try to cast it to string.
// This handles both simple functions and ones that might return errors too.
func (r *Registry) Execute(name string, argsJson string) (string, error) {

	def, exists := r.definitions[name]
	if !exists {
		return "", fmt.Errorf("tool %s not found", name)
	}

	// reflect.New creates a pointer to a new zero value of the type.
	// So if ArgsType is WeatherArgs, we get *WeatherArgs.
	// We need a pointer because json.Unmarshal requires one.
	argsInstance := reflect.New(def.ArgsType)

	// Unmarshal fills the struct with the LLM's arguments.
	// We have to call .Interface() because json.Unmarshal doesn't understand
	// reflect.Value - it needs a regular Go interface{}.
	if err := json.Unmarshal([]byte(argsJson), argsInstance.Interface()); err != nil {
		return "", fmt.Errorf("invalid args: %w", err)
	}

	// Call the function! We pass a slice of arguments.
	// argsInstance.Elem() gets us the actual struct value (not the pointer).
	results := def.Func.Call([]reflect.Value{argsInstance.Elem()})

	// Handle different return types:
	// Most tools return just a string, but some might return (string, error).
	// We check the first result's type and convert appropriately.
	if len(results) == 0 {
		return "", fmt.Errorf("function returned no results")
	}
	if results[0].Kind() == reflect.String {
		return results[0].String(), nil
	}
	if results[0].Kind() == reflect.Interface {
		if str, ok := results[0].Interface().(string); ok {
			return str, nil
		}
	}
	return "", fmt.Errorf("function did not return a string")
}
