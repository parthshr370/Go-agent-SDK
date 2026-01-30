package jsonschema

import (
	"reflect"
	"strings"
)

// GenerateSchema takes a struct type and returns a map[string]any
// representing the JSON Schema required for OpenAI tool definitions.
func GenerateSchema(t reflect.Type) map[string]any {
	// Handle pointers (dereference them)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	// Base cases for primitive types
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	}

	// Complex case: Structs
	if t.Kind() == reflect.Struct {
		properties := make(map[string]any)
		required := []string{}

		// Iterate over struct fields
		for i := 0; i < t.NumField(); i++ {
			field := t.Field(i)

			// Get the JSON tag name (e.g. `json:"city"`)
			jsonTag := field.Tag.Get("json")
			if jsonTag == "" || jsonTag == "-" {
				continue // Skip fields without JSON tags
			}

			// Handle "omitempty"
			name := jsonTag
			if strings.Contains(jsonTag, ",") {
				parts := strings.Split(jsonTag, ",")
				name = parts[0]
				// If not omitempty, it's required
				if !strings.Contains(jsonTag, "omitempty") {
					required = append(required, name)
				}
			} else {
				// No commas means required by default in our logic
				required = append(required, name)
			}

			// Recursively generate schema for the field's type
			fieldSchema := GenerateSchema(field.Type)

			// Add description if present (e.g. `description:"City name"`)
			if desc := field.Tag.Get("description"); desc != "" {
				fieldSchema["description"] = desc
			}

			properties[name] = fieldSchema
		}

		return map[string]any{
			"type":       "object",
			"properties": properties,
			"required":   required,
		}
	}

	return nil
}
