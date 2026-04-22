package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"go-agent-sdk/agent"
	"go-agent-sdk/llm/openai"
)

// MovieInfo is what we want the agent to return.
// The json tags become the JSON Schema sent to the LLM.
// The description tags become field descriptions inside the schema.
type MovieInfo struct {
	Title    string   `json:"title"    description:"The full movie title"`
	Director string   `json:"director" description:"The director's full name"`
	Year     int      `json:"year"     description:"The release year as a 4-digit number"`
	Genres   []string `json:"genres"   description:"List of genre labels, e.g. Sci-Fi, Thriller"`
	Summary  string   `json:"summary"  description:"A one-sentence plot summary"`
}

func main() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("Set OPENROUTER_API_KEY environment variable")
	}

	provider := openai.NewOpenRouter(apiKey, "moonshotai/kimi-k2.6")

	// NewTyped[MovieInfo] wires the agent to return MovieInfo from Run.
	// Under the hood: a "final_result" tool is registered whose JSON Schema
	// matches MovieInfo. The LLM calls that tool instead of returning text.
	// We intercept the call and unmarshal the arguments into MovieInfo.
	myAgent := agent.NewTyped[MovieInfo](provider,
		agent.WithSystemPrompts("You are a movie database assistant. Answer questions about movies accurately."),
		agent.WithMaxRetries(3),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	movie, err := myAgent.Run(ctx, "Tell me about the movie Inception.")
	println("Prompt = Tell me about the Movie Inception")
	if err != nil {
		log.Fatalf("Failed: %v", err)
	}

	// Fields are directly accessible -- no string parsing, no json.Unmarshal,
	// no interface{} casting. The compiler knows the exact type.
	fmt.Printf("Title:    %s\n", movie.Title)
	fmt.Printf("Director: %s\n", movie.Director)
	fmt.Printf("Year:     %d\n", movie.Year)
	fmt.Printf("Genres:   %v\n", movie.Genres)
	fmt.Printf("Summary:  %s\n", movie.Summary)
}
