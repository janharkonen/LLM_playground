package main

import (
	"context"
	"fmt"
	"log"
	"os"

	openai "github.com/sashabaranov/go-openai"
)

func main() {
	temp := 0.3
	// Initialize OpenAI client
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	prompt := "What is the capital of France?"

	// Call ChatGPT (gpt-3.5 or gpt-4 depending on your plan)
	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT4, // or openai.GPT3Dot5Turbo
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleSystem,
					Content: "You are a helpful assistant.",
				},
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
			Temperature: float32(temp),
		},
	)

	if err != nil {
		log.Fatalf("ChatCompletion error: %v", err)
	}

	// Print response
	fmt.Println("Response:")
	fmt.Println(resp.Choices[0].Message.Content)
}
