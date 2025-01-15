# examples.py

# CompletionRequest Example
COMPLETION_REQUEST_EXAMPLE = [
        {
            "model": "tinystarcoder",
            "prompt": "Write a Python function to calculate Fibonacci numbers.",
            "max_tokens": 50,
            "temperature": 0.7,
            "top_p": 1.0,
        }
    ]

# ChatMessage Example
CHAT_MESSAGE_EXAMPLE = [
        {
            "role": "user",
            "content": "What are the benefits of using FastAPI for APIs?",
        }
    ]

# ChatCompletionRequest Example
CHAT_COMPLETION_REQUEST_EXAMPLE = [
        {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain the difference between REST and GraphQL."},
            ],
            "temperature": 0.7,
        }
    ]

# CompletionChoice Example
COMPLETION_CHOICE_EXAMPLE = [
        {
            "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)",
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop",
        }
    ]

# CompletionResponse Example
COMPLETION_RESPONSE_EXAMPLE = [
        {
            "id": "cmpl-xyz123",
            "object": "text_completion",
            "created": 1616173200,
            "model": "tinystarcoder",
            "choices": [
                {
                    "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 50,
                "total_tokens": 60,
            },
        }
    ]

# EmbeddingRequest Example
EMBEDDING_REQUEST_EXAMPLE = [
        {
            "model": "deepseekcoder",
            "input": "Generate embeddings for this input text",
        }
    ]

# EmbeddingResponse Example
EMBEDDING_RESPONSE_EXAMPLE = [
        {
            "model": "deepseekcoder",
            "embeddings": [0.1345, -0.6723, 0.2354, 0.6721, -0.2341],
            "input": "Generate embeddings for this input text",
        }
    ]
