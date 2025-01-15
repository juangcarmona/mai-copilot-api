from typing import List, Optional
from pydantic import BaseModel
from .examples import (
    COMPLETION_REQUEST_EXAMPLE,
    CHAT_MESSAGE_EXAMPLE,
    CHAT_COMPLETION_REQUEST_EXAMPLE,
    COMPLETION_CHOICE_EXAMPLE,
    COMPLETION_RESPONSE_EXAMPLE,
    EMBEDDING_REQUEST_EXAMPLE,
    EMBEDDING_RESPONSE_EXAMPLE,
)

# Request Models
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7


# Response Models
class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[dict]
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[dict]


class EmbeddingRequest(BaseModel):
    model: str
    input: str


class EmbeddingResponse(BaseModel):
    model: str
    embeddings: List[float]
    input: str
