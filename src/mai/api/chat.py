from typing import Annotated
from fastapi import APIRouter, Body, HTTPException
from mai.models.examples import CHAT_COMPLETION_REQUEST_EXAMPLE
from mai.models.openai_models import ChatCompletionRequest, CompletionResponse, ChatMessage
from mai.generators.generator_manager import GeneratorManager
from mai.crosscutting.logging import get_logger
import time

router = APIRouter()
logger = get_logger("chat_completions")

generator_manager = GeneratorManager()


@router.post("/chat/completions", response_model=CompletionResponse)
async def chat_completion(
        request: Annotated[
            ChatCompletionRequest,
            Body(examples=CHAT_COMPLETION_REQUEST_EXAMPLE),
        ]):
    """
    Generate chat-style completions.
    """
    try:
        generator = generator_manager.get(request.model)
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        
        # Generate text using the chat prompt
        generated_text = generator.generate(prompt, {
            "temperature": request.temperature or 0.7,
            "max_new_tokens": 512,  # Adjust as necessary
        })
        
        return CompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "text": generated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(prompt.split()) + len(generated_text.split())
            },
        )
    except Exception as e:
        logger.error(f"Error generating chat completion: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
