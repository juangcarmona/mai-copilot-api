import time
from typing import Annotated
from fastapi import APIRouter, Body, HTTPException
from mai.models.examples import COMPLETION_REQUEST_EXAMPLE
from mai.models.openai_models import CompletionRequest, CompletionResponse
from mai.generators.generator_manager import GeneratorManager
from mai.crosscutting.logging import get_logger

router = APIRouter()
logger = get_logger("completions")

generator_manager = GeneratorManager()

@router.post("/completions", response_model=CompletionResponse)
async def generate_completion(
        request: Annotated[
            CompletionRequest,
            Body(examples=COMPLETION_REQUEST_EXAMPLE),
        ]):
    """
    Generate text/code completions.
    """
    try:
        generator = generator_manager.get(request.model)
        generated_text = generator.generate(request.prompt, {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        })
        return CompletionResponse(
            id="cmpl-unique-id",
            created=int(time.time()),
            model=request.model,
            choices=[{"text": generated_text, "index": 0, "logprobs": None, "finish_reason": "stop"}],
            usage={"prompt_tokens": len(request.prompt.split()), "completion_tokens": len(generated_text.split())},
        )
    except Exception as e:
        logger.error(f"Error generating completion: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
