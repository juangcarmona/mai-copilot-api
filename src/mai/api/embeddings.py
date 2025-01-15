from typing import Annotated
from fastapi import APIRouter, Body, HTTPException
from mai.models.examples import EMBEDDING_REQUEST_EXAMPLE
from mai.models.openai_models import EmbeddingRequest, EmbeddingResponse
from mai.generators.generator_manager import GeneratorManager
from mai.crosscutting.logging import get_logger

router = APIRouter()
logger = get_logger("embeddings")

generator_manager = GeneratorManager()


@router.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(
    request: Annotated[
        EmbeddingRequest,
        Body(examples=EMBEDDING_REQUEST_EXAMPLE),
    ]):
    """
    Generate embeddings for the provided input text.
    """
    try:
        generator = generator_manager.get(request.model)
        embeddings = generator.generate_embeddings(request.input)  # Ensure the generator supports this
        
        return EmbeddingResponse(
            model=request.model,
            embeddings=embeddings,
            input=request.input
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
