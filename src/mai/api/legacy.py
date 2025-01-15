from typing import Annotated
import warnings
from fastapi import APIRouter, Body, HTTPException
from mai.models.generate_request import GenerateRequest
from mai.generators.generator_manager import GeneratorManager
from mai.crosscutting.logging import get_logger

router = APIRouter()
logger = get_logger("legacy")

generator_manager = GeneratorManager()

@router.post("/api/generate/")
async def generate_code(
    request: Annotated[
        GenerateRequest,
        Body(
            examples=[
                {
                    "inputs": "Write a Python function to calculate factorial.",
                    "parameters": {
                        "temperature": 0.7
                    }
                }
            ]
        )
    ],
):
    """
    Endpoint to generate code suggestions based on the input prompt.
    """
    warnings.warn(
        "This endpoint is deprecated and will be removed in a future release. Please migrate to /v1/completions.",
        DeprecationWarning,
    )
    default_generator = generator_manager.get_default()

    try:
        inputs = request.inputs
        parameters = request.parameters

        logger.info(f"Received inputs: {inputs}")
        logger.info(f"Received parameters: {parameters}")

        # Use the default generator
        generated_text = default_generator.generate(inputs, parameters)

        logger.info(f"Generated code: {generated_text}")

        # Return the response
        return {
            "generated_text": generated_text,
            "status": 200
        }

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
