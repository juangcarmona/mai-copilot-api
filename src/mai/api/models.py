from fastapi import APIRouter
from mai.generators.generator_manager import GeneratorManager

router = APIRouter()
generator_manager = GeneratorManager()

@router.get("/models")
async def list_models():
    """
    List all available models.
    """
    models = [
        {"id": name, "object": "model"}
        for name in generator_manager.generators.keys()
    ]
    return {"data": models}
