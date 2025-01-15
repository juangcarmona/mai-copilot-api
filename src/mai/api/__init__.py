import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from mai.api.completions import router as completions_router
from mai.api.chat import router as chat_router
from mai.api.embeddings import router as embeddings_router
from mai.api.models import router as models_router
from mai.api.legacy import router as legacy_router
from mai.generators.generator_manager import GeneratorManager
from mai.crosscutting.logging import get_logger
from mai.core.constants import COPILOT_API_NAME

logger = get_logger(COPILOT_API_NAME)

def create_app() -> FastAPI:
    # Initialize app
    app = FastAPI(
        title="MAI Copilot API",
        description="OpenAI-like API for code completions, chat, and embeddings.",
        version="0.0.2",
    )
    app.add_middleware(CORSMiddleware)

    # Register routes
    app.include_router(completions_router, prefix="/v1")
    app.include_router(chat_router, prefix="/v1")
    app.include_router(embeddings_router, prefix="/v1")
    app.include_router(models_router, prefix="/v1")
    # Even legacy endpoint(s)
    app.include_router(legacy_router)

    # Add redirection to Swagger UI
    @app.get("/", include_in_schema=False, response_class=RedirectResponse)
    async def redirect_to_docs():
        logger.info("Redirecting to Swagger UI...")
        return RedirectResponse(url="/docs")

    # Get the singleton instance of GeneratorManager
    generator_manager = GeneratorManager()
    generator_manager.register_all()

    # Preload models based on environment variables
    default_model = os.getenv("DEFAULT_GENERATOR", "")
    chat_model = os.getenv("CHAT_GENERATOR", None)

    if default_model:
        logger.info(f"Preloading default model: {default_model}")
    if chat_model:
        logger.info(f"Preloading chat model: {chat_model}")

    generator_manager.load_models(default_model, chat_model)
    logger.info("All required models preloaded successfully.")

    return app