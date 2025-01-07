from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from mai.core.constants import COPILOT_API_NAME
from mai.crosscutting.logging import get_logger
from mai.generators.code_llama_generator import CodeLlamaGenerator
from mai.generators.generator_manager import GeneratorManager
from mai.generators.gpt2_generator import GPT2Generator
from mai.generators.qwen2_5_coder_generator import Qwen2_5CoderGenerator
# from mai.generators.replit_coder_generator import ReplitCodeGenerator
from mai.generators.santa_coder_generator import SantaCoderGenerator
from mai.generators.star_coder_generator import StarCoderGenerator
from mai.generators.tiny_star_coder_generator import TinyStarCoderGenerator
from mai.models.generate_request import GenerateRequest
import debugpy
import os
import uvicorn

logger = get_logger(COPILOT_API_NAME)

# Initialize the manager
generator_manager = GeneratorManager()

# Register available generators
generator_manager.register("codellama", CodeLlamaGenerator(pretrained="TheBloke/CodeLlama-7B-Python-AWQ", device="cpu"))
generator_manager.register("gpt2", GPT2Generator(pretrained="gpt2", device="cpu"))
generator_manager.register("tinystarcoder", TinyStarCoderGenerator())
generator_manager.register("starcoder", StarCoderGenerator(pretrained="bigcode/starcoder", device="cpu"))
# generator_manager.register("santacoder", SantaCoderGenerator())
generator_manager.register("qwen",Qwen2_5CoderGenerator() )

# Load the default generator by ENV VAR
# default_generator = generator_manager.get_default()
default_generator = generator_manager.get("gpt2")
default_generator.load()  


logger.info("Model and tokenizer loaded successfully.")

# Initialize FastAPI
app = FastAPI(
    title="MAI Copilot API",
    description="Code completion API able to use multiple LLMs.",
    version="0.0.1",
)
app.add_middleware(
    CORSMiddleware
)  

def clean_and_format_generated_text(inputs: str, generated_text: str) -> str:
    """
    Clean and format the generated text.
    """
    # Remove the original prompt from the generated text
    cleaned_text = generated_text.replace(inputs, "").strip()

    cleaned_text = (
    cleaned_text.replace("<fim_prefix>", "")
    .replace("<fim_suffix>", "")
    .replace("<fim_middle>", "")
    .strip()
    )
    
    
    # Decode escaped quotes and ensure comments are preserved
    cleaned_text = cleaned_text.replace('\\"', '"').replace("\\'", "'")

    # Separate the lines and ensure there are no extra whitespaces
    lines = [line.strip() for line in cleaned_text.split("\n") if line.strip()]
    
    # Split lines to clean and reassemble with proper formatting
    lines = cleaned_text.split("\n")
    formatted_lines = []

    for line in lines:
        if line.strip():  # Skip empty lines
            formatted_lines.append(line.strip())

    # Join the formatted lines into a single text
    return "\n".join(formatted_lines)

@app.post("/api/generate/")
async def generate_code(request: GenerateRequest):
    """
    Endpoint to generate code suggestions based on the input prompt.
    """
    try:
        inputs = request.inputs 
        parameters = request.parameters 
        
        logger.info(f"Received inputs: {inputs}")
        logger.info(f"Received parameters: {parameters}")

        # Use the default generator
        generated_text = default_generator.generate(inputs, parameters)

        # Clean and format the output
        cleaned_text = clean_and_format_generated_text(inputs, generated_text)

        logger.info(f"Generated code: {cleaned_text}")

        # Return the response
        return {
            "generated_text": cleaned_text, 
            "status": 200
        }
    
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/", include_in_schema=False, response_class=RedirectResponse)
async def redirect_to_swagger():    
    logger.info("Redirect to swagger...")
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    if os.getenv("DEBUG_MODE") == "true":
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
    uvicorn.run(app, host="0.0.0.0", port=8000)
