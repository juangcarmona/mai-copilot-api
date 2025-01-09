import os
from mai.crosscutting.logging import get_logger
from mai.generators.code_llama_generator import CodeLlamaGenerator
from mai.generators.deepseek_coder_generator import DeepSeekCoderGenerator
from mai.generators.generator_base import GeneratorBase
from mai.generators.gpt2_generator import GPT2Generator
from mai.generators.qwen2_5_coder_generator import Qwen2_5CoderGenerator
from mai.generators.star_coder_generator import StarCoderGenerator
from mai.generators.tiny_star_coder_generator import TinyStarCoderGenerator

logger = get_logger()

class GeneratorManager:
    """
    Manage and dynamically select generators.
    """
    def __init__(self):
        self.generators = {}

    def register(self, name: str, generator: GeneratorBase):
        self.generators[name] = generator

    def register_all(self):
        self.register("codellama", CodeLlamaGenerator(pretrained="TheBloke/CodeLlama-7B-Python-AWQ"))
        self.register("gpt2", GPT2Generator(pretrained="gpt2"))
        self.register("tinystarcoder", TinyStarCoderGenerator())
        self.register("starcoder", StarCoderGenerator(pretrained="bigcode/starcoder"))
        self.register("qwen",Qwen2_5CoderGenerator() )
        self.register("deepseekcoder",DeepSeekCoderGenerator(pretrained="deepseek-ai/deepseek-coder-6.7b-base"))        
        logger.info(f"Registered generators: {', '.join(self.generators.keys())}")

    def get(self, name: str) -> GeneratorBase:
        if name not in self.generators:
            raise ValueError(f"Generator '{name}' is not registered.")
        return self.generators[name]
    
    def get_default(self) -> GeneratorBase:
        """
        Get the default generator based on an environment variable.
        """
        generator_name = os.getenv("DEFAULT_GENERATOR", "")
        if generator_name not in self.generators:
            raise ValueError(f"Default generator '{generator_name}' is not registered.")
        logger.info(f"Using default generator: {generator_name}")
        return self.get(generator_name)