import os
from threading import Lock
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
    Singleton Manager for handling and dynamically selecting generators.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.generators = {}
            self._initialized = True

    def register(self, name: str, generator: GeneratorBase):
        """
        Register a new generator with a lazy-loaded flag.
        """
        self.generators[name] = {"generator": generator, "loaded": False}

    def register_all(self):
        """
        Register all supported generators.
        """
        self.register("codellama", CodeLlamaGenerator(pretrained="TheBloke/CodeLlama-7B-Python-AWQ"))
        self.register("gpt2", GPT2Generator(pretrained="gpt2"))
        self.register("tinystarcoder", TinyStarCoderGenerator())
        self.register("starcoder", StarCoderGenerator(pretrained="bigcode/starcoder"))
        self.register("qwen", Qwen2_5CoderGenerator())
        self.register("deepseekcoder", DeepSeekCoderGenerator(pretrained="deepseek-ai/deepseek-coder-6.7b-base"))
        logger.info(f"Registered generators: {', '.join(self.generators.keys())}")

    def load(self, name: str):
        """
        Load a generator by name if it hasn't been loaded yet.
        """
        if name not in self.generators:
            raise ValueError(f"Generator '{name}' is not registered.")

        generator_entry = self.generators[name]
        if not generator_entry["loaded"]:
            logger.info(f"Loading generator '{name}'...")
            generator_entry["generator"].load()
            generator_entry["loaded"] = True
            logger.info(f"Generator '{name}' loaded successfully.")

    def load_models(self, default_model: str, chat_model: str = None):
        """
        Load the default and optional chat model specified by environment variables.
        """
        self.load(default_model)
        if chat_model:
            self.load(chat_model)

    def get(self, name: str) -> GeneratorBase:
        """
        Get a generator by name. Ensures the generator is loaded.
        """
        if name not in self.generators:
            raise ValueError(f"Generator '{name}' is not registered.")
        self.load(name)  # Ensure the generator is loaded before returning
        return self.generators[name]["generator"]

    def get_default(self) -> GeneratorBase:
        """
        Get the default generator based on an environment variable.
        """
        generator_name = os.getenv("DEFAULT_GENERATOR", "")
        if generator_name not in self.generators:
            raise ValueError(f"Default generator '{generator_name}' is not registered.")
        return self.get(generator_name)
