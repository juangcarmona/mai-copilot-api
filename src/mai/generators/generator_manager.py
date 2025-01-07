import os
from mai.generators.generator_base import GeneratorBase

class GeneratorManager:
    """
    Manage and dynamically select generators.
    """
    def __init__(self):
        self.generators = {}

    def register(self, name: str, generator: GeneratorBase):
        self.generators[name] = generator

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
        return self.get(generator_name)