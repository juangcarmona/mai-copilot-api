from typing import Dict
from mai.generators.generator_base import GeneratorBase


class GPT2Generator(GeneratorBase):
    def __init__(self, pretrained: str, device: str = "cpu"):
        """
        Generator for GPT2 models.
        """
        super().__init__(pretrained, device)
    
    @property
    def unsupported_parameters(self) -> set:
        """
        Parameters not supported by gpt2.
        """
        return {"return_full_text"}

    def load(self):
        """
        Load the tokenizer and model for GPT2.
        """
        super().load()
        # Custom parameters for GPT2 (if needed) 
        self.default_parameters.update({
            "max_new_tokens": 50,
            "temperature": 0.8,
        })
    