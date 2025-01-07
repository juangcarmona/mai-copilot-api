from mai.generators.generator_base import GeneratorBase


class CodeLlamaGenerator(GeneratorBase):
    def __init__(self, pretrained: str, device: str = "cpu"):
        """
        Generator for CodeLlama models.
        """
        super().__init__(pretrained, device, trust_remote_code=True)

    @property
    def unsupported_parameters(self) -> set:
        """
        Parameters not supported by CodeLlama.
        """
        return {"return_full_text"}
    
    def load(self):
        """
        Load the tokenizer and model for CodeLlama.
        """
        super().load()
        # Custom parameters for CodeLlama (if needed) 
        self.default_parameters.update({
            "max_new_tokens": 100,
            "temperature": 0.7,
        })