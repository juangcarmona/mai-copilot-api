from mai.generators.generator_base import GeneratorBase
from transformers import AutoModelForCausalLM, AutoTokenizer
from mai.crosscutting.logging import get_logger

logger = get_logger()

class DeepSeekCoderGenerator(GeneratorBase):
    def __init__(self, pretrained: str = "deepseek-ai/deepseek-coder-6.7b-base", device: str = "cpu"):
        """
        Generator for DeepSeek-Coder.
        """
        super().__init__(pretrained=pretrained, device=device)

    @property
    def unsupported_parameters(self) -> set:
        """
        Parameters not supported by DeepSeek-Coder.
        """
        return {"return_full_text"}  # Adjust based on DeepSeek specifics

    def load(self):
        """
        Load the tokenizer and model for DeepSeek-Coder.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.pretrained, trust_remote_code=True)
            self.model.to(self.device)

            # Ensure a pad token is set
            self.default_parameters["pad_token_id"] = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

            # Log success
            logger.info(
                f"DeepSeek-Coder Generator initialized:\n"
                f"  - Checkpoint: {self.pretrained}\n"
                f"  - Device: {self.device}\n"
                f"  - Default Parameters: {self.default_parameters}\n"
            )
        except Exception as e:
            logger.error(f"Failed to load DeepSeek-Coder Generator: {e}")
            raise RuntimeError("Failed to initialize DeepSeek-Coder Generator.")

    # def generate(self, query: str, parameters: dict = None) -> str:
    #     """
    #     Generate text using DeepSeek-Coder.
    #     """
    #     if self.tokenizer is None or self.model is None:
    #         raise RuntimeError("Model and tokenizer must be loaded before generation.")

    #     try:
    #         # Tokenize input
    #         input_ids = self.tokenizer.encode(query, return_tensors="pt").to(self.device)

    #         # Merge default parameters with provided ones
    #         params = {**self.default_parameters, **(parameters or {})}

    #         # Generate text
    #         output_ids = self.model.generate(input_ids, **params)

    #         # Decode and return the generated text
    #         return self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    #     except Exception as e:
    #         logger.error(f"Error during text generation: {e}")
    #         raise RuntimeError("Text generation failed.")
