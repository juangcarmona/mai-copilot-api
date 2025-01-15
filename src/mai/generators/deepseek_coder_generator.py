import torch
from mai.generators.generator_base import GeneratorBase
from transformers import AutoModelForCausalLM, AutoTokenizer
from mai.crosscutting.logging import get_logger

logger = get_logger()

class DeepSeekCoderGenerator(GeneratorBase):
    def __init__(self, pretrained: str = "deepseek-ai/DeepSeek-Coder-V2-Base", device: str = "cpu"):
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

    def generate_embeddings(self, inputs: str) -> list:
        """
        Generate embeddings for the given input using the model.
        """
        try:
            # Tokenize the input
            input_ids = self.tokenizer(inputs, return_tensors="pt").to(self.device)

            # Get the model's embeddings
            with torch.no_grad():
                outputs = self.model(**input_ids, output_hidden_states=True)

            # Extract embeddings from the last hidden state or a specific layer
            embeddings = outputs.hidden_states[-1].mean(dim=1).squeeze().tolist()
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}")