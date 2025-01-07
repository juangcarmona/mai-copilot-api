import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict

from mai.generators.generator_base import GeneratorBase
from mai.crosscutting.logging import get_logger

logger = get_logger()

class TinyStarCoderGenerator(GeneratorBase):
    def __init__(self, pretrained: str = "bigcode/tiny_starcoder_py", device: str = "cpu"):
        """
        TinyStarCoderGenerator: Specific generator for TinyStarCoder.
        """
        super().__init__(pretrained=pretrained, device=device)

    @property
    def unsupported_parameters(self) -> set:
        """
        Parameters not supported by TinyStarCoder. 
        """
        return {"return_full_text"}  # Add other unsupported parameters if needed

    def load(self):
        """
        Load the tokenizer and model with specific configurations for TinyStarCoder.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained, low_cpu_mem_usage=True, torch_dtype=torch.float32)
        self.model.to(device=self.device)

        # Set pad token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.default_parameters["pad_token_id"] = self.tokenizer.pad_token_id

        # Log detailed information
        logger.info(
            f"TinyStarCoderGenerator initialized:\n"
            f"  - Checkpoint: {self.pretrained}\n"
            f"  - Device: {self.device}\n"
            f"  - Default Parameters: {self.default_parameters}\n"
            f"  - Tokenizer PAD Token ID: {self.default_parameters['pad_token_id']}\n"
        )
    
    def generate(self, query: str, parameters: Dict = None) -> str:
        """
        Generate text using the base implementation, ensuring attention_mask is passed.
        """
        if self.tokenizer is None or self.model is None:
            raise ValueError("Model and tokenizer must be loaded before generation.")

        # Tokenize input and create attention mask 
        tokenized_inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=parameters.get("max_length", 512),  # Default max length
        )
        input_ids = tokenized_inputs["input_ids"].to(self.device)
        attention_mask = tokenized_inputs["attention_mask"].to(self.device)

        # Merge attention_mask into parameters
        parameters = parameters or {}
        parameters["attention_mask"] = attention_mask 

        # Call the base class generate method
        return super().generate(query=query, parameters=parameters) 

