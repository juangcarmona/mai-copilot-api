import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mai.generators.generator_base import GeneratorBase
from mai.crosscutting.logging import get_logger

logger = get_logger()

class Qwen2_5CoderGenerator(GeneratorBase):
    def __init__(self, pretrained: str = "Qwen/Qwen2.5-Coder-3B-Instruct", device: str = "cpu"):
        """
        Generator for Qwen2.5-Coder-3B-Instruct.
        """
        super().__init__(pretrained=pretrained, device=device, trust_remote_code=True)

    @property
    def unsupported_parameters(self) -> set:
        """
        Parameters not supported by Qwen.
        """
        return {"return_full_text"}  # Adjust based on model specifics

    def load(self):
        """
        Load the tokenizer and model for Qwen2.5-Coder.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.pretrained,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device != "cpu" else None,
            )
            self.model.to(self.device)

            # Ensure a pad token is set
            self.default_parameters["pad_token_id"] = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

            # Log success
            logger.info(
                f"Qwen2.5-Coder-3B-Instruct Generator initialized:\n"
                f"  - Checkpoint: {self.pretrained}\n"
                f"  - Device: {self.device}\n"
                f"  - Default Parameters: {self.default_parameters}\n"
            )
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-Coder-3B-Instruct Generator: {e}")
            raise RuntimeError("Failed to initialize Qwen2.5-Coder-3B-Instruct Generator.")

    def generate(self, query: str, parameters: dict = None) -> str:
        """
        Generate text using Qwen2.5-Coder.
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model and tokenizer must be loaded before generation.")

        try:
            # Prepare the input message for the Qwen chat template
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": query},
            ]

            # Apply the chat template and tokenize
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            # Merge default and provided parameters
            params = {**self.default_parameters, **(parameters or {})}

            # Generate text
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=params.get("max_new_tokens", 512),
                temperature=params.get("temperature", 0.8),
                top_p=params.get("top_p", 0.9),
                do_sample=params.get("do_sample", True),
            )

            # Extract and decode the generated output
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_ids)
            ]
            output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return output_text
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise RuntimeError("Text generation failed.")
