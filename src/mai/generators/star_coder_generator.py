import torch
from transformers import pipeline, GenerationConfig
from mai.generators.generator_base import GeneratorBase
from mai.crosscutting.logging import get_logger

logger = get_logger()

class StarCoderGenerator(GeneratorBase):
    def __init__(self, pretrained: str = "bigcode/starcoder", device: str = "cpu", device_map: str = None):
        """
        Generator for StarCoder.
        """
        super().__init__(pretrained=pretrained, device=device)
        self.device_map = device_map
        self.pipe = None
        self.generation_config = None

    def load(self):
        """
        Load the StarCoder pipeline and generation configuration.
        """
        try:
            # Initialize the pipeline for text generation
            self.pipe = pipeline(
                "text-generation",
                model=self.pretrained,
                torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                device=self.device if self.device != "cpu" else -1,  # Adjust for CPUs
                device_map=self.device_map,
            )

            # Load generation configuration
            self.generation_config = GenerationConfig.from_pretrained(self.pretrained)
            self.generation_config.pad_token_id = self.pipe.tokenizer.pad_token_id or self.pipe.tokenizer.eos_token_id

            # Log success
            logger.info(
                f"StarCoderGenerator initialized:\n"
                f"  - Checkpoint: {self.pretrained}\n"
                f"  - Device: {self.device}\n"
                f"  - Pad Token ID: {self.generation_config.pad_token_id}\n"
            )
        except Exception as e:
            logger.error(f"Failed to load StarCoderGenerator: {e}")
            raise RuntimeError("Failed to initialize StarCoderGenerator.")

    def generate(self, query: str, parameters: dict = None) -> str:
        """
        Generate text using the StarCoder pipeline.
        """
        if not self.pipe or not self.generation_config:
            raise RuntimeError("Model and configuration must be loaded before generating text.")

        try:
            # Merge default and provided parameters
            config_dict = {**self.generation_config.to_dict(), **(parameters or {})}
            config_dict = self.filter_parameters(config_dict)

            # Validate `do_sample` and dependent parameters
            if not config_dict.get("do_sample", True):
                for param in ["temperature", "top_p"]:
                    if param in config_dict:
                        logger.warning(f"Removing incompatible parameter '{param}' due to 'do_sample=False'")
                        del config_dict[param]

            # Apply the filtered configuration
            config = GenerationConfig.from_dict(config_dict)

            # Generate text
            json_response = self.pipe(query, generation_config=config)[0]
            generated_text = json_response["generated_text"]

            return generated_text
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise RuntimeError("Text generation failed.")
