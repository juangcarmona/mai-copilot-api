from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from mai.generators.generator_base import GeneratorBase

class SantaCoderGenerator(GeneratorBase):
    def __init__(self, pretrained: str = "bigcode/santacoder", device: str = "cuda"):
        """
        Generator for SantaCoder.
        """
        super().__init__(pretrained=pretrained, device=device)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.model.config)
        self.generation_config.pad_token_id = self.tokenizer.eos_token_id

    def load(self):
        """
        Load the model and tokenizer for SantaCoder.
        """
        self.model.to(device=self.device)
        super().load()

    def generate(self, query: str, parameters: dict = None) -> str:
        """
        Generate text using SantaCoder.
        """
        input_ids = self.tokenizer.encode(query, return_tensors="pt").to(self.device)
        config = GenerationConfig.from_dict({
            **self.generation_config.to_dict(),
            **(parameters or {})
        })
        output_ids = self.model.generate(input_ids, generation_config=config)
        output_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
