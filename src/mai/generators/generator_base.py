from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from typing import Dict, Optional

from mai.crosscutting.logging import get_logger

logger = get_logger()

class GeneratorBase:
    def __init__(self, pretrained: str, device: str = "cpu", trust_remote_code: bool = False):
        """
        Base class for text generators.
        """
        self.pretrained = pretrained
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.default_parameters: Dict = {
            "max_new_tokens": 50,
            "temperature": 0.8,
            "top_p": 0.9,
            "do_sample": True,
        }
    

    @property
    def unsupported_parameters(self) -> set:
        """
        Parameters not supported by this generator.
        Subclasses can override this to specify their unsupported parameters.
        """
        return set()  # Default: No unsupported parameters

    def filter_parameters(self, parameters: Dict) -> Dict:
        """
        Remove unsupported parameters from the dictionary.
        """
        return {k: v for k, v in parameters.items() if k not in self.unsupported_parameters}


    def load(self):
        """
        Load the tokenizer and model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained, trust_remote_code=self.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained, trust_remote_code=self.trust_remote_code)
        self.model.to(device=self.device)

        # Set pad token if not defined
        self.default_parameters["pad_token_id"] = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # Log detailed information
        logger.info(
            f"Generator '{self.__class__.__name__}' initialized:\n"
            f"  - Checkpoint: {self.pretrained}\n"
            f"  - Device: {self.device}\n"
            f"  - Default Parameters: {self.default_parameters}\n"
            f"  - Tokenizer PAD Token ID: {self.default_parameters['pad_token_id']}\n"
        )

    def generate(self, query: str, parameters: Dict = None) -> str:
        """
        Generate text based on the query and parameters.
        """
        if self.tokenizer is None or self.model is None:
            raise ValueError("Model and tokenizer must be loaded before generation.")

        # Tokenize input
        input_ids = self.tokenizer.encode(query, return_tensors="pt").to(self.device)

        # Merge default parameters with provided ones
        params = {**self.default_parameters, **(parameters or {})}

        # Filter unsupported parameters
        params = self.filter_parameters(params)

        # Generate text
        output_ids = self.model.generate(input_ids, **params)

        # Decode and return the generated text
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
