from typing import Optional
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    model: str = "gpt2"
    inputs: str
    parameters: Optional[dict] = {}