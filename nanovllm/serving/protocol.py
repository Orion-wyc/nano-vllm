from pydantic import BaseModel, Field, field_validator
import math


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[int]
    max_tokens: int = Field(default=64, gt=0)
    temperature: float = 1.0
    stream: bool = False
    ignore_eos: bool = False
    stop: str | list[str] | None = None
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = -1
    repetition_penalty: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value != -1 and value < 1:
            raise ValueError("top_k must be -1 or positive")
        return value

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("temperature must be greater than or equal to 0")
        return value


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64, gt=0)
    temperature: float = 1.0
    stream: bool = False
    ignore_eos: bool = False
    stop: str | list[str] | None = None
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = -1
    repetition_penalty: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value != -1 and value < 1:
            raise ValueError("top_k must be -1 or positive")
        return value

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("temperature must be greater than or equal to 0")
        return value


class ModelInfo(BaseModel):
    id: str
    object: str = "model"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
