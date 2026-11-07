from dataclasses import dataclass
import math


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 or positive")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be finite and positive")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
