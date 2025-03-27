from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """
    Model to track token usage across different processors.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Allow adding TokenUsage objects together."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def __str__(self) -> str:
        """String representation of token usage."""
        return f"Input tokens: {self.input_tokens}, Output tokens: {self.output_tokens}, Total tokens: {self.total_tokens}"

    @classmethod
    def from_usage_metadata(cls, usage_metadata: Dict[str, Any]) -> "TokenUsage":
        """
        Create TokenUsage from LLM response usage metadata.
        """
        if not usage_metadata:
            return cls()

        return cls(
            input_tokens=usage_metadata.get("input_tokens", 0),
            output_tokens=usage_metadata.get("output_tokens", 0),
        )


class State(BaseModel):
    """
    State model for processing workflow.
    """

    video_path: str = ""
    audio_path: str = ""
    context: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_usage: Dict[str, TokenUsage] = Field(default_factory=dict)

    @property
    def total_token_usage(self) -> TokenUsage:
        """Calculate total token usage across all processors."""
        total = TokenUsage()
        for usage in self.token_usage.values():
            total += usage
        return total

    def add_token_usage_from_metadata(
        self, processor_name: str, usage_metadata: Dict[str, Any]
    ) -> None:
        """
        Add token usage from LLM response metadata for a specific processor.
        """
        usage = TokenUsage.from_usage_metadata(usage_metadata)

        if processor_name in self.token_usage:
            self.token_usage[processor_name] += usage
        else:
            self.token_usage[processor_name] = usage
