from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class State(BaseModel):
    """
    State model for processing workflow.
    """

    video_path: str
    audio_path: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
