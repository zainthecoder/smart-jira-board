"""Configuration management for Llama 3 inference."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceConfig(BaseSettings):
    """Configuration settings for Llama 3 inference.
    
    Loads configuration from environment variables or .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Model Configuration
    model_name: str = Field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        description="Hugging Face model identifier"
    )
    cache_dir: Path = Field(
        default=Path("./model_cache"),
        description="Directory to cache downloaded models"
    )
    
    # Inference Settings
    max_length: int = Field(
        default=512,
        ge=1,
        le=8192,
        description="Maximum length of generated text"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    top_k: int = Field(
        default=50,
        ge=0,
        description="Top-k sampling parameter"
    )
    
    # # Performance Settings
    # use_quantization: bool = Field(
    #     default=False,
    #     description="Enable model quantization for lower memory usage"
    # )
    # quantization_bits: Literal[4, 8] = Field(
    #     default=8,
    #     description="Quantization bit precision (4 or 8)"
    # )
    device: str = Field(
        default="auto",
        description="Device to run inference on (auto, cuda, cpu)"
    )
    
    # Authentication
    hf_token: Optional[str] = Field(
        default=None,
        description="Hugging Face authentication token for gated models"
    )
    
    @field_validator("cache_dir")
    @classmethod
    def create_cache_dir(cls, v: Path) -> Path:
        """Ensure cache directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    def __repr__(self) -> str:
        """String representation masking sensitive information."""
        return (
            f"InferenceConfig(model_name='{self.model_name}', "
            f"device='{self.device}', "
            #f"quantization={self.use_quantization})"
        )


