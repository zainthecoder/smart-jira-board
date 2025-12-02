"""Model loading and initialization for Llama 3."""

import logging
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .config import InferenceConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and initialization of Llama 3 models."""
    
    def __init__(self, config: InferenceConfig):
        """Initialize the model loader.
        
        Args:
            config: Configuration object with model settings
        """
        self.config = config
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
    
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            RuntimeError: If model loading fails
        """
        if self._model is not None and self._tokenizer is not None:
            logger.info("Model already loaded, returning cached instances")
            return self._model, self._tokenizer
        
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Load tokenizer
            self._tokenizer = self._load_tokenizer()
            
            # Load model with optional quantization
            self._model = self._load_model()
            
            logger.info("Model and tokenizer loaded successfully")
            return self._model, self._tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer.
        
        Returns:
            Loaded tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=str(self.config.cache_dir),
            token=self.config.hf_token,
            trust_remote_code=True,
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _load_model(self) -> PreTrainedModel:
        """Load the model with optional quantization.
        
        Returns:
            Loaded model
        """
        # Configure quantization if enabled
        quantization_config = None
        # if self.config.use_quantization:
        #     logger.info(
        #         f"Using {self.config.quantization_bits}-bit quantization"
        #     )
        #     quantization_config = BitsAndBytesConfig(
        #         load_in_4bit=(self.config.quantization_bits == 4),
        #         load_in_8bit=(self.config.quantization_bits == 8),
        #         bnb_4bit_compute_dtype=torch.float16,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_use_double_quant=True,
        #     )
        
        # Determine device map
        device_map = self._get_device_map()
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            cache_dir=str(self.config.cache_dir),
            token=self.config.hf_token,
            #quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Set to evaluation mode
        model.eval()
        
        return model
    
    def _get_device_map(self) -> str:
        """Determine the appropriate device map.
        
        Returns:
            Device map string
        """
        if self.config.device == "auto":
            return "auto"
        elif self.config.device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
            return "cuda"
        else:
            return "cpu"
    
    def unload(self) -> None:
        """Unload the model and tokenizer to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model and tokenizer unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None and self._tokenizer is not None


