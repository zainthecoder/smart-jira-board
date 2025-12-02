"""Inference engine for Llama 3 text generation."""

import logging
from typing import Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import InferenceConfig

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Handles text generation inference with Llama 3."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: InferenceConfig,
    ):
        """Initialize the inference engine.
        
        Args:
            model: Loaded language model
            tokenizer: Loaded tokenizer
            config: Configuration object
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        **kwargs,
    ) -> Union[str, List[str]]:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text (overrides config)
            temperature: Sampling temperature (overrides config)
            top_p: Nucleus sampling parameter (overrides config)
            top_k: Top-k sampling parameter (overrides config)
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            **kwargs: Additional arguments passed to model.generate()
        
        Returns:
            Generated text (single string if num_return_sequences=1, else list)
        """
        # Use config defaults if not specified
        max_length = max_length or self.config.max_length
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k
        
        logger.debug(f"Generating with prompt: {prompt[:50]}...")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )
            
            # Decode outputs
            generated_texts = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )
            
            # Remove the prompt from generated text
            generated_texts = [
                text[len(prompt):].strip() if text.startswith(prompt) else text
                for text in generated_texts
            ]
            
            logger.debug(f"Generated {len(generated_texts)} sequence(s)")
            
            return generated_texts[0] if num_return_sequences == 1 else generated_texts
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}") from e
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate a chat response using the chat template.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     Example: [{"role": "user", "content": "Hello!"}]
            max_length: Maximum length of generated text (overrides config)
            temperature: Sampling temperature (overrides config)
            top_p: Nucleus sampling parameter (overrides config)
            top_k: Top-k sampling parameter (overrides config)
            **kwargs: Additional arguments passed to model.generate()
        
        Returns:
            Generated response text
        """
        try:
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # Generate response
            response = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1,
                **kwargs,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            raise RuntimeError(f"Chat generation failed: {e}") from e
    
    def batch_generate(
        self,
        prompts: List[str],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum length of generated text (overrides config)
            temperature: Sampling temperature (overrides config)
            top_p: Nucleus sampling parameter (overrides config)
            top_k: Top-k sampling parameter (overrides config)
            **kwargs: Additional arguments passed to model.generate()
        
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1,
                **kwargs,
            )
            results.append(result)
        
        return results


