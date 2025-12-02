"""Main entry point for Llama 3 8B inference."""

import logging
from typing import List, Dict

from src.config import InferenceConfig
from src.model_loader import ModelLoader
from src.inference_engine import InferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate Llama 3 inference."""
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = InferenceConfig()
        logger.info(f"Config: {config}")
        
        # Initialize model loader
        logger.info("Initializing model loader...")
        loader = ModelLoader(config)
        
        # Load model and tokenizer
        logger.info("Loading model (this may take a few minutes)...")
        model, tokenizer = loader.load()
        
        # Initialize inference engine
        logger.info("Initializing inference engine...")
        engine = InferenceEngine(model, tokenizer, config)
        
        # Example 1: Simple text generation
        logger.info("\n" + "="*50)
        logger.info("Example 1: Simple Text Generation")
        logger.info("="*50)
        
        prompt = "The future of artificial intelligence is"
        logger.info(f"Prompt: {prompt}")
        
        response = engine.generate(
            prompt=prompt,
            max_length=100,
            temperature=0.7,
        )
        logger.info(f"Response: {response}")
        
        # Example 2: Chat-style interaction
        logger.info("\n" + "="*50)
        logger.info("Example 2: Chat Interaction")
        logger.info("="*50)
        
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What are the benefits of using Python?"},
        ]
        
        logger.info(f"Messages: {messages}")
        
        chat_response = engine.chat(
            messages=messages,
            max_length=200,
            temperature=0.7,
        )
        logger.info(f"Response: {chat_response}")
        
        # Example 3: Batch generation
        logger.info("\n" + "="*50)
        logger.info("Example 3: Batch Generation")
        logger.info("="*50)
        
        prompts = [
            "Write a haiku about coding:",
            "Explain machine learning in one sentence:",
        ]
        
        logger.info(f"Prompts: {prompts}")
        
        batch_responses = engine.batch_generate(
            prompts=prompts,
            max_length=100,
            temperature=0.8,
        )
        
        for i, (prompt, response) in enumerate(zip(prompts, batch_responses), 1):
            logger.info(f"\nBatch {i}:")
            logger.info(f"  Prompt: {prompt}")
            logger.info(f"  Response: {response}")
        
        logger.info("\n" + "="*50)
        logger.info("Inference completed successfully!")
        logger.info("="*50)
        
        # Clean up
        loader.unload()
        
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()


