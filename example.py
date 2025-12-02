"""Advanced usage examples for Llama 3 8B inference."""

import logging
from typing import List, Dict

from src.config import InferenceConfig
from src.model_loader import ModelLoader
from src.inference_engine import InferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_creative_writing(engine: InferenceEngine) -> None:
    """Example: Creative writing with higher temperature."""
    print("\n" + "="*60)
    print("EXAMPLE: Creative Writing (High Temperature)")
    print("="*60)
    
    prompt = "Once upon a time in a digital realm,"
    
    response = engine.generate(
        prompt=prompt,
        max_length=150,
        temperature=1.2,  # Higher temperature for more creativity
        top_p=0.95,
    )
    
    print(f"Prompt: {prompt}")
    print(f"\nStory: {response}")


def example_code_generation(engine: InferenceEngine) -> None:
    """Example: Code generation with lower temperature."""
    print("\n" + "="*60)
    print("EXAMPLE: Code Generation (Low Temperature)")
    print("="*60)
    
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": "You are an expert Python programmer.",
        },
        {
            "role": "user",
            "content": "Write a Python function to calculate fibonacci numbers.",
        },
    ]
    
    response = engine.chat(
        messages=messages,
        max_length=300,
        temperature=0.2,  # Lower temperature for precise code
    )
    
    print("Task: Generate fibonacci function")
    print(f"\nResponse:\n{response}")


def example_qa_system(engine: InferenceEngine) -> None:
    """Example: Question-answering system."""
    print("\n" + "="*60)
    print("EXAMPLE: Question-Answering System")
    print("="*60)
    
    questions = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What is photosynthesis?",
    ]
    
    for i, question in enumerate(questions, 1):
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": "You are a knowledgeable assistant. Provide concise, accurate answers.",
            },
            {"role": "user", "content": question},
        ]
        
        answer = engine.chat(
            messages=messages,
            max_length=100,
            temperature=0.3,
        )
        
        print(f"\n{i}. Q: {question}")
        print(f"   A: {answer}")


def example_conversation(engine: InferenceEngine) -> None:
    """Example: Multi-turn conversation."""
    print("\n" + "="*60)
    print("EXAMPLE: Multi-turn Conversation")
    print("="*60)
    
    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a friendly AI assistant."},
        {"role": "user", "content": "What's the weather like today?"},
    ]
    
    print(f"User: {conversation[-1]['content']}")
    
    # First response
    response1 = engine.chat(messages=conversation, max_length=100)
    print(f"Assistant: {response1}")
    
    # Continue conversation
    conversation.append({"role": "assistant", "content": response1})
    conversation.append({"role": "user", "content": "That's helpful, thank you!"})
    
    print(f"\nUser: {conversation[-1]['content']}")
    
    # Second response
    response2 = engine.chat(messages=conversation, max_length=100)
    print(f"Assistant: {response2}")


def example_summarization(engine: InferenceEngine) -> None:
    """Example: Text summarization."""
    print("\n" + "="*60)
    print("EXAMPLE: Text Summarization")
    print("="*60)
    
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. Colloquially, the term 
    "artificial intelligence" is often used to describe machines (or computers) 
    that mimic "cognitive" functions that humans associate with the human mind, 
    such as "learning" and "problem solving".
    """
    
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": "You are an expert at summarizing text concisely.",
        },
        {
            "role": "user",
            "content": f"Summarize this text in one sentence:\n{long_text}",
        },
    ]
    
    summary = engine.chat(messages=messages, max_length=100, temperature=0.3)
    
    print(f"Original (truncated): {long_text[:100]}...")
    print(f"\nSummary: {summary}")


def main():
    """Run all examples."""
    try:
        # Initialize
        config = InferenceConfig()
        loader = ModelLoader(config)
        
        print("Loading model... (this may take a few minutes)")
        model, tokenizer = loader.load()
        
        engine = InferenceEngine(model, tokenizer, config)
        
        # Run examples
        example_creative_writing(engine)
        example_code_generation(engine)
        example_qa_system(engine)
        example_conversation(engine)
        example_summarization(engine)
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
        # Cleanup
        loader.unload()
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()


