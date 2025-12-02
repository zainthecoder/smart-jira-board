#!/bin/bash
# Setup script for Llama 3 8B Inference

set -e

echo "🚀 Setting up Llama 3 8B Inference Environment"
echo "=============================================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"




# Create cache directory
echo ""
echo "📁 Creating cache directory..."
mkdir -p model_cache
echo "   ✅ Cache directory created"

echo ""
echo "=============================================="
echo "✨ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your Hugging Face token"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the example: python main.py"
echo ""
echo "For more examples, run: python example.py"
echo "=============================================="


