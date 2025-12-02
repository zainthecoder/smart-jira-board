#!/bin/bash
# Setup script for Llama 3 8B Inference using Conda

set -e

echo "🚀 Setting up Llama 3 8B Inference with Conda"
echo "=============================================="

ENV_NAME="llama3-inference"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

echo "✅ Conda found"

# Create conda environment
echo ""
echo "🔧 Creating conda environment: $ENV_NAME"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "   ℹ️  Environment '$ENV_NAME' already exists"
    read -p "   Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
        echo "   ✅ Old environment removed"
    else
        echo "   ℹ️  Using existing environment"
    fi
fi

if ! conda env list | grep -q "^$ENV_NAME "; then
    conda create -n $ENV_NAME python=3.10 -y
    echo "   ✅ Conda environment created"
fi

# Activate environment
echo ""
echo "🔌 Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
echo "   ✅ Environment activated"

# Install PyTorch with CUDA support (if available)
echo ""
echo "📦 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "   🎮 CUDA detected - installing PyTorch with CUDA support"
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
else
    echo "   💻 No CUDA detected - installing CPU-only PyTorch"
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi
echo "   ✅ PyTorch installed"

# Install remaining dependencies with pip
echo ""
echo "📦 Installing remaining dependencies..."
pip install -r requirements.txt
echo "   ✅ Dependencies installed"

# Create .env file if it doesn't exist
echo ""
echo "⚙️  Setting up configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "   ✅ .env file created"
    echo "   ⚠️  Please edit .env and add your Hugging Face token!"
else
    echo "   ℹ️  .env file already exists"
fi

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
echo "2. Activate the conda environment:"
echo "   conda activate $ENV_NAME"
echo "3. Run the example:"
echo "   python main.py"
echo ""
echo "To deactivate: conda deactivate"
echo "=============================================="



