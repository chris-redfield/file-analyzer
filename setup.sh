#!/bin/bash
# Setup script for AI File Processor

set -e

echo "🚀 Setting up AI File Processor..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Setup Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "🤖 Ollama not found. Would you like to install it? (y/n)"
    read -r install_ollama
    if [[ $install_ollama == "y" || $install_ollama == "Y" ]]; then
        echo "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        
        echo "Pulling required model (llama3.2-vision)..."
        ollama pull llama3.2-vision
    fi
else
    echo "✅ Ollama is already installed"
    
    # Check if the required model is available
    if ! ollama list | grep -q "llama3.2-vision"; then
        echo "📥 Pulling required model (llama3.2-vision)..."
        ollama pull llama3.2-vision
    fi
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "✅ Please edit .env file to configure your settings"
else
    echo "✅ .env file already exists"
fi

# Create output directory
mkdir -p ai_processing_results

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file to configure your preferences"
echo "2. Run the processor: python3 ai_file_processor.py"
echo ""
echo "For OpenAI mode:"
echo "1. Set USE_OLLAMA=false in .env"
echo "2. Add your OpenAI API key to .env"
echo ""
echo "For more help, check the comments in the Python script."