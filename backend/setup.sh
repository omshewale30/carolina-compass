#!/bin/bash

# Setup script for Carolina Compass backend

set -e

echo "🚀 Setting up Carolina Compass Backend..."
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"
echo ""

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt
echo ""

# Create weights directory
echo "📁 Creating weights directory..."
mkdir -p weights
echo ""

# Check for model weights
if [ ! -f "weights/rs-152-c5-best_params.pth" ]; then
    echo "⚠️  WARNING: Model weights not found!"
    echo "   Please place your weights file at: weights/rs-152-c5-best_params.pth"
    echo "   Or set MODEL_WEIGHTS_PATH environment variable"
    echo ""
else
    echo "✅ Model weights found at weights/rs-152-c5-best_params.pth"
    echo ""
fi

# Check for model class
if [ ! -f "src/model/resnet.py" ]; then
    echo "⚠️  WARNING: ResNet152 model class not found!"
    echo "   Please copy src/model/resnet.py from your model repository"
    echo "   See MODEL_SETUP.md for detailed instructions"
    echo ""
else
    echo "✅ Model class found at src/model/resnet.py"
    echo ""
fi

echo "✨ Setup complete!"
echo ""
echo "To start the server, run:"
echo "  python main.py"
echo ""
echo "Or with uvicorn:"
echo "  uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""

