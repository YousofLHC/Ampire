#!/bin/bash

# Exit on error
set -e

echo "🚀 Setting up the development environment for Ampire..."

# 1️⃣ Check if virtual environment exists, otherwise create one
if [ ! -d ".venv" ]; then
    echo "📁 Creating virtual environment (.venv)..."
    python -m venv .venv
else
    echo "✅ Virtual environment (.venv) already exists."
fi

# 2️⃣ Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate || source .venv/Scripts/activate

# 3️⃣ Upgrade pip and install dependencies
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 4️⃣ Install Ampire in editable mode
echo "🔄 Installing Ampire in editable mode..."
pip install -e .

# 5️⃣ Confirm successful setup
echo "🎉 Ampire development environment is ready!"
echo "Run 'source .venv/bin/activate' (Linux/Mac) or '.venv\\Scripts\\activate' (Windows) to activate the virtual environment."
