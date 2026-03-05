#!/bin/bash
# ============================================================
# Thesis Project Setup: "When Do Language Models Learn to Trade?"
# Run this once to create the full project structure
# ============================================================

PROJECT_DIR="$HOME/thesis-llm-trading"

echo "Creating project structure at $PROJECT_DIR ..."

mkdir -p "$PROJECT_DIR"/{data/{raw/{news,returns},processed,labels},models/{checkpoints,configs},src/{data,models,evaluation,interpretability,utils},notebooks,thesis/{chapters,figures,tables},results/{replication,sensitivity,transfer_learning,interpretability,architecture},scripts,logs}

# Create __init__.py files for Python package structure
find "$PROJECT_DIR/src" -type d -exec touch {}/__init__.py \;

echo "Done. Project structure created."
echo ""
echo "Next steps:"
echo "  1. cd $PROJECT_DIR"
echo "  2. conda env create -f environment.yml"
echo "  3. conda activate thesis"
echo "  4. python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
