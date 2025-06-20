"""
Setup script for easy project initialization
"""
import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_gpu():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️ CUDA not available, using CPU")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet")
        return False

def main():
    """Main setup function"""
    print("🚀 Self-Healing Classification System Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check GPU availability
    check_gpu()
    
    # Create necessary directories
    directories = ["logs", "models", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("\n🎯 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python finetune_model.py' to train the model")
    print("2. Run 'python main.py' to start the classification system")
    print("\nFor help, run 'python main.py --help'")

if __name__ == "__main__":
    main()