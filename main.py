#!/usr/bin/env python3
"""
DCGAN Main Entry Point

Usage:
    python main.py train                    # Start training
    python main.py train --resume checkpoints/checkpoint_epoch_0010.pth  # Resume training
    python main.py evaluate --checkpoint checkpoints/checkpoint_final.pth  # Generate samples
    python main.py evaluate --checkpoint checkpoints/checkpoint_final.pth --interpolation  # Generate interpolations
    python main.py evaluate --sample_sheet  # Create sample sheet from all checkpoints
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'train':
        from train import main
        # Remove 'train' from argv so train.py's argparse works
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main()
    
    elif command == 'evaluate':
        from evaluate import main as eval_main
        # Remove 'evaluate' from argv
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        eval_main()
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)