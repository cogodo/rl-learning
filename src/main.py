#!/usr/bin/env python3
"""
Main entry point for RL experiments.
Reads configuration and orchestrates experiment runs.
"""

import yaml
import argparse
from pathlib import Path
from .runners.experiment import run_experiment
from .utils.seeding import set_global_seed

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Run RL experiments')
    parser.add_argument('--config', type=str, default='configs/defaults.yaml',
                       help='Path to configuration file')
    parser.add_argument('--sweep', action='store_true',
                       help='Run parameter sweep')
    parser.add_argument('--seed', type=int, default=41,
                       help='Seed for random number generator')
    parser.add_argument('--eval_only', action='store_true',
                       help='Run evaluation only')

    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    set_global_seed(args.seed)
    if args.sweep:
        # Load sweep configuration
        sweep_config = load_config('configs/Sweep.yaml')
        # TODO: Implement sweep logic
        print("Sweep mode not yet implemented")
    else:
        # Run single experiment
        run_experiment(config)


if __name__ == "__main__":
    main() 