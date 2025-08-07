"""
Command-line interface for robo-rlhf-multimodal.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from robo_rlhf import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="Robo-RLHF-Multimodal: End-to-end multimodal RLHF for robotics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  robo-rlhf collect --env MujocoManipulation --episodes 100
  robo-rlhf train --config configs/rlhf_config.yaml
  robo-rlhf evaluate --policy checkpoints/best_policy.pt
        """
    )
    
    parser.add_argument(
        "--version", action="version", version=f"robo-rlhf-multimodal {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Collect command
    collect_parser = subparsers.add_parser(
        "collect", help="Collect teleoperation demonstrations"
    )
    collect_parser.add_argument(
        "--env", required=True, help="Environment name"
    )
    collect_parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to collect"
    )
    collect_parser.add_argument(
        "--device", choices=["keyboard", "spacemouse", "vr"], 
        default="keyboard", help="Input device"
    )
    collect_parser.add_argument(
        "--output", default="data/demonstrations", help="Output directory"
    )
    collect_parser.add_argument(
        "--modalities", nargs="+", 
        default=["rgb", "proprioception"],
        help="Observation modalities to record"
    )
    collect_parser.add_argument(
        "--render", action="store_true", help="Render environment during collection"
    )
    
    # Preference command
    pref_parser = subparsers.add_parser(
        "preferences", help="Generate and collect preference pairs"
    )
    pref_parser.add_argument(
        "--demos", required=True, help="Path to demonstrations directory"
    )
    pref_parser.add_argument(
        "--pairs", type=int, default=100, help="Number of preference pairs"
    )
    pref_parser.add_argument(
        "--strategy", choices=["random", "diversity", "uncertainty"],
        default="diversity", help="Pair selection strategy"
    )
    pref_parser.add_argument(
        "--output", default="data/preference_pairs.pkl", help="Output file"
    )
    pref_parser.add_argument(
        "--segment-length", type=int, default=50, help="Segment length in frames"
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train RLHF policy"
    )
    train_parser.add_argument(
        "--config", help="Training configuration file"
    )
    train_parser.add_argument(
        "--preferences", help="Path to preference pairs file"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    train_parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate"
    )
    train_parser.add_argument(
        "--output", default="checkpoints/", help="Output directory for checkpoints"
    )
    train_parser.add_argument(
        "--wandb", action="store_true", help="Use Weights & Biases logging"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate trained policy"
    )
    eval_parser.add_argument(
        "--policy", required=True, help="Path to policy checkpoint"
    )
    eval_parser.add_argument(
        "--env", required=True, help="Environment name"
    )
    eval_parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    eval_parser.add_argument(
        "--render", action="store_true", help="Render during evaluation"
    )
    eval_parser.add_argument(
        "--output", help="Output file for results"
    )
    
    # Server command
    server_parser = subparsers.add_parser(
        "server", help="Launch preference collection server"
    )
    server_parser.add_argument(
        "--pairs", required=True, help="Path to preference pairs file"
    )
    server_parser.add_argument(
        "--port", type=int, default=8080, help="Server port"
    )
    server_parser.add_argument(
        "--output", default="data/preferences.json", help="Output file for preferences"
    )
    server_parser.add_argument(
        "--annotators", nargs="+", help="List of annotator names"
    )
    
    # Quantum autonomous SDLC command
    quantum_parser = subparsers.add_parser(
        "quantum", help="Run quantum autonomous SDLC execution"
    )
    quantum_parser.add_argument(
        "--project", default=".", help="Project path for SDLC execution"
    )
    quantum_parser.add_argument(
        "--phases", nargs="+", 
        choices=["analysis", "design", "implementation", "testing", "integration", "deployment", "monitoring", "optimization"],
        help="SDLC phases to execute"
    )
    quantum_parser.add_argument(
        "--config", help="Quantum configuration file"
    )
    quantum_parser.add_argument(
        "--demo", action="store_true", help="Run comprehensive quantum demo"
    )
    quantum_parser.add_argument(
        "--optimization-target", choices=["time", "quality", "resources", "reliability"],
        default="quality", help="Primary optimization target"
    )
    quantum_parser.add_argument(
        "--auto-approve", action="store_true", help="Auto-approve autonomous actions"
    )
    
    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "collect":
            from robo_rlhf.scripts.collect_demos import main as collect_main
            collect_main(args)
        elif args.command == "preferences":
            from robo_rlhf.scripts.generate_preferences import main as pref_main
            pref_main(args)
        elif args.command == "train":
            from robo_rlhf.scripts.train_rlhf import main as train_main
            train_main(args)
        elif args.command == "evaluate":
            from robo_rlhf.scripts.evaluate import main as eval_main
            eval_main(args)
        elif args.command == "server":
            from robo_rlhf.preference.server import main as server_main
            server_main(args)
        elif args.command == "quantum":
            from robo_rlhf.quantum.cli import main as quantum_main
            quantum_main(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()