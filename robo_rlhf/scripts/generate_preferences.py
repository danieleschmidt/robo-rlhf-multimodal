"""
Script for generating preference pairs from demonstrations.
"""

import argparse
from pathlib import Path

from robo_rlhf.preference.pair_generator import PreferencePairGenerator


def main(args: argparse.Namespace) -> None:
    """Main preference pair generation function."""
    print(f"Generating preference pairs from: {args.demos}")
    print(f"Using {args.strategy} sampling strategy")
    
    # Create generator
    generator = PreferencePairGenerator(
        demo_dir=args.demos,
        pair_selection=args.strategy,
        seed=42
    )
    
    # Generate pairs
    pairs = generator.generate_pairs(
        num_pairs=args.pairs,
        segment_length=args.segment_length
    )
    
    # Save pairs
    generator.save_pairs(pairs, args.output)
    
    # Print statistics
    print(f"\n=== Generation Complete ===")
    print(f"Generated {len(pairs)} preference pairs")
    print(f"Segment length: {args.segment_length} frames")
    print(f"Selection strategy: {args.strategy}")
    print(f"Pairs saved to: {args.output}")
    
    # Show strategy-specific stats
    strategies = [pair.metadata.get("strategy", "unknown") for pair in pairs]
    print(f"Strategy distribution: {dict(zip(*zip(*[(s, strategies.count(s)) for s in set(strategies)])))}") if strategies else print("No strategy information available")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate preference pairs")
    
    parser.add_argument("--demos", required=True, help="Demonstrations directory")
    parser.add_argument("--pairs", type=int, default=100, help="Number of pairs")
    parser.add_argument("--strategy", default="diversity", help="Selection strategy")
    parser.add_argument("--output", default="data/preference_pairs.pkl", help="Output file")
    parser.add_argument("--segment-length", type=int, default=50, help="Segment length")
    
    args = parser.parse_args()
    main(args)