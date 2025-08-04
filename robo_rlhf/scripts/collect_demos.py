"""
Script for collecting teleoperation demonstrations.
"""

import argparse
from pathlib import Path
import gymnasium as gym

from robo_rlhf.collectors.base import TeleOpCollector
from robo_rlhf.envs import make_env


def main(args: argparse.Namespace) -> None:
    """Main demonstration collection function."""
    print(f"Starting demonstration collection for environment: {args.env}")
    print(f"Collecting {args.episodes} episodes using {args.device} device")
    
    # Create environment
    env = make_env(args.env)
    
    # Create collector
    collector = TeleOpCollector(
        env=env,
        modalities=args.modalities,
        device=args.device,
        recording_fps=30
    )
    
    # Collect demonstrations
    demonstrations = collector.collect(
        num_episodes=args.episodes,
        save_dir=args.output,
        render=args.render
    )
    
    # Print summary statistics
    successful_demos = sum(1 for demo in demonstrations if demo.success)
    total_steps = sum(len(demo.actions) for demo in demonstrations)
    avg_duration = sum(demo.duration for demo in demonstrations) / len(demonstrations)
    
    print(f"\n=== Collection Complete ===")
    print(f"Total episodes: {len(demonstrations)}")
    print(f"Successful episodes: {successful_demos} ({100 * successful_demos / len(demonstrations):.1f}%)")
    print(f"Total steps: {total_steps}")
    print(f"Average duration: {avg_duration:.2f}s")
    print(f"Data saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect teleoperation demonstrations")
    
    parser.add_argument("--env", required=True, help="Environment name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--device", default="keyboard", help="Input device")
    parser.add_argument("--output", default="data/demonstrations", help="Output directory")
    parser.add_argument("--modalities", nargs="+", default=["rgb", "proprioception"])
    parser.add_argument("--render", action="store_true", help="Render environment")
    
    args = parser.parse_args()
    main(args)