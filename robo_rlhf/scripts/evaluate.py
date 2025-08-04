"""
Script for evaluating trained policies.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch

from robo_rlhf.envs import make_env
from robo_rlhf.models.actors import VisionLanguageActor


class PolicyEvaluator:
    """Evaluates trained policies on environments."""
    
    def __init__(self, policy_path: str, env_name: str):
        """Initialize evaluator with policy and environment."""
        self.env = make_env(env_name)
        
        # Load policy (simplified - in practice would need proper model loading)
        self.policy = VisionLanguageActor()  # Would load from config
        checkpoint = torch.load(policy_path, map_location="cpu")
        if "policy_state" in checkpoint:
            self.policy.load_state_dict(checkpoint["policy_state"])
        else:
            self.policy.load_state_dict(checkpoint)
        
        self.policy.eval()
        
    def evaluate(
        self,
        num_episodes: int = 10,
        render: bool = False,
        max_steps: int = 1000
    ) -> dict:
        """Evaluate policy on environment."""
        results = {
            "episodes": [],
            "success_rate": 0.0,
            "average_return": 0.0,
            "average_length": 0.0,
            "completion_times": []
        }
        
        successful_episodes = 0
        total_return = 0.0
        total_length = 0
        
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            
            obs, info = self.env.reset()
            episode_return = 0.0
            episode_length = 0
            start_time = time.time()
            
            for step in range(max_steps):
                # Get action from policy
                if isinstance(obs, dict):
                    rgb = obs.get("pixels", obs.get("rgb", np.zeros((64, 64, 3))))
                    proprio = obs.get("robot_state", obs.get("proprioception", np.zeros(7)))
                else:
                    rgb = np.zeros((64, 64, 3))
                    proprio = obs if len(obs.shape) == 1 else np.zeros(7)
                
                # Ensure proper shapes
                if len(rgb.shape) == 3:
                    rgb = rgb.transpose(2, 0, 1)  # HWC -> CHW
                rgb = rgb.astype(np.float32) / 255.0
                
                action = self.policy.get_action(rgb, proprio)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_return += reward
                episode_length += 1
                
                if render:
                    self.env.render()
                
                if terminated or truncated:
                    break
            
            completion_time = time.time() - start_time
            success = info.get("success", terminated and not truncated)
            
            if success:
                successful_episodes += 1
            
            total_return += episode_return
            total_length += episode_length
            
            episode_result = {
                "episode": episode + 1,
                "return": episode_return,
                "length": episode_length,
                "success": success,
                "completion_time": completion_time
            }
            results["episodes"].append(episode_result)
            results["completion_times"].append(completion_time)
            
            print(f"  Return: {episode_return:.2f}, Length: {episode_length}, "
                  f"Success: {success}, Time: {completion_time:.2f}s")
        
        # Calculate summary statistics
        results["success_rate"] = successful_episodes / num_episodes
        results["average_return"] = total_return / num_episodes
        results["average_length"] = total_length / num_episodes
        results["average_completion_time"] = np.mean(results["completion_times"])
        results["std_completion_time"] = np.std(results["completion_times"])
        
        return results


def main(args: argparse.Namespace) -> None:
    """Main evaluation function."""
    print(f"Evaluating policy: {args.policy}")
    print(f"Environment: {args.env}")
    
    # Create evaluator
    evaluator = PolicyEvaluator(args.policy, args.env)
    
    # Run evaluation
    results = evaluator.evaluate(
        num_episodes=args.episodes,
        render=args.render
    )
    
    # Print results
    print(f"\n=== Evaluation Results ===")
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Average return: {results['average_return']:.2f}")
    print(f"Average length: {results['average_length']:.1f} steps")
    print(f"Average completion time: {results['average_completion_time']:.2f}s")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained policy")
    
    parser.add_argument("--policy", required=True, help="Path to policy checkpoint")
    parser.add_argument("--env", required=True, help="Environment name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render evaluation")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    main(args)