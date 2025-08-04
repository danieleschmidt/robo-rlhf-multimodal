#!/usr/bin/env python3
"""
Basic usage example for Robo-RLHF-Multimodal.

This example demonstrates the complete pipeline:
1. Collect demonstrations
2. Generate preference pairs
3. Collect human preferences
4. Train RLHF policy
5. Evaluate the trained policy
"""

import numpy as np
from pathlib import Path

# Import core components
from robo_rlhf import (
    TeleOpCollector,
    PreferencePairGenerator, 
    PreferenceServer,
    MultimodalRLHF,
    VisionLanguageActor,
    make_env
)


def main():
    """Run the complete RLHF pipeline."""
    print("🤖 Robo-RLHF-Multimodal Basic Usage Example")
    print("=" * 50)
    
    # Configuration
    env_name = "mujoco_manipulation"
    demo_dir = "data/demonstrations"
    pairs_file = "data/preference_pairs.pkl"
    prefs_file = "data/preferences.json"
    checkpoint_dir = "checkpoints/"
    
    # Step 1: Collect Demonstrations
    print("\n📹 Step 1: Collecting Demonstrations")
    print("-" * 30)
    
    # Create environment
    env = make_env(env_name, task="pick_and_place")
    print(f"Environment: {env_name}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Create collector
    collector = TeleOpCollector(
        env=env,
        modalities=["rgb", "proprioception"],
        device="keyboard",  # Change to "spacemouse" or "vr" if available
        recording_fps=30
    )
    
    # Collect demonstrations (interactive)
    print("Starting teleoperation collection...")
    print("Use keyboard controls to demonstrate the task")
    
    demonstrations = collector.collect(
        num_episodes=5,  # Start with a small number
        save_dir=demo_dir,
        render=True
    )
    
    print(f"✅ Collected {len(demonstrations)} demonstrations")
    
    # Step 2: Generate Preference Pairs
    print("\n🔗 Step 2: Generating Preference Pairs")
    print("-" * 30)
    
    generator = PreferencePairGenerator(
        demo_dir=demo_dir,
        pair_selection="diversity_sampling",  # Use diverse pairs
        seed=42
    )
    
    pairs = generator.generate_pairs(
        num_pairs=20,  # Small number for example
        segment_length=30
    )
    
    # Save pairs
    generator.save_pairs(pairs, pairs_file)
    print(f"✅ Generated {len(pairs)} preference pairs")
    
    # Step 3: Collect Human Preferences (Optional - can be done separately)
    print("\n👥 Step 3: Collecting Human Preferences")
    print("-" * 30)
    print("You can now launch the preference collection server:")
    print(f"robo-rlhf server --pairs {pairs_file} --output {prefs_file}")
    print("Or run it programmatically (uncomment the code below)")
    
    # Uncomment to run preference server
    # server = PreferenceServer(
    #     pairs=pairs,
    #     port=8080,
    #     annotators=["expert1", "expert2"],
    #     output_file=prefs_file
    # )
    # print("🌐 Server running at http://localhost:8080")
    # server.run()  # This will block until server is stopped
    
    # For this example, let's create mock preferences
    print("Creating mock preferences for demonstration...")
    import random
    from robo_rlhf.preference.models import PreferenceChoice
    
    for pair in pairs:
        # Randomly assign preferences
        choice = random.choice([PreferenceChoice.SEGMENT_A, PreferenceChoice.SEGMENT_B])
        confidence = random.uniform(0.6, 0.9)
        pair.add_preference("mock_annotator", choice, confidence)
    
    print(f"✅ Added mock preferences to {len(pairs)} pairs")
    
    # Step 4: Train RLHF Policy
    print("\n🎯 Step 4: Training RLHF Policy")
    print("-" * 30)
    
    # Create policy model
    policy = VisionLanguageActor(
        vision_encoder="resnet18",
        proprioception_dim=7,
        action_dim=7,
        hidden_dim=256
    )
    
    print(f"Policy model: {sum(p.numel() for p in policy.parameters()):,} parameters")
    
    # Create RLHF trainer
    trainer = MultimodalRLHF(
        model=policy,
        preferences=pairs,
        reward_model="bradley_terry",
        optimizer="adamw",
        lr=3e-4,
        use_wandb=False  # Set to True to log to W&B
    )
    
    # Train the policy
    print("Starting RLHF training...")
    stats = trainer.train(
        epochs=20,  # Small number for example
        batch_size=16,
        validation_split=0.2,
        checkpoint_dir=checkpoint_dir,
        reward_epochs=10,
        policy_epochs=20
    )
    
    print("✅ Training completed!")
    print(f"Final validation accuracy: {stats['validation_accuracy'][-1]:.3f}")
    
    # Step 5: Evaluate Policy
    print("\n📊 Step 5: Evaluating Policy")
    print("-" * 30)
    
    # Reset environment for evaluation
    env = make_env(env_name, task="pick_and_place")
    
    # Run evaluation episodes
    num_eval_episodes = 3
    returns = []
    
    for episode in range(num_eval_episodes):
        obs, info = env.reset()
        episode_return = 0
        done = False
        step = 0
        max_steps = 100
        
        print(f"Evaluating episode {episode + 1}/{num_eval_episodes}")
        
        while not done and step < max_steps:
            # Get action from policy
            if isinstance(obs, dict):
                rgb = obs.get("pixels", obs.get("rgb", np.zeros((3, 64, 64))))
                proprio = obs.get("robot_state", obs.get("proprioception", np.zeros(7)))
            else:
                rgb = np.zeros((3, 64, 64))
                proprio = obs[:7] if len(obs) >= 7 else np.zeros(7)
            
            # Ensure proper format
            rgb = rgb.astype(np.float32)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            
            action = policy.get_action(rgb, proprio)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated
            step += 1
        
        returns.append(episode_return)
        success = info.get("success", False)
        print(f"  Episode {episode + 1}: Return={episode_return:.2f}, Success={success}, Steps={step}")
    
    # Print evaluation summary
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    print(f"\n✅ Evaluation Results:")
    print(f"Average Return: {avg_return:.2f} ± {std_return:.2f}")
    print(f"Episodes: {num_eval_episodes}")
    
    # Step 6: Summary
    print("\n🎉 Pipeline Complete!")
    print("-" * 30)
    print("Summary:")
    print(f"• Collected {len(demonstrations)} demonstrations")
    print(f"• Generated {len(pairs)} preference pairs")
    print(f"• Trained RLHF policy with {stats['validation_accuracy'][-1]:.1%} accuracy")
    print(f"• Achieved {avg_return:.2f} average return on evaluation")
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print(f"Data saved to: {demo_dir}")
    
    print("\n🚀 Next Steps:")
    print("1. Collect more demonstrations for better coverage")
    print("2. Use the web interface for human preference annotation")
    print("3. Experiment with different model architectures")
    print("4. Deploy the policy on a real robot!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()