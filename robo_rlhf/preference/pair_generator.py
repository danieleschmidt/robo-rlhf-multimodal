"""
Generate preference pairs from collected demonstrations.
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from datetime import datetime
import pickle

from robo_rlhf.preference.models import PreferencePair, Segment


class PreferencePairGenerator:
    """
    Generate preference pairs for human annotation.
    
    Supports various sampling strategies for creating diverse pairs.
    """
    
    def __init__(
        self,
        demo_dir: str,
        pair_selection: str = "diversity_sampling",
        seed: Optional[int] = None
    ):
        """
        Initialize preference pair generator.
        
        Args:
            demo_dir: Directory containing demonstrations
            pair_selection: Strategy for selecting pairs
                - 'random': Random pair selection
                - 'diversity_sampling': Maximize diversity in pairs
                - 'uncertainty_sampling': Focus on uncertain regions
                - 'disagreement_sampling': Sample where models disagree
            seed: Random seed for reproducibility
        """
        self.demo_dir = Path(demo_dir)
        self.selection_strategy = pair_selection
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load demonstrations
        self.demonstrations = self._load_demonstrations()
        print(f"Loaded {len(self.demonstrations)} demonstrations")
    
    def _load_demonstrations(self) -> List[Dict[str, Any]]:
        """Load all demonstrations from directory."""
        demonstrations = []
        
        for episode_dir in sorted(self.demo_dir.glob("episode_*")):
            if not episode_dir.is_dir():
                continue
            
            # Load metadata
            meta_path = episode_dir / "metadata.json"
            if not meta_path.exists():
                continue
            
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # Load actions
            actions = np.load(episode_dir / "actions.npy")
            
            # Load observations
            observations = {}
            for obs_file in episode_dir.glob("*.npy"):
                if obs_file.stem not in ["actions", "rewards"]:
                    observations[obs_file.stem] = np.load(obs_file)
            
            # Load rewards if available
            rewards_path = episode_dir / "rewards.npy"
            rewards = np.load(rewards_path) if rewards_path.exists() else None
            
            demonstrations.append({
                "episode_id": metadata["episode_id"],
                "path": episode_dir,
                "metadata": metadata,
                "actions": actions,
                "observations": observations,
                "rewards": rewards,
                "length": len(actions)
            })
        
        return demonstrations
    
    def generate_pairs(
        self,
        num_pairs: int,
        segment_length: int = 50,
        overlap_threshold: float = 0.0,
        min_distance: int = 10
    ) -> List[PreferencePair]:
        """
        Generate preference pairs from demonstrations.
        
        Args:
            num_pairs: Number of pairs to generate
            segment_length: Length of each segment in frames
            overlap_threshold: Maximum allowed temporal overlap (0-1)
            min_distance: Minimum frame distance between segments
            
        Returns:
            List of preference pairs
        """
        pairs = []
        
        if self.selection_strategy == "random":
            pairs = self._random_sampling(num_pairs, segment_length, min_distance)
        elif self.selection_strategy == "diversity_sampling":
            pairs = self._diversity_sampling(num_pairs, segment_length, min_distance)
        elif self.selection_strategy == "uncertainty_sampling":
            pairs = self._uncertainty_sampling(num_pairs, segment_length, min_distance)
        elif self.selection_strategy == "disagreement_sampling":
            pairs = self._disagreement_sampling(num_pairs, segment_length, min_distance)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
        
        # Filter overlapping pairs if needed
        if overlap_threshold < 1.0:
            pairs = self._filter_overlapping(pairs, overlap_threshold)
        
        print(f"Generated {len(pairs)} preference pairs")
        return pairs
    
    def _random_sampling(
        self,
        num_pairs: int,
        segment_length: int,
        min_distance: int
    ) -> List[PreferencePair]:
        """Random pair sampling."""
        pairs = []
        
        for i in range(num_pairs):
            # Select two random demonstrations
            if len(self.demonstrations) < 2:
                demo1 = demo2 = random.choice(self.demonstrations)
            else:
                demo1, demo2 = random.sample(self.demonstrations, 2)
            
            # Extract random segments
            seg1 = self._extract_segment(demo1, segment_length)
            seg2 = self._extract_segment(demo2, segment_length)
            
            # Create pair
            pair = PreferencePair(
                pair_id=f"pair_{i:05d}",
                segment_a=seg1,
                segment_b=seg2,
                metadata={
                    "strategy": "random",
                    "timestamp": datetime.now().isoformat()
                }
            )
            pairs.append(pair)
        
        return pairs
    
    def _diversity_sampling(
        self,
        num_pairs: int,
        segment_length: int,
        min_distance: int
    ) -> List[PreferencePair]:
        """
        Diversity-based sampling to maximize coverage.
        
        Selects pairs that are maximally different from each other.
        """
        pairs = []
        selected_segments = []
        
        for i in range(num_pairs):
            # Find most diverse segment pair
            best_pair = None
            best_diversity = -float('inf')
            
            for _ in range(100):  # Sample candidates
                demo1 = random.choice(self.demonstrations)
                demo2 = random.choice(self.demonstrations)
                
                seg1 = self._extract_segment(demo1, segment_length)
                seg2 = self._extract_segment(demo2, segment_length)
                
                # Calculate diversity score
                diversity = self._calculate_diversity(seg1, seg2, selected_segments)
                
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_pair = (seg1, seg2)
            
            if best_pair:
                seg1, seg2 = best_pair
                pair = PreferencePair(
                    pair_id=f"pair_{i:05d}",
                    segment_a=seg1,
                    segment_b=seg2,
                    metadata={
                        "strategy": "diversity",
                        "diversity_score": best_diversity,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                pairs.append(pair)
                selected_segments.extend([seg1, seg2])
        
        return pairs
    
    def _uncertainty_sampling(
        self,
        num_pairs: int,
        segment_length: int,
        min_distance: int
    ) -> List[PreferencePair]:
        """
        Uncertainty-based sampling for active learning.
        
        Focuses on regions where reward model is uncertain.
        """
        pairs = []
        
        for i in range(num_pairs):
            # In a real implementation, this would use a trained reward model
            # to identify uncertain regions
            
            # For now, sample from demonstrations with high variance in rewards
            high_variance_demos = self._get_high_variance_demos()
            
            if len(high_variance_demos) >= 2:
                demo1, demo2 = random.sample(high_variance_demos, 2)
            else:
                demo1 = demo2 = random.choice(self.demonstrations)
            
            seg1 = self._extract_segment(demo1, segment_length, prefer_uncertain=True)
            seg2 = self._extract_segment(demo2, segment_length, prefer_uncertain=True)
            
            pair = PreferencePair(
                pair_id=f"pair_{i:05d}",
                segment_a=seg1,
                segment_b=seg2,
                metadata={
                    "strategy": "uncertainty",
                    "timestamp": datetime.now().isoformat()
                }
            )
            pairs.append(pair)
        
        return pairs
    
    def _disagreement_sampling(
        self,
        num_pairs: int,
        segment_length: int,
        min_distance: int
    ) -> List[PreferencePair]:
        """
        Disagreement-based sampling for ensemble learning.
        
        Samples where multiple models disagree on preferences.
        """
        pairs = []
        
        for i in range(num_pairs):
            # In production, this would use ensemble predictions
            # For now, use a heuristic based on action diversity
            
            demo1 = random.choice(self.demonstrations)
            demo2 = random.choice(self.demonstrations)
            
            # Find segments with high action variance (proxy for disagreement)
            seg1 = self._extract_segment(demo1, segment_length, prefer_diverse_actions=True)
            seg2 = self._extract_segment(demo2, segment_length, prefer_diverse_actions=True)
            
            pair = PreferencePair(
                pair_id=f"pair_{i:05d}",
                segment_a=seg1,
                segment_b=seg2,
                metadata={
                    "strategy": "disagreement",
                    "timestamp": datetime.now().isoformat()
                }
            )
            pairs.append(pair)
        
        return pairs
    
    def _extract_segment(
        self,
        demo: Dict[str, Any],
        length: int,
        prefer_uncertain: bool = False,
        prefer_diverse_actions: bool = False
    ) -> Segment:
        """Extract a segment from a demonstration."""
        max_start = max(0, demo["length"] - length)
        
        if max_start == 0:
            start_idx = 0
        elif prefer_uncertain and demo["rewards"] is not None:
            # Find region with high reward variance
            reward_vars = []
            for i in range(max_start):
                segment_rewards = demo["rewards"][i:i+length]
                reward_vars.append(np.var(segment_rewards))
            start_idx = np.argmax(reward_vars)
        elif prefer_diverse_actions:
            # Find region with diverse actions
            action_vars = []
            for i in range(max_start):
                segment_actions = demo["actions"][i:i+length]
                action_vars.append(np.mean(np.var(segment_actions, axis=0)))
            start_idx = np.argmax(action_vars)
        else:
            start_idx = random.randint(0, max_start)
        
        end_idx = min(start_idx + length, demo["length"])
        
        # Extract segment data
        segment_obs = {}
        for key, obs in demo["observations"].items():
            segment_obs[key] = obs[start_idx:end_idx]
        
        return Segment(
            episode_id=demo["episode_id"],
            start_frame=start_idx,
            end_frame=end_idx,
            observations=segment_obs,
            actions=demo["actions"][start_idx:end_idx],
            rewards=demo["rewards"][start_idx:end_idx] if demo["rewards"] is not None else None,
            metadata=demo["metadata"]
        )
    
    def _calculate_diversity(
        self,
        seg1: Segment,
        seg2: Segment,
        existing_segments: List[Segment]
    ) -> float:
        """Calculate diversity score between segments."""
        # Simple diversity based on action differences
        action_diff = np.mean(np.abs(seg1.actions - seg2.actions))
        
        # Penalize similarity to existing segments
        if existing_segments:
            similarities = []
            for seg in existing_segments:
                if len(seg.actions) == len(seg1.actions):
                    sim1 = np.mean(np.abs(seg.actions - seg1.actions))
                    similarities.append(sim1)
                if len(seg.actions) == len(seg2.actions):
                    sim2 = np.mean(np.abs(seg.actions - seg2.actions))
                    similarities.append(sim2)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                diversity = action_diff - 0.5 * avg_similarity
            else:
                diversity = action_diff
        else:
            diversity = action_diff
        
        return diversity
    
    def _get_high_variance_demos(self) -> List[Dict[str, Any]]:
        """Get demonstrations with high reward variance."""
        demos_with_variance = []
        
        for demo in self.demonstrations:
            if demo["rewards"] is not None and len(demo["rewards"]) > 0:
                variance = np.var(demo["rewards"])
                demos_with_variance.append((variance, demo))
        
        if not demos_with_variance:
            return self.demonstrations
        
        # Sort by variance and return top 50%
        demos_with_variance.sort(key=lambda x: x[0], reverse=True)
        top_half = len(demos_with_variance) // 2
        return [demo for _, demo in demos_with_variance[:top_half]]
    
    def _filter_overlapping(
        self,
        pairs: List[PreferencePair],
        threshold: float
    ) -> List[PreferencePair]:
        """Filter pairs with too much temporal overlap."""
        filtered = []
        
        for pair in pairs:
            # Check overlap with existing pairs
            has_overlap = False
            
            for existing in filtered:
                overlap_a = self._calculate_overlap(
                    pair.segment_a,
                    existing.segment_a
                ) if pair.segment_a.episode_id == existing.segment_a.episode_id else 0
                
                overlap_b = self._calculate_overlap(
                    pair.segment_b,
                    existing.segment_b
                ) if pair.segment_b.episode_id == existing.segment_b.episode_id else 0
                
                if max(overlap_a, overlap_b) > threshold:
                    has_overlap = True
                    break
            
            if not has_overlap:
                filtered.append(pair)
        
        return filtered
    
    def _calculate_overlap(self, seg1: Segment, seg2: Segment) -> float:
        """Calculate temporal overlap between segments."""
        if seg1.episode_id != seg2.episode_id:
            return 0.0
        
        overlap_start = max(seg1.start_frame, seg2.start_frame)
        overlap_end = min(seg1.end_frame, seg2.end_frame)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_frames = overlap_end - overlap_start
        min_length = min(
            seg1.end_frame - seg1.start_frame,
            seg2.end_frame - seg2.start_frame
        )
        
        return overlap_frames / min_length if min_length > 0 else 0.0
    
    def save_pairs(self, pairs: List[PreferencePair], output_path: str) -> None:
        """Save preference pairs to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(pairs, f)
        
        print(f"Saved {len(pairs)} pairs to {output_path}")
    
    def load_pairs(self, path: str) -> List[PreferencePair]:
        """Load preference pairs from file."""
        with open(path, 'rb') as f:
            pairs = pickle.load(f)
        
        print(f"Loaded {len(pairs)} pairs from {path}")
        return pairs