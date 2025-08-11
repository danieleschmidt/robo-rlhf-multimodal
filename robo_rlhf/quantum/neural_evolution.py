"""
Neural Evolution Engine for Autonomous SDLC Enhancement.

Advanced AI-powered system that uses neural networks and evolutionary algorithms
to continuously optimize the SDLC process, learning from historical data and
predicting optimal configurations for future projects.
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time

from robo_rlhf.core import get_logger, get_config
from robo_rlhf.core.exceptions import RoboRLHFError
from robo_rlhf.core.performance import PerformanceMonitor, CacheManager, optimize_memory
from robo_rlhf.core.validators import validate_numeric, validate_dict


class NeuralArchitecture(Enum):
    """Neural network architectures for different SDLC optimization tasks."""
    TRANSFORMER = "transformer"
    LSTM = "lstm" 
    GRU = "gru"
    ATTENTION = "attention"
    HYBRID = "hybrid"


class EvolutionStrategy(Enum):
    """Evolution strategies for neural network optimization."""
    GENETIC_ALGORITHM = "genetic"
    DIFFERENTIAL_EVOLUTION = "differential" 
    PARTICLE_SWARM = "particle_swarm"
    NEUROEVOLUTION = "neuroevolution"
    HYBRID_EVOLUTION = "hybrid_evolution"


@dataclass
class NeuralGenome:
    """Genome representation for neural network evolution."""
    id: str
    architecture: NeuralArchitecture
    layers: List[int]
    activation_functions: List[str]
    learning_rate: float
    dropout_rate: float
    batch_size: int
    epochs: int
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_rate: float = 0.1


@dataclass
class SDLCPattern:
    """Pattern learned from historical SDLC executions."""
    pattern_id: str
    project_type: str
    complexity_score: float
    optimal_configuration: Dict[str, Any]
    success_probability: float
    performance_metrics: Dict[str, float]
    learned_from_executions: int
    confidence_level: float


class NeuralEvolutionEngine:
    """Advanced neural evolution engine for SDLC optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = config or get_config().to_dict()
        
        # Neural evolution parameters
        self.population_size = self.config.get("neural_evolution", {}).get("population_size", 50)
        self.generations = self.config.get("neural_evolution", {}).get("generations", 100)
        self.mutation_rate = self.config.get("neural_evolution", {}).get("mutation_rate", 0.1)
        self.crossover_rate = self.config.get("neural_evolution", {}).get("crossover_rate", 0.8)
        self.elite_ratio = self.config.get("neural_evolution", {}).get("elite_ratio", 0.2)
        
        # Neural network parameters
        self.max_layers = self.config.get("neural_evolution", {}).get("max_layers", 10)
        self.max_neurons_per_layer = self.config.get("neural_evolution", {}).get("max_neurons", 512)
        self.activation_functions = ["relu", "sigmoid", "tanh", "leaky_relu", "swish"]
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.cache_manager = CacheManager(max_size=1000, ttl=3600)
        
        # Evolution state
        self.current_population: List[NeuralGenome] = []
        self.best_genome: Optional[NeuralGenome] = None
        self.evolution_history: List[Dict[str, Any]] = []
        self.learned_patterns: Dict[str, SDLCPattern] = {}
        
        # Thread pool for parallel evolution
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("NeuralEvolutionEngine initialized with advanced AI capabilities")
    
    async def evolve_optimal_sdlc_configuration(self, 
                                               project_context: Dict[str, Any],
                                               historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evolve optimal SDLC configuration using neural evolution."""
        self.logger.info("Starting neural evolution for SDLC optimization")
        
        with self.performance_monitor.measure("neural_evolution"):
            # Initialize population if needed
            if not self.current_population:
                await self._initialize_population()
            
            # Learn patterns from historical data
            await self._learn_patterns_from_history(historical_data)
            
            # Evolve population for optimal configuration
            best_genome = await self._evolve_population(project_context)
            
            # Generate optimized SDLC configuration
            optimal_config = await self._generate_sdlc_config(best_genome, project_context)
            
            # Cache result for future use
            cache_key = self._generate_cache_key(project_context)
            self.cache_manager.set(cache_key, optimal_config)
            
            self.logger.info(f"Neural evolution complete. Best fitness: {best_genome.fitness_score:.4f}")
            
            return optimal_config
    
    async def _initialize_population(self) -> None:
        """Initialize the neural evolution population."""
        self.logger.info(f"Initializing population of {self.population_size} neural genomes")
        
        self.current_population = []
        for i in range(self.population_size):
            genome = self._create_random_genome(f"gen0_ind{i}")
            self.current_population.append(genome)
        
        self.logger.info("Population initialized with diverse neural architectures")
    
    def _create_random_genome(self, genome_id: str) -> NeuralGenome:
        """Create a random neural genome."""
        # Random architecture selection
        architecture = np.random.choice(list(NeuralArchitecture))
        
        # Random layer configuration
        num_layers = np.random.randint(2, self.max_layers + 1)
        layers = [np.random.randint(32, self.max_neurons_per_layer + 1) for _ in range(num_layers)]
        
        # Random activation functions
        activations = [np.random.choice(self.activation_functions) for _ in range(num_layers)]
        
        # Random hyperparameters
        learning_rate = np.random.uniform(0.0001, 0.1)
        dropout_rate = np.random.uniform(0.0, 0.5)
        batch_size = np.random.choice([16, 32, 64, 128, 256])
        epochs = np.random.randint(10, 200)
        
        return NeuralGenome(
            id=genome_id,
            architecture=architecture,
            layers=layers,
            activation_functions=activations,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            epochs=epochs,
            mutation_rate=self.mutation_rate
        )
    
    async def _learn_patterns_from_history(self, historical_data: List[Dict[str, Any]]) -> None:
        """Learn patterns from historical SDLC executions."""
        self.logger.info(f"Learning patterns from {len(historical_data)} historical executions")
        
        if not historical_data:
            return
        
        # Cluster similar projects
        project_clusters = await self._cluster_projects(historical_data)
        
        # Extract patterns for each cluster
        for cluster_id, projects in project_clusters.items():
            pattern = await self._extract_pattern_from_cluster(cluster_id, projects)
            self.learned_patterns[pattern.pattern_id] = pattern
        
        self.logger.info(f"Learned {len(self.learned_patterns)} distinct SDLC patterns")
    
    async def _cluster_projects(self, historical_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster similar projects using AI techniques."""
        clusters = {}
        
        for project in historical_data:
            # Simple clustering based on project characteristics
            project_type = project.get("project_type", "unknown")
            complexity = project.get("complexity_score", 0.5)
            
            cluster_key = f"{project_type}_{int(complexity * 10)}"
            
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(project)
        
        return clusters
    
    async def _extract_pattern_from_cluster(self, cluster_id: str, projects: List[Dict[str, Any]]) -> SDLCPattern:
        """Extract optimal pattern from a cluster of similar projects."""
        if not projects:
            return self._create_default_pattern(cluster_id)
        
        # Aggregate metrics
        success_rates = [p.get("success_rate", 0.5) for p in projects]
        performance_metrics = {}
        
        # Extract common configuration patterns
        configurations = [p.get("configuration", {}) for p in projects]
        optimal_config = self._merge_configurations(configurations)
        
        # Calculate pattern statistics
        avg_success_rate = np.mean(success_rates)
        complexity_scores = [p.get("complexity_score", 0.5) for p in projects]
        avg_complexity = np.mean(complexity_scores)
        
        # Performance metrics aggregation
        for project in projects:
            metrics = project.get("performance_metrics", {})
            for metric, value in metrics.items():
                if metric not in performance_metrics:
                    performance_metrics[metric] = []
                performance_metrics[metric].append(value)
        
        # Average performance metrics
        avg_performance = {
            metric: np.mean(values) 
            for metric, values in performance_metrics.items()
        }
        
        return SDLCPattern(
            pattern_id=cluster_id,
            project_type=cluster_id.split("_")[0],
            complexity_score=avg_complexity,
            optimal_configuration=optimal_config,
            success_probability=avg_success_rate,
            performance_metrics=avg_performance,
            learned_from_executions=len(projects),
            confidence_level=min(0.95, len(projects) / 20.0)  # Higher confidence with more data
        )
    
    def _create_default_pattern(self, cluster_id: str) -> SDLCPattern:
        """Create a default pattern when no historical data is available."""
        return SDLCPattern(
            pattern_id=cluster_id,
            project_type="unknown",
            complexity_score=0.5,
            optimal_configuration={
                "timeout_multiplier": 1.0,
                "retry_count": 3,
                "parallel_execution": True,
                "optimization_frequency": 10
            },
            success_probability=0.7,
            performance_metrics={"execution_time": 300.0, "memory_usage": 512.0},
            learned_from_executions=0,
            confidence_level=0.3
        )
    
    def _merge_configurations(self, configurations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple configurations to find optimal settings."""
        if not configurations:
            return {}
        
        merged = {}
        
        # Collect all keys
        all_keys = set()
        for config in configurations:
            all_keys.update(config.keys())
        
        # For each key, find the most common or optimal value
        for key in all_keys:
            values = [config.get(key) for config in configurations if key in config]
            
            if not values:
                continue
            
            # Handle different value types
            if all(isinstance(v, (int, float)) for v in values):
                # For numeric values, use median
                merged[key] = float(np.median(values))
            elif all(isinstance(v, bool) for v in values):
                # For boolean values, use majority vote
                merged[key] = sum(values) > len(values) / 2
            else:
                # For other types, use most common
                from collections import Counter
                counter = Counter(values)
                merged[key] = counter.most_common(1)[0][0]
        
        return merged
    
    async def _evolve_population(self, project_context: Dict[str, Any]) -> NeuralGenome:
        """Evolve the population to find optimal neural configuration."""
        self.logger.info(f"Evolving population for {self.generations} generations")
        
        for generation in range(self.generations):
            # Evaluate fitness for all genomes
            await self._evaluate_population_fitness(project_context)
            
            # Sort by fitness
            self.current_population.sort(key=lambda g: g.fitness_score, reverse=True)
            
            # Track best genome
            if not self.best_genome or self.current_population[0].fitness_score > self.best_genome.fitness_score:
                self.best_genome = self.current_population[0]
            
            # Log generation statistics
            avg_fitness = np.mean([g.fitness_score for g in self.current_population])
            self.logger.info(f"Generation {generation}: Best={self.best_genome.fitness_score:.4f}, Avg={avg_fitness:.4f}")
            
            # Create next generation
            if generation < self.generations - 1:
                await self._create_next_generation(generation + 1)
            
            # Optimize memory every 10 generations
            if generation % 10 == 0:
                optimize_memory()
        
        self.logger.info(f"Evolution complete. Best fitness: {self.best_genome.fitness_score:.4f}")
        return self.best_genome
    
    async def _evaluate_population_fitness(self, project_context: Dict[str, Any]) -> None:
        """Evaluate fitness of all genomes in the population."""
        tasks = []
        for genome in self.current_population:
            task = asyncio.create_task(self._evaluate_genome_fitness(genome, project_context))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def _evaluate_genome_fitness(self, genome: NeuralGenome, project_context: Dict[str, Any]) -> None:
        """Evaluate the fitness of a single genome."""
        # Simulate neural network performance based on genome characteristics
        
        # Base fitness from architecture efficiency
        architecture_scores = {
            NeuralArchitecture.TRANSFORMER: 0.9,
            NeuralArchitecture.LSTM: 0.8,
            NeuralArchitecture.GRU: 0.75,
            NeuralArchitecture.ATTENTION: 0.85,
            NeuralArchitecture.HYBRID: 0.95
        }
        
        base_score = architecture_scores.get(genome.architecture, 0.7)
        
        # Adjust for layer configuration
        layer_penalty = 0.0
        if len(genome.layers) > 8:  # Too many layers
            layer_penalty += 0.1
        if any(neurons > 256 for neurons in genome.layers):  # Too many neurons
            layer_penalty += 0.05
        
        # Learning rate optimization
        lr_penalty = 0.0
        if genome.learning_rate < 0.001 or genome.learning_rate > 0.01:
            lr_penalty += 0.05
        
        # Dropout rate optimization
        dropout_penalty = 0.0
        if genome.dropout_rate > 0.3:  # Too much dropout
            dropout_penalty += 0.05
        
        # Project context matching
        context_bonus = await self._calculate_context_matching_bonus(genome, project_context)
        
        # Final fitness calculation
        fitness = base_score - layer_penalty - lr_penalty - dropout_penalty + context_bonus
        fitness = max(0.0, min(1.0, fitness))  # Clamp to [0, 1]
        
        # Add some noise for diversity
        noise = np.random.normal(0, 0.02)
        fitness += noise
        
        genome.fitness_score = max(0.0, fitness)
    
    async def _calculate_context_matching_bonus(self, genome: NeuralGenome, 
                                              project_context: Dict[str, Any]) -> float:
        """Calculate bonus based on how well genome matches project context."""
        bonus = 0.0
        
        # Project complexity matching
        complexity = project_context.get("complexity_score", 0.5)
        if complexity > 0.7:  # High complexity projects
            if genome.architecture in [NeuralArchitecture.TRANSFORMER, NeuralArchitecture.HYBRID]:
                bonus += 0.1
            if len(genome.layers) >= 5:  # Deeper networks for complex projects
                bonus += 0.05
        elif complexity < 0.3:  # Simple projects
            if len(genome.layers) <= 3:  # Simpler networks for simple projects
                bonus += 0.05
        
        # Project type matching
        project_type = project_context.get("project_type", "unknown")
        if project_type == "ml_training" and genome.architecture == NeuralArchitecture.TRANSFORMER:
            bonus += 0.1
        elif project_type == "web_service" and genome.architecture in [NeuralArchitecture.LSTM, NeuralArchitecture.GRU]:
            bonus += 0.05
        
        return bonus
    
    async def _create_next_generation(self, generation: int) -> None:
        """Create the next generation using evolutionary operators."""
        elite_count = int(self.population_size * self.elite_ratio)
        
        # Keep elite individuals
        new_population = self.current_population[:elite_count].copy()
        
        # Set generation for elite
        for genome in new_population:
            genome.generation = generation
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents using tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = await self._crossover(parent1, parent2, generation)
                new_population.extend([child1, child2])
            else:
                # Direct copying with mutation
                child = await self._mutate(parent1, generation)
                new_population.append(child)
            
            # Ensure we don't exceed population size
            if len(new_population) >= self.population_size:
                break
        
        # Trim to exact population size
        self.current_population = new_population[:self.population_size]
    
    def _tournament_selection(self, tournament_size: int = 5) -> NeuralGenome:
        """Select parent using tournament selection."""
        tournament = np.random.choice(self.current_population, size=tournament_size, replace=False)
        return max(tournament, key=lambda g: g.fitness_score)
    
    async def _crossover(self, parent1: NeuralGenome, parent2: NeuralGenome, generation: int) -> Tuple[NeuralGenome, NeuralGenome]:
        """Create two children through crossover."""
        child1_id = f"gen{generation}_cross_{len(self.current_population)}"
        child2_id = f"gen{generation}_cross_{len(self.current_population) + 1}"
        
        # Architecture crossover (random selection)
        child1_arch = parent1.architecture if np.random.random() < 0.5 else parent2.architecture
        child2_arch = parent2.architecture if np.random.random() < 0.5 else parent1.architecture
        
        # Layer crossover (uniform)
        max_len = max(len(parent1.layers), len(parent2.layers))
        child1_layers = []
        child2_layers = []
        
        for i in range(max_len):
            p1_layer = parent1.layers[i] if i < len(parent1.layers) else parent1.layers[-1]
            p2_layer = parent2.layers[i] if i < len(parent2.layers) else parent2.layers[-1]
            
            if np.random.random() < 0.5:
                child1_layers.append(p1_layer)
                child2_layers.append(p2_layer)
            else:
                child1_layers.append(p2_layer)
                child2_layers.append(p1_layer)
        
        # Hyperparameter crossover (uniform)
        child1_lr = parent1.learning_rate if np.random.random() < 0.5 else parent2.learning_rate
        child2_lr = parent2.learning_rate if np.random.random() < 0.5 else parent1.learning_rate
        
        child1_dropout = parent1.dropout_rate if np.random.random() < 0.5 else parent2.dropout_rate
        child2_dropout = parent2.dropout_rate if np.random.random() < 0.5 else parent1.dropout_rate
        
        # Create children
        child1 = NeuralGenome(
            id=child1_id,
            architecture=child1_arch,
            layers=child1_layers,
            activation_functions=parent1.activation_functions.copy(),
            learning_rate=child1_lr,
            dropout_rate=child1_dropout,
            batch_size=parent1.batch_size,
            epochs=parent1.epochs,
            generation=generation,
            parent_ids=[parent1.id, parent2.id]
        )
        
        child2 = NeuralGenome(
            id=child2_id,
            architecture=child2_arch,
            layers=child2_layers,
            activation_functions=parent2.activation_functions.copy(),
            learning_rate=child2_lr,
            dropout_rate=child2_dropout,
            batch_size=parent2.batch_size,
            epochs=parent2.epochs,
            generation=generation,
            parent_ids=[parent1.id, parent2.id]
        )
        
        # Mutate children
        child1 = await self._mutate(child1, generation, force=False)
        child2 = await self._mutate(child2, generation, force=False)
        
        return child1, child2
    
    async def _mutate(self, genome: NeuralGenome, generation: int, force: bool = True) -> NeuralGenome:
        """Mutate a genome."""
        if not force and np.random.random() > genome.mutation_rate:
            return genome
        
        # Create mutated copy
        mutated = NeuralGenome(
            id=f"gen{generation}_mut_{hash(genome.id) % 10000}",
            architecture=genome.architecture,
            layers=genome.layers.copy(),
            activation_functions=genome.activation_functions.copy(),
            learning_rate=genome.learning_rate,
            dropout_rate=genome.dropout_rate,
            batch_size=genome.batch_size,
            epochs=genome.epochs,
            generation=generation,
            parent_ids=[genome.id]
        )
        
        # Mutate architecture (rare)
        if np.random.random() < 0.1:
            mutated.architecture = np.random.choice(list(NeuralArchitecture))
        
        # Mutate layers
        if np.random.random() < 0.3:
            # Add layer
            if len(mutated.layers) < self.max_layers:
                new_layer_size = np.random.randint(32, self.max_neurons_per_layer + 1)
                insert_pos = np.random.randint(0, len(mutated.layers) + 1)
                mutated.layers.insert(insert_pos, new_layer_size)
                mutated.activation_functions.insert(insert_pos, np.random.choice(self.activation_functions))
        
        if np.random.random() < 0.2:
            # Remove layer
            if len(mutated.layers) > 2:
                remove_pos = np.random.randint(0, len(mutated.layers))
                mutated.layers.pop(remove_pos)
                mutated.activation_functions.pop(remove_pos)
        
        # Mutate layer sizes
        for i in range(len(mutated.layers)):
            if np.random.random() < 0.4:
                mutation_factor = np.random.normal(1.0, 0.2)
                new_size = int(mutated.layers[i] * mutation_factor)
                mutated.layers[i] = max(16, min(self.max_neurons_per_layer, new_size))
        
        # Mutate hyperparameters
        if np.random.random() < 0.5:
            mutated.learning_rate *= np.random.uniform(0.5, 2.0)
            mutated.learning_rate = max(0.0001, min(0.1, mutated.learning_rate))
        
        if np.random.random() < 0.3:
            mutated.dropout_rate += np.random.normal(0, 0.1)
            mutated.dropout_rate = max(0.0, min(0.8, mutated.dropout_rate))
        
        return mutated
    
    async def _generate_sdlc_config(self, best_genome: NeuralGenome, 
                                   project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized SDLC configuration from the best genome."""
        config = {
            "neural_optimized": True,
            "optimization_source": "neural_evolution",
            "genome_id": best_genome.id,
            "fitness_score": best_genome.fitness_score,
            "generation": best_genome.generation,
            
            # Neural architecture translation to SDLC parameters
            "parallel_execution": len(best_genome.layers) > 4,  # Complex networks suggest parallel execution
            "timeout_multiplier": 1.0 + (best_genome.learning_rate * 10),  # Learning rate affects patience
            "retry_count": min(5, max(1, int(best_genome.dropout_rate * 10))),  # Dropout suggests retries
            "batch_processing": best_genome.batch_size > 64,
            
            # Optimization frequency based on network complexity
            "optimization_frequency": max(5, 20 - len(best_genome.layers)),
            
            # Quality thresholds based on architecture sophistication
            "quality_threshold": 0.7 + (best_genome.fitness_score * 0.2),
            
            # Performance settings
            "max_parallel_actions": min(8, len(best_genome.layers)),
            "auto_rollback": True,  # Always enable for safety
            
            # Neural network specific metadata
            "neural_metadata": {
                "architecture": best_genome.architecture.value,
                "layers": best_genome.layers,
                "activations": best_genome.activation_functions,
                "hyperparameters": {
                    "learning_rate": best_genome.learning_rate,
                    "dropout_rate": best_genome.dropout_rate,
                    "batch_size": best_genome.batch_size,
                    "epochs": best_genome.epochs
                }
            }
        }
        
        # Apply learned patterns if available
        pattern = await self._find_matching_pattern(project_context)
        if pattern and pattern.confidence_level > 0.5:
            # Merge pattern configuration
            for key, value in pattern.optimal_configuration.items():
                if key not in config:
                    config[key] = value
            
            config["pattern_applied"] = pattern.pattern_id
            config["pattern_confidence"] = pattern.confidence_level
        
        return config
    
    async def _find_matching_pattern(self, project_context: Dict[str, Any]) -> Optional[SDLCPattern]:
        """Find the best matching learned pattern for the project context."""
        if not self.learned_patterns:
            return None
        
        project_type = project_context.get("project_type", "unknown")
        complexity = project_context.get("complexity_score", 0.5)
        
        best_match = None
        best_score = 0.0
        
        for pattern in self.learned_patterns.values():
            score = 0.0
            
            # Type matching
            if pattern.project_type == project_type:
                score += 0.5
            
            # Complexity matching
            complexity_diff = abs(pattern.complexity_score - complexity)
            score += max(0, 0.5 - complexity_diff)
            
            # Confidence weighting
            score *= pattern.confidence_level
            
            if score > best_score:
                best_score = score
                best_match = pattern
        
        return best_match if best_score > 0.3 else None
    
    def _generate_cache_key(self, project_context: Dict[str, Any]) -> str:
        """Generate cache key for project context."""
        context_str = json.dumps(project_context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get statistics about the neural evolution process."""
        if not self.current_population:
            return {"status": "not_initialized"}
        
        fitness_scores = [g.fitness_score for g in self.current_population]
        
        return {
            "population_size": len(self.current_population),
            "best_fitness": max(fitness_scores),
            "average_fitness": np.mean(fitness_scores),
            "fitness_std": np.std(fitness_scores),
            "generations_completed": self.best_genome.generation if self.best_genome else 0,
            "learned_patterns": len(self.learned_patterns),
            "best_architecture": self.best_genome.architecture.value if self.best_genome else None,
            "diversity_score": len(set(g.architecture for g in self.current_population)) / len(NeuralArchitecture)
        }
    
    async def save_evolution_state(self, filepath: Path) -> None:
        """Save the current evolution state to disk."""
        state = {
            "population": [self._genome_to_dict(g) for g in self.current_population],
            "best_genome": self._genome_to_dict(self.best_genome) if self.best_genome else None,
            "learned_patterns": {k: self._pattern_to_dict(p) for k, p in self.learned_patterns.items()},
            "evolution_history": self.evolution_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"Evolution state saved to {filepath}")
    
    async def load_evolution_state(self, filepath: Path) -> None:
        """Load evolution state from disk."""
        if not filepath.exists():
            self.logger.warning(f"Evolution state file not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Restore population
        self.current_population = [self._dict_to_genome(d) for d in state["population"]]
        
        # Restore best genome
        if state["best_genome"]:
            self.best_genome = self._dict_to_genome(state["best_genome"])
        
        # Restore learned patterns
        self.learned_patterns = {k: self._dict_to_pattern(d) for k, d in state["learned_patterns"].items()}
        
        # Restore history
        self.evolution_history = state["evolution_history"]
        
        self.logger.info(f"Evolution state loaded from {filepath}")
    
    def _genome_to_dict(self, genome: NeuralGenome) -> Dict[str, Any]:
        """Convert genome to dictionary for serialization."""
        return {
            "id": genome.id,
            "architecture": genome.architecture.value,
            "layers": genome.layers,
            "activation_functions": genome.activation_functions,
            "learning_rate": genome.learning_rate,
            "dropout_rate": genome.dropout_rate,
            "batch_size": genome.batch_size,
            "epochs": genome.epochs,
            "fitness_score": genome.fitness_score,
            "generation": genome.generation,
            "parent_ids": genome.parent_ids,
            "mutation_rate": genome.mutation_rate
        }
    
    def _dict_to_genome(self, data: Dict[str, Any]) -> NeuralGenome:
        """Convert dictionary to genome."""
        return NeuralGenome(
            id=data["id"],
            architecture=NeuralArchitecture(data["architecture"]),
            layers=data["layers"],
            activation_functions=data["activation_functions"],
            learning_rate=data["learning_rate"],
            dropout_rate=data["dropout_rate"],
            batch_size=data["batch_size"],
            epochs=data["epochs"],
            fitness_score=data["fitness_score"],
            generation=data["generation"],
            parent_ids=data["parent_ids"],
            mutation_rate=data["mutation_rate"]
        )
    
    def _pattern_to_dict(self, pattern: SDLCPattern) -> Dict[str, Any]:
        """Convert pattern to dictionary for serialization."""
        return {
            "pattern_id": pattern.pattern_id,
            "project_type": pattern.project_type,
            "complexity_score": pattern.complexity_score,
            "optimal_configuration": pattern.optimal_configuration,
            "success_probability": pattern.success_probability,
            "performance_metrics": pattern.performance_metrics,
            "learned_from_executions": pattern.learned_from_executions,
            "confidence_level": pattern.confidence_level
        }
    
    def _dict_to_pattern(self, data: Dict[str, Any]) -> SDLCPattern:
        """Convert dictionary to pattern."""
        return SDLCPattern(
            pattern_id=data["pattern_id"],
            project_type=data["project_type"],
            complexity_score=data["complexity_score"],
            optimal_configuration=data["optimal_configuration"],
            success_probability=data["success_probability"],
            performance_metrics=data["performance_metrics"],
            learned_from_executions=data["learned_from_executions"],
            confidence_level=data["confidence_level"]
        )
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        optimize_memory()