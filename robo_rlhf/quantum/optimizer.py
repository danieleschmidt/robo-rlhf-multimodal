"""
Quantum-inspired optimization algorithms for autonomous SDLC execution.

Implements quantum annealing, genetic algorithms, and multi-objective optimization
for resource allocation, performance tuning, and system adaptation.
"""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

from robo_rlhf.core import get_logger, get_config
from robo_rlhf.core.exceptions import RoboRLHFError


class OptimizationObjective(Enum):
    """Optimization objectives."""
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_RESOURCES = "minimize_resources"
    MAXIMIZE_QUALITY = "maximize_quality"
    MAXIMIZE_RELIABILITY = "maximize_reliability"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_PERFORMANCE = "maximize_performance"


@dataclass
class OptimizationConstraint:
    """Optimization constraint definition."""
    name: str
    constraint_type: str  # "equality", "inequality", "bound"
    target_value: float
    tolerance: float = 0.0
    weight: float = 1.0


@dataclass
class OptimizationSolution:
    """Solution representation for optimization problems."""
    parameters: Dict[str, float]
    objectives: Dict[str, float]
    constraints_satisfied: bool
    fitness_score: float
    generation: int = 0
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    quantum_energy: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class OptimizationProblem:
    """Multi-objective optimization problem definition."""
    name: str
    objectives: List[OptimizationObjective]
    constraints: List[OptimizationConstraint] = field(default_factory=list)
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    objective_weights: Dict[OptimizationObjective, float] = field(default_factory=dict)
    evaluation_function: Optional[Callable] = None
    target_solutions: int = 100
    max_generations: int = 1000
    convergence_threshold: float = 1e-6


class QuantumOptimizer:
    """Quantum-inspired optimization engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = config or get_config().to_dict()
        
        # Quantum parameters
        self.quantum_temperature = self.config.get("quantum", {}).get("temperature", 1.0)
        self.annealing_schedule = self.config.get("quantum", {}).get("annealing_schedule", "exponential")
        self.coherence_time = self.config.get("quantum", {}).get("coherence_time", 100.0)
        
        # Optimization parameters
        self.population_size = self.config.get("optimization", {}).get("population_size", 50)
        self.elite_size = self.config.get("optimization", {}).get("elite_size", 10)
        self.mutation_rate = self.config.get("optimization", {}).get("mutation_rate", 0.1)
        self.crossover_rate = self.config.get("optimization", {}).get("crossover_rate", 0.8)
        
        # State tracking
        self.current_solutions: List[OptimizationSolution] = []
        self.best_solutions: List[OptimizationSolution] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        self.logger.info("QuantumOptimizer initialized with quantum annealing capabilities")
    
    async def optimize(self, problem: OptimizationProblem) -> List[OptimizationSolution]:
        """Optimize using quantum-inspired algorithms."""
        self.logger.info(f"Starting quantum optimization: {problem.name}")
        
        # Initialize quantum population
        population = self._initialize_quantum_population(problem)
        
        # Quantum optimization loop
        best_fitness_history = []
        convergence_count = 0
        
        for generation in range(problem.max_generations):
            # Update quantum temperature (simulated annealing)
            current_temperature = self._update_quantum_temperature(generation, problem.max_generations)
            
            # Evaluate population
            await self._evaluate_population(population, problem)
            
            # Track best fitness
            best_fitness = max(sol.fitness_score for sol in population)
            best_fitness_history.append(best_fitness)
            
            # Check convergence
            if len(best_fitness_history) > 10:
                recent_improvement = best_fitness_history[-1] - best_fitness_history[-10]
                if abs(recent_improvement) < problem.convergence_threshold:
                    convergence_count += 1
                    if convergence_count >= 5:
                        self.logger.info(f"Converged at generation {generation}")
                        break
                else:
                    convergence_count = 0
            
            # Quantum evolution step
            population = await self._quantum_evolution_step(
                population, problem, current_temperature, generation
            )
            
            # Log progress
            if generation % 50 == 0:
                avg_fitness = np.mean([sol.fitness_score for sol in population])
                self.logger.debug(f"Generation {generation}: best={best_fitness:.4f}, avg={avg_fitness:.4f}, temp={current_temperature:.4f}")
        
        # Select final solutions
        final_solutions = self._select_pareto_optimal_solutions(population, problem)
        
        self.current_solutions = population
        self.best_solutions = final_solutions
        
        self.logger.info(f"Quantum optimization completed: {len(final_solutions)} Pareto-optimal solutions found")
        return final_solutions
    
    def _initialize_quantum_population(self, problem: OptimizationProblem) -> List[OptimizationSolution]:
        """Initialize quantum population with superposition states."""
        population = []
        
        for i in range(self.population_size):
            # Generate random parameters within bounds
            parameters = {}
            for param_name, (min_val, max_val) in problem.parameter_bounds.items():
                # Quantum superposition initialization
                parameters[param_name] = np.random.uniform(min_val, max_val)
            
            # Create quantum solution
            solution = OptimizationSolution(
                parameters=parameters,
                objectives={},
                constraints_satisfied=False,
                fitness_score=0.0,
                generation=0,
                quantum_energy=np.random.exponential(self.quantum_temperature)
            )
            
            population.append(solution)
        
        self.logger.debug(f"Initialized quantum population with {len(population)} solutions")
        return population
    
    async def _evaluate_population(self, population: List[OptimizationSolution], problem: OptimizationProblem) -> None:
        """Evaluate population fitness using quantum measurement."""
        tasks = []
        
        for solution in population:
            task = asyncio.create_task(self._evaluate_solution(solution, problem))
            tasks.append(task)
        
        # Wait for all evaluations
        await asyncio.gather(*tasks)
    
    async def _evaluate_solution(self, solution: OptimizationSolution, problem: OptimizationProblem) -> None:
        """Evaluate individual solution using quantum measurement."""
        # Use custom evaluation function if provided
        if problem.evaluation_function:
            objectives = problem.evaluation_function(solution.parameters)
        else:
            # Default SDLC optimization objectives
            objectives = self._evaluate_sdlc_objectives(solution.parameters, problem)
        
        solution.objectives = objectives
        
        # Check constraints
        solution.constraints_satisfied = self._check_constraints(solution, problem)
        
        # Calculate fitness using quantum measurement
        solution.fitness_score = self._calculate_quantum_fitness(solution, problem)
    
    def _evaluate_sdlc_objectives(self, parameters: Dict[str, float], problem: OptimizationProblem) -> Dict[str, float]:
        """Evaluate SDLC-specific optimization objectives."""
        objectives = {}
        
        # Time-based objectives
        if OptimizationObjective.MINIMIZE_TIME in problem.objectives:
            # Simulate time estimation based on resource allocation
            cpu_allocation = parameters.get("cpu_allocation", 0.5)
            parallel_workers = parameters.get("parallel_workers", 1)
            estimated_time = 100 / (cpu_allocation * parallel_workers)
            objectives["time"] = estimated_time
        
        # Resource-based objectives
        if OptimizationObjective.MINIMIZE_RESOURCES in problem.objectives:
            resource_usage = sum(parameters.get(f"{resource}_allocation", 0.5) 
                               for resource in ["cpu", "memory", "storage", "network"])
            objectives["resources"] = resource_usage
        
        # Quality-based objectives
        if OptimizationObjective.MAXIMIZE_QUALITY in problem.objectives:
            test_coverage = parameters.get("test_coverage", 0.8)
            code_quality = parameters.get("code_quality", 0.7)
            quality_score = (test_coverage + code_quality) / 2
            objectives["quality"] = quality_score
        
        # Performance-based objectives
        if OptimizationObjective.MAXIMIZE_PERFORMANCE in problem.objectives:
            optimization_level = parameters.get("optimization_level", 0.5)
            caching_efficiency = parameters.get("caching_efficiency", 0.6)
            performance_score = (optimization_level + caching_efficiency) / 2
            objectives["performance"] = performance_score
        
        # Reliability-based objectives
        if OptimizationObjective.MAXIMIZE_RELIABILITY in problem.objectives:
            error_handling = parameters.get("error_handling", 0.7)
            monitoring_coverage = parameters.get("monitoring_coverage", 0.8)
            reliability_score = (error_handling + monitoring_coverage) / 2
            objectives["reliability"] = reliability_score
        
        return objectives
    
    def _check_constraints(self, solution: OptimizationSolution, problem: OptimizationProblem) -> bool:
        """Check if solution satisfies all constraints."""
        for constraint in problem.constraints:
            if constraint.constraint_type == "bound":
                param_value = solution.parameters.get(constraint.name, 0.0)
                if not (constraint.target_value - constraint.tolerance <= param_value <= constraint.target_value + constraint.tolerance):
                    return False
            
            elif constraint.constraint_type == "inequality":
                objective_value = solution.objectives.get(constraint.name, 0.0)
                if objective_value > constraint.target_value + constraint.tolerance:
                    return False
            
            elif constraint.constraint_type == "equality":
                objective_value = solution.objectives.get(constraint.name, 0.0)
                if abs(objective_value - constraint.target_value) > constraint.tolerance:
                    return False
        
        return True
    
    def _calculate_quantum_fitness(self, solution: OptimizationSolution, problem: OptimizationProblem) -> float:
        """Calculate fitness using quantum measurement principles."""
        if not solution.constraints_satisfied:
            return -float('inf')  # Heavily penalize constraint violations
        
        # Multi-objective fitness calculation
        weighted_objectives = 0.0
        total_weight = 0.0
        
        for objective_type in problem.objectives:
            weight = problem.objective_weights.get(objective_type, 1.0)
            
            if objective_type == OptimizationObjective.MINIMIZE_TIME:
                value = 1.0 / (1.0 + solution.objectives.get("time", 100))
            elif objective_type == OptimizationObjective.MINIMIZE_RESOURCES:
                value = 1.0 / (1.0 + solution.objectives.get("resources", 4))
            elif objective_type == OptimizationObjective.MAXIMIZE_QUALITY:
                value = solution.objectives.get("quality", 0.5)
            elif objective_type == OptimizationObjective.MAXIMIZE_PERFORMANCE:
                value = solution.objectives.get("performance", 0.5)
            elif objective_type == OptimizationObjective.MAXIMIZE_RELIABILITY:
                value = solution.objectives.get("reliability", 0.5)
            else:
                value = 0.5  # Default value
            
            # Apply quantum uncertainty
            quantum_noise = np.random.normal(0, 0.01) * np.exp(-time.time() / self.coherence_time)
            value += quantum_noise
            
            weighted_objectives += weight * value
            total_weight += weight
        
        # Normalize fitness
        if total_weight > 0:
            fitness = weighted_objectives / total_weight
        else:
            fitness = 0.0
        
        # Add quantum energy contribution
        quantum_contribution = solution.quantum_energy * 0.01
        fitness += quantum_contribution
        
        return max(0.0, fitness)
    
    async def _quantum_evolution_step(
        self,
        population: List[OptimizationSolution],
        problem: OptimizationProblem,
        temperature: float,
        generation: int
    ) -> List[OptimizationSolution]:
        """Perform quantum evolution step."""
        new_population = []
        
        # Sort by fitness (quantum measurement)
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Elitism - keep best solutions
        elite_count = min(self.elite_size, len(population))
        for i in range(elite_count):
            elite_copy = self._copy_solution(population[i])
            elite_copy.generation = generation
            new_population.append(elite_copy)
        
        # Generate offspring using quantum operations
        while len(new_population) < self.population_size:
            # Quantum selection
            parent1 = self._quantum_selection(population, temperature)
            parent2 = self._quantum_selection(population, temperature)
            
            # Quantum crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._quantum_crossover(parent1, parent2, problem)
            else:
                child1, child2 = self._copy_solution(parent1), self._copy_solution(parent2)
            
            # Quantum mutation
            if np.random.random() < self.mutation_rate:
                child1 = self._quantum_mutation(child1, problem, temperature)
            if np.random.random() < self.mutation_rate:
                child2 = self._quantum_mutation(child2, problem, temperature)
            
            # Update generation
            child1.generation = generation
            child2.generation = generation
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        return new_population[:self.population_size]
    
    def _quantum_selection(self, population: List[OptimizationSolution], temperature: float) -> OptimizationSolution:
        """Quantum tournament selection."""
        tournament_size = min(5, len(population))
        tournament = np.random.choice(population, tournament_size, replace=False)
        
        # Quantum selection probabilities based on Boltzmann distribution
        energies = [-sol.fitness_score / temperature for sol in tournament]
        probabilities = np.exp(energies)
        probabilities /= np.sum(probabilities)
        
        # Quantum measurement
        selected_idx = np.random.choice(len(tournament), p=probabilities)
        return tournament[selected_idx]
    
    def _quantum_crossover(
        self,
        parent1: OptimizationSolution,
        parent2: OptimizationSolution,
        problem: OptimizationProblem
    ) -> Tuple[OptimizationSolution, OptimizationSolution]:
        """Quantum crossover operation."""
        child1_params = {}
        child2_params = {}
        
        for param_name in parent1.parameters:
            # Quantum superposition crossover
            alpha = np.random.beta(2, 2)  # Quantum interference pattern
            
            val1 = parent1.parameters[param_name]
            val2 = parent2.parameters[param_name]
            
            # Quantum entangled crossover
            child1_params[param_name] = alpha * val1 + (1 - alpha) * val2
            child2_params[param_name] = (1 - alpha) * val1 + alpha * val2
            
            # Ensure bounds
            if param_name in problem.parameter_bounds:
                min_val, max_val = problem.parameter_bounds[param_name]
                child1_params[param_name] = np.clip(child1_params[param_name], min_val, max_val)
                child2_params[param_name] = np.clip(child2_params[param_name], min_val, max_val)
        
        child1 = OptimizationSolution(
            parameters=child1_params,
            objectives={},
            constraints_satisfied=False,
            fitness_score=0.0,
            quantum_energy=(parent1.quantum_energy + parent2.quantum_energy) / 2
        )
        
        child2 = OptimizationSolution(
            parameters=child2_params,
            objectives={},
            constraints_satisfied=False,
            fitness_score=0.0,
            quantum_energy=(parent1.quantum_energy + parent2.quantum_energy) / 2
        )
        
        return child1, child2
    
    def _quantum_mutation(
        self,
        solution: OptimizationSolution,
        problem: OptimizationProblem,
        temperature: float
    ) -> OptimizationSolution:
        """Quantum mutation operation."""
        mutated_params = solution.parameters.copy()
        
        for param_name, param_value in mutated_params.items():
            if np.random.random() < 0.3:  # Parameter mutation probability
                # Quantum tunneling mutation
                if param_name in problem.parameter_bounds:
                    min_val, max_val = problem.parameter_bounds[param_name]
                    range_size = max_val - min_val
                    
                    # Quantum tunneling with temperature dependence
                    mutation_strength = temperature * range_size * 0.1
                    quantum_shift = np.random.normal(0, mutation_strength)
                    
                    new_value = param_value + quantum_shift
                    new_value = np.clip(new_value, min_val, max_val)
                    
                    mutated_params[param_name] = new_value
        
        mutated_solution = OptimizationSolution(
            parameters=mutated_params,
            objectives={},
            constraints_satisfied=False,
            fitness_score=0.0,
            quantum_energy=solution.quantum_energy * np.random.uniform(0.9, 1.1)
        )
        
        return mutated_solution
    
    def _update_quantum_temperature(self, generation: int, max_generations: int) -> float:
        """Update quantum temperature for simulated annealing."""
        progress = generation / max_generations
        
        if self.annealing_schedule == "exponential":
            return self.quantum_temperature * np.exp(-5 * progress)
        elif self.annealing_schedule == "linear":
            return self.quantum_temperature * (1 - progress)
        elif self.annealing_schedule == "logarithmic":
            return self.quantum_temperature / (1 + np.log(1 + generation))
        else:
            return self.quantum_temperature * (0.95 ** generation)
    
    def _select_pareto_optimal_solutions(
        self,
        population: List[OptimizationSolution],
        problem: OptimizationProblem
    ) -> List[OptimizationSolution]:
        """Select Pareto-optimal solutions from population."""
        pareto_solutions = []
        
        for candidate in population:
            if not candidate.constraints_satisfied:
                continue
                
            is_dominated = False
            
            for other in population:
                if not other.constraints_satisfied or candidate == other:
                    continue
                
                # Check if other dominates candidate
                dominates = True
                for obj_type in problem.objectives:
                    obj_name = obj_type.value.split("_")[1]  # Extract objective name
                    
                    candidate_val = candidate.objectives.get(obj_name, 0.0)
                    other_val = other.objectives.get(obj_name, 0.0)
                    
                    # For minimization objectives, lower is better
                    if obj_type.value.startswith("minimize"):
                        if candidate_val < other_val:
                            dominates = False
                            break
                    # For maximization objectives, higher is better
                    else:
                        if candidate_val > other_val:
                            dominates = False
                            break
                
                if dominates:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_solutions.append(candidate)
        
        # Sort by fitness score
        pareto_solutions.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Limit number of solutions
        max_solutions = min(problem.target_solutions, len(pareto_solutions))
        return pareto_solutions[:max_solutions]
    
    def _copy_solution(self, solution: OptimizationSolution) -> OptimizationSolution:
        """Create a copy of solution."""
        return OptimizationSolution(
            parameters=solution.parameters.copy(),
            objectives=solution.objectives.copy(),
            constraints_satisfied=solution.constraints_satisfied,
            fitness_score=solution.fitness_score,
            generation=solution.generation,
            mutation_rate=solution.mutation_rate,
            crossover_rate=solution.crossover_rate,
            quantum_energy=solution.quantum_energy
        )


class MultiObjectiveOptimizer:
    """Multi-objective optimization for complex SDLC scenarios."""
    
    def __init__(self, quantum_optimizer: QuantumOptimizer):
        self.logger = get_logger(__name__)
        self.quantum_optimizer = quantum_optimizer
        self.optimization_results: List[Dict[str, Any]] = []
    
    async def optimize_sdlc_pipeline(
        self,
        pipeline_config: Dict[str, Any],
        objectives: List[OptimizationObjective],
        constraints: Optional[List[OptimizationConstraint]] = None
    ) -> List[OptimizationSolution]:
        """Optimize entire SDLC pipeline configuration."""
        self.logger.info("Starting multi-objective SDLC pipeline optimization")
        
        # Define optimization problem
        problem = self._create_sdlc_optimization_problem(
            pipeline_config, objectives, constraints or []
        )
        
        # Run quantum optimization
        start_time = time.time()
        solutions = await self.quantum_optimizer.optimize(problem)
        optimization_time = time.time() - start_time
        
        # Analyze results
        analysis = self._analyze_optimization_results(solutions, problem)
        
        # Store results
        result_record = {
            "timestamp": time.time(),
            "pipeline_config": pipeline_config,
            "objectives": [obj.value for obj in objectives],
            "solutions_count": len(solutions),
            "optimization_time": optimization_time,
            "analysis": analysis
        }
        
        self.optimization_results.append(result_record)
        
        self.logger.info(f"SDLC pipeline optimization completed: {len(solutions)} solutions in {optimization_time:.2f}s")
        return solutions
    
    def _create_sdlc_optimization_problem(
        self,
        config: Dict[str, Any],
        objectives: List[OptimizationObjective],
        constraints: List[OptimizationConstraint]
    ) -> OptimizationProblem:
        """Create SDLC optimization problem definition."""
        # Parameter bounds for SDLC optimization
        parameter_bounds = {
            # Resource allocation
            "cpu_allocation": (0.1, 1.0),
            "memory_allocation": (0.1, 1.0),
            "storage_allocation": (0.1, 1.0),
            "network_allocation": (0.1, 1.0),
            
            # Parallelization
            "parallel_workers": (1, 16),
            "batch_size": (1, 128),
            
            # Quality parameters
            "test_coverage": (0.7, 1.0),
            "code_quality": (0.6, 1.0),
            "documentation_coverage": (0.5, 1.0),
            
            # Performance parameters
            "optimization_level": (0.0, 1.0),
            "caching_efficiency": (0.0, 1.0),
            
            # Reliability parameters
            "error_handling": (0.6, 1.0),
            "monitoring_coverage": (0.7, 1.0),
            "backup_frequency": (0.1, 1.0)
        }
        
        # Objective weights based on importance
        objective_weights = {
            OptimizationObjective.MINIMIZE_TIME: 0.25,
            OptimizationObjective.MINIMIZE_RESOURCES: 0.20,
            OptimizationObjective.MAXIMIZE_QUALITY: 0.25,
            OptimizationObjective.MAXIMIZE_RELIABILITY: 0.20,
            OptimizationObjective.MAXIMIZE_PERFORMANCE: 0.10
        }
        
        problem = OptimizationProblem(
            name="SDLC Pipeline Optimization",
            objectives=objectives,
            constraints=constraints,
            parameter_bounds=parameter_bounds,
            objective_weights=objective_weights,
            target_solutions=50,
            max_generations=500,
            convergence_threshold=1e-4
        )
        
        return problem
    
    def _analyze_optimization_results(
        self,
        solutions: List[OptimizationSolution],
        problem: OptimizationProblem
    ) -> Dict[str, Any]:
        """Analyze optimization results and provide insights."""
        if not solutions:
            return {"error": "No solutions found"}
        
        # Fitness statistics
        fitness_scores = [sol.fitness_score for sol in solutions]
        fitness_stats = {
            "mean": np.mean(fitness_scores),
            "std": np.std(fitness_scores),
            "min": np.min(fitness_scores),
            "max": np.max(fitness_scores)
        }
        
        # Parameter analysis
        parameter_stats = {}
        for param_name in solutions[0].parameters:
            param_values = [sol.parameters[param_name] for sol in solutions]
            parameter_stats[param_name] = {
                "mean": np.mean(param_values),
                "std": np.std(param_values),
                "min": np.min(param_values),
                "max": np.max(param_values)
            }
        
        # Objective analysis
        objective_stats = {}
        for obj_type in problem.objectives:
            obj_name = obj_type.value.split("_")[1]
            if solutions[0].objectives and obj_name in solutions[0].objectives:
                obj_values = [sol.objectives[obj_name] for sol in solutions if obj_name in sol.objectives]
                if obj_values:
                    objective_stats[obj_name] = {
                        "mean": np.mean(obj_values),
                        "std": np.std(obj_values),
                        "min": np.min(obj_values),
                        "max": np.max(obj_values)
                    }
        
        # Trade-off analysis
        tradeoffs = self._analyze_objective_tradeoffs(solutions, problem)
        
        return {
            "fitness_statistics": fitness_stats,
            "parameter_statistics": parameter_stats,
            "objective_statistics": objective_stats,
            "tradeoff_analysis": tradeoffs,
            "pareto_front_size": len(solutions)
        }
    
    def _analyze_objective_tradeoffs(
        self,
        solutions: List[OptimizationSolution],
        problem: OptimizationProblem
    ) -> Dict[str, Any]:
        """Analyze trade-offs between objectives."""
        tradeoffs = {}
        
        if len(problem.objectives) >= 2:
            # Analyze correlation between objectives
            obj_names = [obj.value.split("_")[1] for obj in problem.objectives]
            
            for i, obj1 in enumerate(obj_names):
                for j, obj2 in enumerate(obj_names):
                    if i < j:
                        values1 = [sol.objectives.get(obj1, 0) for sol in solutions]
                        values2 = [sol.objectives.get(obj2, 0) for sol in solutions]
                        
                        if values1 and values2:
                            correlation = np.corrcoef(values1, values2)[0, 1]
                            tradeoffs[f"{obj1}_vs_{obj2}"] = {
                                "correlation": correlation,
                                "tradeoff_strength": abs(correlation),
                                "relationship": "negative" if correlation < -0.3 else "positive" if correlation > 0.3 else "neutral"
                            }
        
        return tradeoffs
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization runs."""
        if not self.optimization_results:
            return {"message": "No optimization runs completed"}
        
        total_runs = len(self.optimization_results)
        total_solutions = sum(r["solutions_count"] for r in self.optimization_results)
        avg_optimization_time = np.mean([r["optimization_time"] for r in self.optimization_results])
        
        # Most common objectives
        objective_counts = {}
        for result in self.optimization_results:
            for obj in result["objectives"]:
                objective_counts[obj] = objective_counts.get(obj, 0) + 1
        
        return {
            "total_optimization_runs": total_runs,
            "total_solutions_generated": total_solutions,
            "average_optimization_time": avg_optimization_time,
            "most_common_objectives": objective_counts,
            "quantum_efficiency": total_solutions / (total_runs * avg_optimization_time) if avg_optimization_time > 0 else 0
        }