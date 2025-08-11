"""
Cognitive Automation Engine for Advanced SDLC Intelligence.

Implements human-like reasoning, decision-making, and learning capabilities
for autonomous SDLC execution. Uses cognitive computing principles to make
intelligent decisions about code quality, testing strategies, and deployment.
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import re
import hashlib
from collections import defaultdict, deque
import time
from concurrent.futures import ThreadPoolExecutor

from robo_rlhf.core import get_logger, get_config
from robo_rlhf.core.exceptions import RoboRLHFError, ValidationError
from robo_rlhf.core.performance import PerformanceMonitor, optimize_memory
from robo_rlhf.core.validators import validate_dict, validate_numeric


class CognitionLevel(Enum):
    """Cognitive processing levels for decision-making."""
    REACTIVE = "reactive"           # Simple rule-based responses
    ADAPTIVE = "adaptive"           # Learning from patterns
    STRATEGIC = "strategic"         # Long-term planning
    METACOGNITIVE = "metacognitive" # Self-aware optimization


class ReasoningType(Enum):
    """Types of reasoning for different scenarios."""
    DEDUCTIVE = "deductive"         # Rule-based logical inference
    INDUCTIVE = "inductive"         # Pattern recognition and generalization
    ABDUCTIVE = "abductive"         # Best explanation reasoning
    ANALOGICAL = "analogical"       # Similarity-based reasoning
    CAUSAL = "causal"              # Cause-effect relationships


class DecisionConfidence(Enum):
    """Confidence levels for cognitive decisions."""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class CognitiveMemory:
    """Memory structure for storing cognitive experiences."""
    experience_id: str
    context: Dict[str, Any]
    decision: Dict[str, Any]
    outcome: Dict[str, Any]
    confidence: float
    reasoning_type: ReasoningType
    timestamp: float
    success: bool
    learned_patterns: List[str] = field(default_factory=list)


@dataclass
class CognitiveRule:
    """Cognitive rule for decision-making."""
    rule_id: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Dict[str, Any]]
    priority: int
    confidence: float
    activation_count: int = 0
    success_rate: float = 1.0
    last_used: float = 0.0


@dataclass
class CognitiveContext:
    """Context for cognitive decision-making."""
    project_state: Dict[str, Any]
    execution_history: List[Dict[str, Any]]
    current_goals: List[str]
    constraints: Dict[str, Any]
    risk_factors: List[str]
    available_resources: Dict[str, Any]


class CognitiveAutomationEngine:
    """Advanced cognitive automation engine for SDLC intelligence."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = config or get_config().to_dict()
        
        # Cognitive parameters
        self.cognition_level = CognitionLevel(
            self.config.get("cognitive", {}).get("level", CognitionLevel.STRATEGIC.value)
        )
        self.memory_capacity = self.config.get("cognitive", {}).get("memory_capacity", 10000)
        self.learning_rate = self.config.get("cognitive", {}).get("learning_rate", 0.1)
        self.confidence_threshold = self.config.get("cognitive", {}).get("confidence_threshold", 0.7)
        
        # Memory systems
        self.working_memory: Dict[str, Any] = {}
        self.episodic_memory: List[CognitiveMemory] = []
        self.semantic_memory: Dict[str, Any] = {}
        self.procedural_memory: List[CognitiveRule] = []
        
        # Knowledge base
        self.knowledge_base = self._initialize_knowledge_base()
        self.pattern_library = defaultdict(list)
        self.decision_tree = {}
        
        # Cognitive processes
        self.attention_focus: List[str] = []
        self.current_goals: List[str] = []
        self.reasoning_chain: List[Dict[str, Any]] = []
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.decision_accuracy = deque(maxlen=100)
        self.response_times = deque(maxlen=100)
        
        # Thread pool for parallel cognitive processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        self._initialize_cognitive_rules()
        
        self.logger.info(f"CognitiveAutomationEngine initialized with {self.cognition_level.value} level cognition")
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize the cognitive knowledge base."""
        return {
            "sdlc_best_practices": {
                "testing": {
                    "unit_test_coverage": {"min": 0.8, "optimal": 0.9},
                    "integration_tests": {"required": True, "parallel": True},
                    "performance_tests": {"frequency": "every_release", "threshold": "10%"}
                },
                "security": {
                    "vulnerability_scan": {"frequency": "every_commit", "tools": ["bandit", "safety"]},
                    "dependency_check": {"automated": True, "update_strategy": "conservative"},
                    "secrets_detection": {"pre_commit": True, "tools": ["truffhog", "gitleaks"]}
                },
                "deployment": {
                    "staging_required": True,
                    "rollback_strategy": {"automated": True, "threshold": "5% error_rate"},
                    "health_checks": {"timeout": 30, "retries": 3}
                }
            },
            "project_patterns": {
                "web_service": {
                    "typical_build_time": 120,
                    "test_suite_time": 300,
                    "deployment_time": 180,
                    "common_issues": ["dependency_conflicts", "environment_differences"]
                },
                "ml_pipeline": {
                    "typical_build_time": 600,
                    "test_suite_time": 900,
                    "deployment_time": 300,
                    "common_issues": ["data_validation", "model_compatibility", "resource_limits"]
                }
            },
            "failure_patterns": {
                "timeout_issues": {
                    "indicators": ["timeout", "TimeoutError", "timed out"],
                    "solutions": ["increase_timeout", "optimize_performance", "parallelize"]
                },
                "dependency_issues": {
                    "indicators": ["ModuleNotFoundError", "ImportError", "version conflict"],
                    "solutions": ["update_dependencies", "resolve_conflicts", "pin_versions"]
                }
            }
        }
    
    def _initialize_cognitive_rules(self) -> None:
        """Initialize cognitive rules for decision-making."""
        # Testing strategy rules
        self.procedural_memory.append(CognitiveRule(
            rule_id="high_coverage_testing",
            condition=lambda ctx: ctx.get("test_coverage", 0) < 0.8,
            action=lambda ctx: {"action": "increase_test_coverage", "priority": "high"},
            priority=9,
            confidence=0.9
        ))
        
        # Security rules
        self.procedural_memory.append(CognitiveRule(
            rule_id="security_vulnerability_detected",
            condition=lambda ctx: ctx.get("security_vulnerabilities", 0) > 0,
            action=lambda ctx: {"action": "fix_security_issues", "priority": "critical", "block_deployment": True},
            priority=10,
            confidence=0.95
        ))
        
        # Performance rules
        self.procedural_memory.append(CognitiveRule(
            rule_id="performance_regression",
            condition=lambda ctx: ctx.get("performance_regression", 0) > 0.1,
            action=lambda ctx: {"action": "investigate_performance", "priority": "high"},
            priority=8,
            confidence=0.85
        ))
        
        # Resource optimization rules
        self.procedural_memory.append(CognitiveRule(
            rule_id="resource_optimization",
            condition=lambda ctx: ctx.get("resource_usage", 0) > 0.8,
            action=lambda ctx: {"action": "optimize_resources", "suggestions": ["parallel_execution", "caching"]},
            priority=6,
            confidence=0.7
        ))
    
    async def make_cognitive_decision(self, 
                                    context: CognitiveContext,
                                    decision_type: str) -> Dict[str, Any]:
        """Make an intelligent cognitive decision based on context."""
        self.logger.info(f"Making cognitive decision: {decision_type}")
        
        start_time = time.time()
        
        with self.performance_monitor.measure("cognitive_decision"):
            # Update working memory with current context
            await self._update_working_memory(context)
            
            # Focus attention on relevant aspects
            await self._focus_attention(context, decision_type)
            
            # Generate decision options through reasoning
            options = await self._generate_decision_options(context, decision_type)
            
            # Evaluate options using multiple reasoning types
            evaluated_options = await self._evaluate_options(options, context)
            
            # Select best option
            best_option = await self._select_best_option(evaluated_options)
            
            # Record decision in memory
            await self._record_decision(context, best_option, decision_type)
            
            # Learn from the decision-making process
            await self._update_cognitive_patterns(context, best_option)
        
        decision_time = time.time() - start_time
        self.response_times.append(decision_time)
        
        self.logger.info(f"Cognitive decision completed in {decision_time:.3f}s with confidence {best_option.get('confidence', 0):.3f}")
        
        return best_option
    
    async def _update_working_memory(self, context: CognitiveContext) -> None:
        """Update working memory with current context."""
        self.working_memory.update({
            "current_context": context,
            "timestamp": time.time(),
            "project_state": context.project_state,
            "execution_history": context.execution_history[-5:],  # Recent history
            "current_goals": context.current_goals,
            "active_constraints": context.constraints
        })
    
    async def _focus_attention(self, context: CognitiveContext, decision_type: str) -> None:
        """Focus cognitive attention on relevant aspects."""
        # Clear previous attention
        self.attention_focus.clear()
        
        # Focus based on decision type
        if decision_type == "testing_strategy":
            self.attention_focus.extend(["test_coverage", "test_failures", "performance"])
        elif decision_type == "deployment_readiness":
            self.attention_focus.extend(["security_scan", "test_results", "performance_metrics"])
        elif decision_type == "failure_recovery":
            self.attention_focus.extend(["error_patterns", "failure_history", "available_fixes"])
        elif decision_type == "optimization":
            self.attention_focus.extend(["resource_usage", "execution_time", "bottlenecks"])
        
        # Add contextual focus based on current issues
        if context.risk_factors:
            self.attention_focus.extend(context.risk_factors)
        
        self.logger.debug(f"Attention focused on: {self.attention_focus}")
    
    async def _generate_decision_options(self, 
                                       context: CognitiveContext,
                                       decision_type: str) -> List[Dict[str, Any]]:
        """Generate decision options through cognitive reasoning."""
        options = []
        
        # Rule-based options (deductive reasoning)
        rule_options = await self._generate_rule_based_options(context)
        options.extend(rule_options)
        
        # Pattern-based options (inductive reasoning)
        pattern_options = await self._generate_pattern_based_options(context, decision_type)
        options.extend(pattern_options)
        
        # Analogical options (analogical reasoning)
        analogical_options = await self._generate_analogical_options(context, decision_type)
        options.extend(analogical_options)
        
        # Novel options (creative reasoning)
        if self.cognition_level in [CognitionLevel.STRATEGIC, CognitionLevel.METACOGNITIVE]:
            novel_options = await self._generate_novel_options(context, decision_type)
            options.extend(novel_options)
        
        # Remove duplicates and invalid options
        options = await self._filter_and_deduplicate_options(options)
        
        self.logger.debug(f"Generated {len(options)} decision options")
        return options
    
    async def _generate_rule_based_options(self, context: CognitiveContext) -> List[Dict[str, Any]]:
        """Generate options using cognitive rules (deductive reasoning)."""
        options = []
        
        for rule in self.procedural_memory:
            try:
                if rule.condition(context.project_state):
                    action = rule.action(context.project_state)
                    option = {
                        "type": "rule_based",
                        "source": rule.rule_id,
                        "action": action,
                        "confidence": rule.confidence * rule.success_rate,
                        "reasoning": ReasoningType.DEDUCTIVE,
                        "priority": rule.priority
                    }
                    options.append(option)
                    
                    # Update rule statistics
                    rule.activation_count += 1
                    rule.last_used = time.time()
                    
            except Exception as e:
                self.logger.warning(f"Rule {rule.rule_id} failed: {e}")
        
        return options
    
    async def _generate_pattern_based_options(self, 
                                            context: CognitiveContext,
                                            decision_type: str) -> List[Dict[str, Any]]:
        """Generate options based on learned patterns (inductive reasoning)."""
        options = []
        
        # Find similar contexts in episodic memory
        similar_experiences = await self._find_similar_experiences(context)
        
        for experience in similar_experiences[:5]:  # Top 5 similar
            if experience.success:
                option = {
                    "type": "pattern_based",
                    "source": f"experience_{experience.experience_id}",
                    "action": experience.decision,
                    "confidence": experience.confidence * 0.8,  # Reduced confidence for reuse
                    "reasoning": ReasoningType.INDUCTIVE,
                    "similarity": self._calculate_context_similarity(context, experience.context),
                    "past_success": experience.success
                }
                options.append(option)
        
        # Generate options from semantic patterns
        pattern_key = self._generate_pattern_key(context, decision_type)
        if pattern_key in self.pattern_library:
            for pattern in self.pattern_library[pattern_key]:
                option = {
                    "type": "semantic_pattern",
                    "source": f"pattern_{pattern_key}",
                    "action": pattern.get("action", {}),
                    "confidence": pattern.get("confidence", 0.6),
                    "reasoning": ReasoningType.INDUCTIVE,
                    "usage_count": pattern.get("usage_count", 0)
                }
                options.append(option)
        
        return options
    
    async def _generate_analogical_options(self, 
                                         context: CognitiveContext,
                                         decision_type: str) -> List[Dict[str, Any]]:
        """Generate options using analogical reasoning."""
        options = []
        
        # Find analogous situations in knowledge base
        analogies = await self._find_analogous_situations(context, decision_type)
        
        for analogy in analogies:
            # Adapt solution from analogous situation
            adapted_action = await self._adapt_analogical_solution(
                analogy["solution"], 
                context, 
                analogy["similarity"]
            )
            
            if adapted_action:
                option = {
                    "type": "analogical",
                    "source": f"analogy_{analogy['situation']}",
                    "action": adapted_action,
                    "confidence": analogy["similarity"] * 0.7,
                    "reasoning": ReasoningType.ANALOGICAL,
                    "analogy_source": analogy["situation"]
                }
                options.append(option)
        
        return options
    
    async def _generate_novel_options(self, 
                                    context: CognitiveContext,
                                    decision_type: str) -> List[Dict[str, Any]]:
        """Generate novel options through creative reasoning."""
        options = []
        
        # Combine existing approaches in novel ways
        existing_actions = [opt.get("action", {}) for opt in 
                          await self._generate_rule_based_options(context)]
        
        if len(existing_actions) >= 2:
            # Combine two different actions
            for i in range(len(existing_actions)):
                for j in range(i + 1, len(existing_actions)):
                    combined_action = await self._combine_actions(
                        existing_actions[i], 
                        existing_actions[j]
                    )
                    
                    if combined_action:
                        option = {
                            "type": "novel_combination",
                            "source": "creative_reasoning",
                            "action": combined_action,
                            "confidence": 0.6,  # Lower confidence for novel approaches
                            "reasoning": ReasoningType.ABDUCTIVE,
                            "novelty": True
                        }
                        options.append(option)
        
        # Generate innovative solutions based on current trends
        if decision_type == "optimization":
            innovative_optimizations = [
                {
                    "type": "innovative",
                    "source": "trend_analysis",
                    "action": {
                        "action": "implement_ai_optimization",
                        "details": "Use machine learning for dynamic parameter tuning"
                    },
                    "confidence": 0.5,
                    "reasoning": ReasoningType.ABDUCTIVE,
                    "innovation_level": "high"
                },
                {
                    "type": "innovative", 
                    "source": "quantum_inspiration",
                    "action": {
                        "action": "quantum_parallel_execution",
                        "details": "Implement superposition-based parallel task execution"
                    },
                    "confidence": 0.4,
                    "reasoning": ReasoningType.ABDUCTIVE,
                    "innovation_level": "experimental"
                }
            ]
            options.extend(innovative_optimizations)
        
        return options
    
    async def _evaluate_options(self, 
                              options: List[Dict[str, Any]], 
                              context: CognitiveContext) -> List[Dict[str, Any]]:
        """Evaluate decision options using multiple criteria."""
        evaluated_options = []
        
        for option in options:
            # Base evaluation scores
            scores = {
                "confidence": option.get("confidence", 0.5),
                "priority": option.get("priority", 5) / 10.0,
                "feasibility": await self._assess_feasibility(option, context),
                "risk": await self._assess_risk(option, context),
                "impact": await self._assess_impact(option, context),
                "resource_cost": await self._assess_resource_cost(option, context)
            }
            
            # Adjust scores based on reasoning type
            reasoning_weights = {
                ReasoningType.DEDUCTIVE: 1.0,     # High weight for proven rules
                ReasoningType.INDUCTIVE: 0.8,     # Good weight for patterns
                ReasoningType.ANALOGICAL: 0.6,    # Medium weight for analogies
                ReasoningType.ABDUCTIVE: 0.4      # Lower weight for novel ideas
            }
            
            reasoning_type = option.get("reasoning", ReasoningType.DEDUCTIVE)
            reasoning_weight = reasoning_weights.get(reasoning_type, 0.5)
            
            # Calculate weighted score
            weighted_score = (
                scores["confidence"] * 0.3 +
                scores["priority"] * 0.2 +
                scores["feasibility"] * 0.2 +
                (1 - scores["risk"]) * 0.15 +  # Lower risk is better
                scores["impact"] * 0.1 +
                (1 - scores["resource_cost"]) * 0.05  # Lower cost is better
            ) * reasoning_weight
            
            option["evaluation"] = {
                "scores": scores,
                "weighted_score": weighted_score,
                "reasoning_weight": reasoning_weight
            }
            
            evaluated_options.append(option)
        
        # Sort by weighted score
        evaluated_options.sort(key=lambda x: x["evaluation"]["weighted_score"], reverse=True)
        
        return evaluated_options
    
    async def _assess_feasibility(self, option: Dict[str, Any], context: CognitiveContext) -> float:
        """Assess the feasibility of implementing the option."""
        action = option.get("action", {})
        
        # Check resource requirements
        if action.get("requires_external_service", False):
            if "external_services" not in context.available_resources:
                return 0.3  # Low feasibility
        
        # Check time requirements
        estimated_time = action.get("estimated_time", 300)  # Default 5 minutes
        available_time = context.constraints.get("max_execution_time", 3600)
        if estimated_time > available_time:
            return 0.4  # Low feasibility due to time constraints
        
        # Check complexity
        complexity = action.get("complexity", "medium")
        complexity_scores = {"low": 0.9, "medium": 0.7, "high": 0.5, "very_high": 0.3}
        
        return complexity_scores.get(complexity, 0.7)
    
    async def _assess_risk(self, option: Dict[str, Any], context: CognitiveContext) -> float:
        """Assess the risk level of the option."""
        action = option.get("action", {})
        
        risk_factors = 0.0
        
        # Novel approaches have higher risk
        if option.get("type") in ["novel_combination", "innovative"]:
            risk_factors += 0.3
        
        # Actions that block deployment have higher risk
        if action.get("block_deployment", False):
            risk_factors += 0.4
        
        # Actions with low confidence have higher risk
        confidence = option.get("confidence", 0.5)
        if confidence < 0.5:
            risk_factors += 0.3
        
        # Past failure increases risk
        if option.get("past_failures", 0) > 0:
            risk_factors += min(0.4, option["past_failures"] * 0.1)
        
        return min(1.0, risk_factors)
    
    async def _assess_impact(self, option: Dict[str, Any], context: CognitiveContext) -> float:
        """Assess the potential positive impact of the option."""
        action = option.get("action", {})
        
        impact_score = 0.5  # Base impact
        
        # Security fixes have high impact
        if "security" in action.get("action", "").lower():
            impact_score += 0.3
        
        # Performance improvements have good impact
        if "performance" in action.get("action", "").lower() or "optimize" in action.get("action", "").lower():
            impact_score += 0.2
        
        # Quality improvements have moderate impact
        if "quality" in action.get("action", "").lower() or "test" in action.get("action", "").lower():
            impact_score += 0.15
        
        # Critical priority has high impact
        if action.get("priority") == "critical":
            impact_score += 0.2
        
        return min(1.0, impact_score)
    
    async def _assess_resource_cost(self, option: Dict[str, Any], context: CognitiveContext) -> float:
        """Assess the resource cost of implementing the option."""
        action = option.get("action", {})
        
        # Base cost
        base_cost = 0.3
        
        # Parallel execution increases cost
        if action.get("parallel_execution", False):
            base_cost += 0.2
        
        # External services increase cost
        if action.get("requires_external_service", False):
            base_cost += 0.3
        
        # Complexity affects cost
        complexity = action.get("complexity", "medium")
        complexity_costs = {"low": 0.1, "medium": 0.2, "high": 0.4, "very_high": 0.6}
        base_cost += complexity_costs.get(complexity, 0.2)
        
        return min(1.0, base_cost)
    
    async def _select_best_option(self, evaluated_options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best option from evaluated candidates."""
        if not evaluated_options:
            return {"action": "no_action", "confidence": 0.0, "reasoning": "no_options_available"}
        
        # Get top option
        best_option = evaluated_options[0].copy()
        
        # Add selection metadata
        best_option["selection_metadata"] = {
            "total_options": len(evaluated_options),
            "selection_time": time.time(),
            "selection_reasoning": "highest_weighted_score",
            "confidence_level": self._determine_confidence_level(best_option["evaluation"]["weighted_score"])
        }
        
        return best_option
    
    def _determine_confidence_level(self, score: float) -> DecisionConfidence:
        """Determine confidence level based on score."""
        if score >= 0.8:
            return DecisionConfidence.VERY_HIGH
        elif score >= 0.6:
            return DecisionConfidence.HIGH
        elif score >= 0.4:
            return DecisionConfidence.MEDIUM
        else:
            return DecisionConfidence.LOW
    
    async def _record_decision(self, 
                             context: CognitiveContext, 
                             decision: Dict[str, Any],
                             decision_type: str) -> None:
        """Record decision in cognitive memory."""
        memory = CognitiveMemory(
            experience_id=hashlib.md5(f"{time.time()}_{decision_type}".encode()).hexdigest()[:8],
            context=context.project_state.copy(),
            decision=decision.copy(),
            outcome={},  # Will be filled when outcome is known
            confidence=decision.get("confidence", 0.5),
            reasoning_type=decision.get("reasoning", ReasoningType.DEDUCTIVE),
            timestamp=time.time(),
            success=True  # Assume success until proven otherwise
        )
        
        self.episodic_memory.append(memory)
        
        # Manage memory capacity
        if len(self.episodic_memory) > self.memory_capacity:
            # Remove oldest memories, keeping successful ones longer
            self.episodic_memory = sorted(self.episodic_memory, 
                                        key=lambda m: (m.success, m.timestamp))
            self.episodic_memory = self.episodic_memory[100:]  # Keep recent 100
    
    async def _update_cognitive_patterns(self, 
                                       context: CognitiveContext,
                                       decision: Dict[str, Any]) -> None:
        """Update cognitive patterns based on decision-making experience."""
        pattern_key = self._generate_pattern_key(context, decision.get("action", {}).get("action", ""))
        
        pattern_data = {
            "action": decision.get("action", {}),
            "confidence": decision.get("confidence", 0.5),
            "context_features": self._extract_context_features(context),
            "timestamp": time.time(),
            "usage_count": 1
        }
        
        # Update or create pattern
        if pattern_key in self.pattern_library:
            # Update existing pattern
            existing_patterns = self.pattern_library[pattern_key]
            similar_pattern = None
            
            for pattern in existing_patterns:
                if self._patterns_similar(pattern_data, pattern):
                    similar_pattern = pattern
                    break
            
            if similar_pattern:
                # Update existing similar pattern
                similar_pattern["usage_count"] += 1
                similar_pattern["confidence"] = (
                    similar_pattern["confidence"] * 0.8 + 
                    pattern_data["confidence"] * 0.2
                )
            else:
                # Add new pattern
                existing_patterns.append(pattern_data)
        else:
            # Create new pattern category
            self.pattern_library[pattern_key] = [pattern_data]
    
    def _generate_pattern_key(self, context: CognitiveContext, action: str) -> str:
        """Generate a key for pattern storage."""
        project_type = context.project_state.get("project_type", "unknown")
        complexity = context.project_state.get("complexity", "medium")
        return f"{project_type}_{complexity}_{action}"
    
    def _extract_context_features(self, context: CognitiveContext) -> Dict[str, Any]:
        """Extract key features from context for pattern matching."""
        return {
            "project_type": context.project_state.get("project_type", "unknown"),
            "complexity": context.project_state.get("complexity_score", 0.5),
            "has_tests": context.project_state.get("test_coverage", 0) > 0,
            "has_security_scan": "security_scan" in context.project_state,
            "execution_failures": len([h for h in context.execution_history if not h.get("success", True)]),
            "resource_constrained": context.constraints.get("resource_usage", 0) > 0.8
        }
    
    def _patterns_similar(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> bool:
        """Check if two patterns are similar."""
        features1 = pattern1.get("context_features", {})
        features2 = pattern2.get("context_features", {})
        
        similarity_score = 0.0
        total_features = 0
        
        for key in set(features1.keys()) | set(features2.keys()):
            total_features += 1
            if features1.get(key) == features2.get(key):
                similarity_score += 1
        
        return (similarity_score / total_features) > 0.7 if total_features > 0 else False
    
    async def _find_similar_experiences(self, context: CognitiveContext) -> List[CognitiveMemory]:
        """Find experiences similar to current context."""
        similarities = []
        
        for experience in self.episodic_memory:
            similarity = self._calculate_context_similarity(context, experience.context)
            if similarity > 0.5:
                similarities.append((similarity, experience))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [exp for _, exp in similarities]
    
    def _calculate_context_similarity(self, 
                                    context: CognitiveContext, 
                                    other_context: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts."""
        current_features = self._extract_context_features(context)
        other_features = other_context
        
        matching_features = 0
        total_features = 0
        
        for key in set(current_features.keys()) | set(other_features.keys()):
            total_features += 1
            if current_features.get(key) == other_features.get(key):
                matching_features += 1
        
        return matching_features / total_features if total_features > 0 else 0.0
    
    async def _find_analogous_situations(self, 
                                       context: CognitiveContext,
                                       decision_type: str) -> List[Dict[str, Any]]:
        """Find analogous situations in knowledge base."""
        analogies = []
        
        # Search in knowledge base patterns
        project_type = context.project_state.get("project_type", "unknown")
        
        for kb_type, kb_data in self.knowledge_base.get("project_patterns", {}).items():
            if kb_type != project_type:  # Look for different but analogous types
                similarity = self._calculate_type_similarity(project_type, kb_type)
                if similarity > 0.3:
                    analogy = {
                        "situation": kb_type,
                        "similarity": similarity,
                        "solution": kb_data,
                        "type": "project_pattern"
                    }
                    analogies.append(analogy)
        
        return analogies
    
    def _calculate_type_similarity(self, type1: str, type2: str) -> float:
        """Calculate similarity between project types."""
        # Simple similarity based on common words
        words1 = set(type1.lower().split("_"))
        words2 = set(type2.lower().split("_"))
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _adapt_analogical_solution(self, 
                                       solution: Dict[str, Any],
                                       context: CognitiveContext,
                                       similarity: float) -> Optional[Dict[str, Any]]:
        """Adapt a solution from analogous situation to current context."""
        if similarity < 0.3:
            return None
        
        # Create adapted solution
        adapted = solution.copy()
        
        # Adjust parameters based on context differences
        if context.project_state.get("complexity_score", 0.5) > 0.7:
            # Increase timeouts for complex projects
            if "typical_build_time" in adapted:
                adapted["typical_build_time"] *= 1.5
            if "test_suite_time" in adapted:
                adapted["test_suite_time"] *= 1.3
        
        # Add adaptation metadata
        adapted_action = {
            "action": "apply_analogical_solution",
            "solution_details": adapted,
            "adaptation_confidence": similarity,
            "original_context": solution.get("context", "unknown")
        }
        
        return adapted_action
    
    async def _combine_actions(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Combine two actions into a novel composite action."""
        # Check if actions are combinable
        if action1.get("block_deployment") and action2.get("block_deployment"):
            return None  # Can't combine two blocking actions
        
        # Create combined action
        combined = {
            "action": "composite_action",
            "components": [action1, action2],
            "sequence": "parallel" if self._actions_compatible(action1, action2) else "sequential",
            "estimated_time": max(action1.get("estimated_time", 0), action2.get("estimated_time", 0)),
            "complexity": "high",  # Combinations are inherently more complex
            "innovation": True
        }
        
        return combined
    
    def _actions_compatible(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> bool:
        """Check if two actions can be executed in parallel."""
        # Simple compatibility check
        incompatible_pairs = [
            ("security_scan", "deploy"),
            ("test", "optimize"),  # Tests might fail if optimization changes behavior
            ("rollback", "deploy")
        ]
        
        action1_name = action1.get("action", "").lower()
        action2_name = action2.get("action", "").lower()
        
        for incompatible in incompatible_pairs:
            if (incompatible[0] in action1_name and incompatible[1] in action2_name) or \
               (incompatible[1] in action1_name and incompatible[0] in action2_name):
                return False
        
        return True
    
    async def _filter_and_deduplicate_options(self, options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and deduplicate decision options."""
        if not options:
            return []
        
        # Remove duplicates based on action similarity
        unique_options = []
        seen_actions = set()
        
        for option in options:
            action_key = str(option.get("action", {}))
            action_hash = hashlib.md5(action_key.encode()).hexdigest()
            
            if action_hash not in seen_actions:
                seen_actions.add(action_hash)
                unique_options.append(option)
        
        # Filter out invalid options
        valid_options = []
        for option in unique_options:
            if self._is_valid_option(option):
                valid_options.append(option)
        
        return valid_options
    
    def _is_valid_option(self, option: Dict[str, Any]) -> bool:
        """Check if an option is valid."""
        # Must have an action
        if not option.get("action"):
            return False
        
        # Must have reasonable confidence
        if option.get("confidence", 0) < 0.1:
            return False
        
        # Must have source information
        if not option.get("source"):
            return False
        
        return True
    
    async def update_decision_outcome(self, 
                                    decision_id: str, 
                                    success: bool,
                                    outcome_data: Dict[str, Any]) -> None:
        """Update the outcome of a previous decision for learning."""
        # Find the decision in episodic memory
        for memory in reversed(self.episodic_memory):  # Search recent first
            if memory.experience_id == decision_id:
                memory.success = success
                memory.outcome = outcome_data
                
                # Update rule success rates if applicable
                if memory.decision.get("type") == "rule_based":
                    rule_id = memory.decision.get("source", "").replace("rule_", "")
                    for rule in self.procedural_memory:
                        if rule.rule_id == rule_id:
                            # Update success rate using exponential moving average
                            rule.success_rate = rule.success_rate * 0.9 + (1.0 if success else 0.0) * 0.1
                            break
                
                # Track decision accuracy for overall monitoring
                self.decision_accuracy.append(1.0 if success else 0.0)
                
                self.logger.info(f"Decision outcome updated: {decision_id}, success={success}")
                break
    
    def get_cognitive_statistics(self) -> Dict[str, Any]:
        """Get statistics about cognitive processing."""
        total_decisions = len(self.decision_accuracy)
        avg_accuracy = np.mean(self.decision_accuracy) if self.decision_accuracy else 0.0
        avg_response_time = np.mean(self.response_times) if self.response_times else 0.0
        
        # Rule statistics
        rule_stats = {}
        for rule in self.procedural_memory:
            rule_stats[rule.rule_id] = {
                "activation_count": rule.activation_count,
                "success_rate": rule.success_rate,
                "confidence": rule.confidence,
                "last_used": rule.last_used
            }
        
        return {
            "cognition_level": self.cognition_level.value,
            "total_decisions": total_decisions,
            "decision_accuracy": avg_accuracy,
            "average_response_time": avg_response_time,
            "episodic_memories": len(self.episodic_memory),
            "learned_patterns": len(self.pattern_library),
            "active_rules": len(self.procedural_memory),
            "working_memory_size": len(self.working_memory),
            "rule_statistics": rule_stats,
            "current_focus": self.attention_focus.copy()
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        optimize_memory()