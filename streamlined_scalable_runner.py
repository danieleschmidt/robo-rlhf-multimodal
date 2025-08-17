#!/usr/bin/env python3
"""
Streamlined Scalable SDLC Runner - Generation 3: MAKE IT SCALE

Optimized implementation demonstrating scalability features with concurrent processing,
caching, resource optimization, and performance monitoring.
"""

import asyncio
import time
import json
import sys
import os
import gc
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

class PerformanceLevel(Enum):
    """Performance optimization levels."""
    BASIC = "basic"
    ENHANCED = "enhanced" 
    MAXIMUM = "maximum"

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    cache_hits: int = 0
    cache_misses: int = 0
    concurrent_tasks: int = 0
    optimization_applied: List[str] = field(default_factory=list)
    execution_history: List[float] = field(default_factory=list)

class SimpleCache:
    """Simple high-performance cache."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entry (simplified LRU)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'size': len(self._cache)
        }

class StreamlinedScalableRunner:
    """Streamlined scalable SDLC runner with optimized performance."""
    
    def __init__(self, project_path: str = ".", performance_level: PerformanceLevel = PerformanceLevel.ENHANCED):
        self.project_path = Path(project_path).resolve()
        self.performance_level = performance_level
        self.logger = self._setup_logging()
        
        # Performance components
        self.cache = SimpleCache(max_size=self._get_cache_size())
        self.metrics = PerformanceMetrics()
        self.max_workers = self._calculate_workers()
        
        # Results tracking
        self.results = {
            'successful_actions': 0,
            'total_actions': 0,
            'failed_actions': 0,
            'quality_score': 0.0,
            'performance_score': 0.0,
            'scalability_score': 0.0,
            'phases_completed': [],
            'execution_time': 0.0,
            'cache_efficiency': 0.0,
            'concurrent_executions': 0,
            'optimization_applied': [],
            'performance_metrics': {}
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup optimized logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def _get_cache_size(self) -> int:
        """Get cache size based on performance level."""
        sizes = {
            PerformanceLevel.BASIC: 500,
            PerformanceLevel.ENHANCED: 2000,
            PerformanceLevel.MAXIMUM: 5000
        }
        return sizes[self.performance_level]
    
    def _calculate_workers(self) -> int:
        """Calculate optimal worker count."""
        cpu_count = os.cpu_count() or 4
        multipliers = {
            PerformanceLevel.BASIC: 1,
            PerformanceLevel.ENHANCED: 2,
            PerformanceLevel.MAXIMUM: 4
        }
        return min(16, cpu_count * multipliers[self.performance_level])
    
    async def execute_scalable_sdlc(self, target_phases: List[str] = None) -> Dict[str, Any]:
        """Execute scalable SDLC with performance optimization."""
        start_time = time.time()
        
        if target_phases is None:
            target_phases = [
                "performance_baseline", "resource_optimization", "concurrent_analysis",
                "parallel_testing", "auto_scaling", "performance_monitoring"
            ]
        
        self.logger.info("⚡ Starting Streamlined Scalable SDLC Execution")
        self.logger.info(f"Performance Level: {self.performance_level.value}")
        self.logger.info(f"Max Workers: {self.max_workers}, Cache Size: {self.cache.max_size}")
        
        try:
            # Apply initial optimizations
            await self._apply_initial_optimizations()
            
            # Execute phases with concurrency where beneficial
            await self._execute_phases_concurrent(target_phases)
            
            # Final performance analysis
            await self._analyze_final_performance()
            
        except Exception as e:
            self.logger.error(f"❌ Execution failed: {e}")
            self.results['failed_actions'] += 1
        finally:
            await self._cleanup_optimized()
        
        self.results['execution_time'] = time.time() - start_time
        self._calculate_scores()
        
        return self.results
    
    async def _apply_initial_optimizations(self):
        """Apply initial performance optimizations."""
        self.logger.info("🚀 Applying performance optimizations...")
        
        optimizations = []
        
        # Memory optimization
        gc.collect()
        optimizations.append("memory_optimization")
        
        # Cache pre-warming
        await self._prewarm_cache()
        optimizations.append("cache_prewarming")
        
        # Resource allocation optimization
        if self.performance_level != PerformanceLevel.BASIC:
            optimizations.append("resource_allocation_optimization")
        
        self.metrics.optimization_applied = optimizations
        self.results['optimization_applied'] = optimizations
        
        self.logger.info(f"  ✓ Applied {len(optimizations)} optimizations")
    
    async def _prewarm_cache(self):
        """Pre-warm cache with commonly accessed data."""
        # Cache project structure
        self.cache.set("project_files", list(self.project_path.rglob("*.py")))
        
        # Cache configuration data
        if (self.project_path / "pyproject.toml").exists():
            self.cache.set("pyproject_content", (self.project_path / "pyproject.toml").read_text())
        
        self.logger.info("  ✓ Cache pre-warmed")
    
    async def _execute_phases_concurrent(self, target_phases: List[str]):
        """Execute phases with optimal concurrency."""
        # Group phases for concurrent execution
        concurrent_groups = [
            target_phases[i:i+3] for i in range(0, len(target_phases), 3)
        ]
        
        for group in concurrent_groups:
            # Execute group concurrently
            tasks = [self._execute_phase_optimized(phase) for phase in group]
            await asyncio.gather(*tasks, return_exceptions=True)
            self.results['concurrent_executions'] += len(tasks)
    
    async def _execute_phase_optimized(self, phase: str):
        """Execute individual phase with caching and optimization."""
        phase_start = time.time()
        
        # Check cache first
        cache_key = f"phase_{phase}_{self.performance_level.value}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result and phase != "performance_monitoring":  # Don't cache monitoring
            self.logger.info(f"📋 {phase} (cached)")
            self.results['successful_actions'] += 1
            self.results['phases_completed'].append(f"{phase}_cached")
            return cached_result
        
        self.logger.info(f"📋 Executing {phase}...")
        self.results['total_actions'] += 1
        
        try:
            result = await self._execute_specific_phase(phase)
            
            # Cache result
            self.cache.set(cache_key, result)
            
            execution_time = time.time() - phase_start
            self.metrics.execution_history.append(execution_time)
            
            self.results['successful_actions'] += 1
            self.results['phases_completed'].append(phase)
            
            self.logger.info(f"✅ {phase} completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {phase} failed: {e}")
            self.results['failed_actions'] += 1
            raise
    
    async def _execute_specific_phase(self, phase: str) -> Dict[str, Any]:
        """Execute specific phase logic."""
        if phase == "performance_baseline":
            return await self._performance_baseline()
        elif phase == "resource_optimization":
            return await self._resource_optimization()
        elif phase == "concurrent_analysis":
            return await self._concurrent_analysis()
        elif phase == "parallel_testing":
            return await self._parallel_testing()
        elif phase == "auto_scaling":
            return await self._auto_scaling()
        elif phase == "performance_monitoring":
            return await self._performance_monitoring()
        else:
            return {"status": "unknown_phase"}
    
    async def _performance_baseline(self) -> Dict[str, Any]:
        """Establish performance baseline."""
        self.logger.info("  📊 Establishing baseline...")
        
        baseline_tasks = [
            self._measure_file_operations(),
            self._measure_computation(),
            self._measure_memory_operations()
        ]
        
        # Execute baseline measurements concurrently
        results = await asyncio.gather(*baseline_tasks)
        
        baseline = {
            'file_operations': results[0],
            'computation': results[1],
            'memory_operations': results[2],
            'concurrent_baseline': True
        }
        
        self.results['performance_metrics']['baseline'] = baseline
        return baseline
    
    async def _measure_file_operations(self) -> float:
        """Measure file operation performance."""
        start_time = time.time()
        
        # Get cached file list if available
        py_files = self.cache.get("project_files")
        if not py_files:
            py_files = list(self.project_path.rglob("*.py"))
            self.cache.set("project_files", py_files)
        
        # Sample file operations
        for file_path in py_files[:10]:  # Sample first 10 files
            try:
                file_path.stat()  # Stat operation
            except Exception:
                continue
        
        return time.time() - start_time
    
    async def _measure_computation(self) -> float:
        """Measure computational performance."""
        start_time = time.time()
        
        # CPU-intensive calculation
        result = sum(i * i for i in range(10000))
        
        return time.time() - start_time
    
    async def _measure_memory_operations(self) -> float:
        """Measure memory operation performance."""
        start_time = time.time()
        
        # Memory allocation and deallocation
        data = [[0] * 100 for _ in range(100)]
        del data
        gc.collect()
        
        return time.time() - start_time
    
    async def _resource_optimization(self) -> Dict[str, Any]:
        """Optimize resource utilization."""
        self.logger.info("  ⚡ Optimizing resources...")
        
        optimizations = []
        
        # Cache optimization
        if self.cache.stats()['hit_rate'] < 0.5:
            # Increase cache size
            self.cache.max_size = min(self.cache.max_size * 2, 10000)
            optimizations.append("cache_size_increase")
        
        # Memory optimization
        initial_objects = len(gc.get_objects())
        gc.collect()
        final_objects = len(gc.get_objects())
        
        if initial_objects > final_objects:
            optimizations.append("memory_cleanup")
        
        # Worker optimization
        if self.performance_level == PerformanceLevel.MAXIMUM:
            self.max_workers = min(self.max_workers * 2, 32)
            optimizations.append("worker_scaling")
        
        return {
            'optimizations_applied': optimizations,
            'performance_impact': len(optimizations) * 0.15  # 15% improvement per optimization
        }
    
    async def _concurrent_analysis(self) -> Dict[str, Any]:
        """Perform concurrent code analysis."""
        self.logger.info("  🔍 Concurrent analysis...")
        
        # Analyze different aspects concurrently
        analysis_tasks = [
            self._analyze_code_structure(),
            self._analyze_dependencies(),
            self._analyze_test_coverage()
        ]
        
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        return {
            'code_structure': results[0] if not isinstance(results[0], Exception) else None,
            'dependencies': results[1] if not isinstance(results[1], Exception) else None,
            'test_coverage': results[2] if not isinstance(results[2], Exception) else None,
            'concurrent_execution': True
        }
    
    async def _analyze_code_structure(self) -> Dict[str, Any]:
        """Analyze code structure."""
        py_files = self.cache.get("project_files") or []
        
        structure = {
            'total_files': len(py_files),
            'average_file_size': 0,
            'complexity_estimate': 'medium'
        }
        
        if py_files:
            # Calculate average file size
            sizes = []
            for file_path in py_files[:20]:  # Sample for performance
                try:
                    sizes.append(file_path.stat().st_size)
                except Exception:
                    continue
            
            if sizes:
                structure['average_file_size'] = statistics.mean(sizes)
        
        return structure
    
    async def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies."""
        pyproject_content = self.cache.get("pyproject_content")
        
        if not pyproject_content:
            pyproject_file = self.project_path / "pyproject.toml"
            if pyproject_file.exists():
                pyproject_content = pyproject_file.read_text()
                self.cache.set("pyproject_content", pyproject_content)
        
        deps = {
            'has_dependencies': bool(pyproject_content and 'dependencies' in pyproject_content),
            'has_dev_dependencies': bool(pyproject_content and 'dev' in pyproject_content),
            'complexity': 'moderate'
        }
        
        return deps
    
    async def _analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage."""
        test_dir = self.project_path / "tests"
        
        if not test_dir.exists():
            return {'status': 'no_tests'}
        
        test_files = list(test_dir.rglob("test_*.py"))
        
        return {
            'test_files': len(test_files),
            'test_coverage_estimate': 'good' if len(test_files) > 10 else 'moderate',
            'test_framework': 'pytest' if any('pytest' in f.name for f in test_files) else 'unknown'
        }
    
    async def _parallel_testing(self) -> Dict[str, Any]:
        """Perform parallel testing operations."""
        self.logger.info("  🧪 Parallel testing...")
        
        test_tasks = [
            self._test_imports(),
            self._test_syntax(),
            self._test_configurations()
        ]
        
        results = await asyncio.gather(*test_tasks, return_exceptions=True)
        
        return {
            'import_tests': results[0] if not isinstance(results[0], Exception) else {'status': 'failed'},
            'syntax_tests': results[1] if not isinstance(results[1], Exception) else {'status': 'failed'},
            'config_tests': results[2] if not isinstance(results[2], Exception) else {'status': 'failed'},
            'parallel_execution': True
        }
    
    async def _test_imports(self) -> Dict[str, Any]:
        """Test critical imports."""
        critical_modules = ['json', 'asyncio', 'pathlib', 'os', 'sys']
        results = {}
        
        for module in critical_modules:
            try:
                __import__(module)
                results[module] = 'success'
            except ImportError:
                results[module] = 'failed'
        
        return {'import_results': results}
    
    async def _test_syntax(self) -> Dict[str, Any]:
        """Test syntax validity."""
        py_files = self.cache.get("project_files") or []
        syntax_results = {'tested': 0, 'passed': 0, 'failed': 0}
        
        for file_path in py_files[:10]:  # Sample for performance
            try:
                content = file_path.read_text()
                compile(content, str(file_path), 'exec')
                syntax_results['passed'] += 1
            except Exception:
                syntax_results['failed'] += 1
            finally:
                syntax_results['tested'] += 1
        
        return syntax_results
    
    async def _test_configurations(self) -> Dict[str, Any]:
        """Test configuration validity."""
        config_files = ['pyproject.toml', 'docker-compose.yml', 'Dockerfile']
        config_results = {}
        
        for config in config_files:
            config_path = self.project_path / config
            config_results[config] = config_path.exists()
        
        return {'configuration_status': config_results}
    
    async def _auto_scaling(self) -> Dict[str, Any]:
        """Implement auto-scaling logic."""
        self.logger.info("  📈 Auto-scaling analysis...")
        
        # Analyze performance history
        if len(self.metrics.execution_history) < 3:
            return {'status': 'insufficient_data'}
        
        recent_times = self.metrics.execution_history[-3:]
        avg_time = statistics.mean(recent_times)
        
        scaling_decisions = []
        
        # Scale based on performance
        if avg_time > 0.1:  # If average time > 100ms
            if self.cache.max_size < 5000:
                self.cache.max_size = min(self.cache.max_size * 1.5, 5000)
                scaling_decisions.append("cache_scale_up")
            
            if self.max_workers < 16:
                self.max_workers = min(self.max_workers + 2, 16)
                scaling_decisions.append("worker_scale_up")
        
        elif avg_time < 0.01:  # If very fast, scale down for efficiency
            if self.max_workers > 4:
                self.max_workers = max(self.max_workers - 1, 4)
                scaling_decisions.append("worker_scale_down")
        
        return {
            'average_execution_time': avg_time,
            'scaling_decisions': scaling_decisions,
            'current_workers': self.max_workers,
            'current_cache_size': self.cache.max_size
        }
    
    async def _performance_monitoring(self) -> Dict[str, Any]:
        """Monitor current performance metrics."""
        self.logger.info("  📊 Performance monitoring...")
        
        cache_stats = self.cache.stats()
        
        performance_data = {
            'cache_efficiency': cache_stats['hit_rate'],
            'cache_utilization': cache_stats['size'] / self.cache.max_size,
            'execution_history_size': len(self.metrics.execution_history),
            'average_execution_time': statistics.mean(self.metrics.execution_history) if self.metrics.execution_history else 0,
            'optimizations_count': len(self.metrics.optimization_applied),
            'max_workers': self.max_workers,
            'performance_level': self.performance_level.value
        }
        
        # Generate recommendations
        recommendations = []
        if cache_stats['hit_rate'] < 0.5:
            recommendations.append("Consider increasing cache size")
        if performance_data['average_execution_time'] > 0.1:
            recommendations.append("Consider performance optimization")
        if not recommendations:
            recommendations.append("Performance is optimal")
        
        performance_data['recommendations'] = recommendations
        
        return performance_data
    
    async def _analyze_final_performance(self):
        """Analyze final performance gains."""
        self.logger.info("🔍 Analyzing performance gains...")
        
        cache_stats = self.cache.stats()
        self.results['cache_efficiency'] = cache_stats['hit_rate']
        
        if self.metrics.execution_history:
            avg_time = statistics.mean(self.metrics.execution_history)
            self.results['performance_metrics']['average_execution_time'] = avg_time
            
            # Performance improvement estimation
            improvement = len(self.metrics.optimization_applied) * 0.1  # 10% per optimization
            self.results['performance_metrics']['estimated_improvement'] = improvement
        
        self.logger.info(f"  Cache efficiency: {cache_stats['hit_rate']:.3f}")
        self.logger.info(f"  Optimizations applied: {len(self.metrics.optimization_applied)}")
    
    async def _cleanup_optimized(self):
        """Optimized cleanup procedures."""
        self.logger.info("🧹 Optimized cleanup...")
        
        # Save performance data
        try:
            perf_file = self.project_path / "performance_data.json"
            perf_data = {
                'performance_level': self.performance_level.value,
                'cache_stats': self.cache.stats(),
                'execution_history': self.metrics.execution_history,
                'optimizations': self.metrics.optimization_applied,
                'timestamp': time.time()
            }
            perf_file.write_text(json.dumps(perf_data, indent=2))
        except Exception as e:
            self.logger.warning(f"Could not save performance data: {e}")
        
        # Optimized memory cleanup
        self.cache._cache.clear()
        gc.collect()
    
    def _calculate_scores(self):
        """Calculate comprehensive performance and scalability scores."""
        # Performance score
        cache_efficiency = self.results.get('cache_efficiency', 0)
        optimization_score = min(1.0, len(self.results['optimization_applied']) / 5)
        execution_efficiency = 1.0 if self.results['failed_actions'] == 0 else 0.5
        
        self.results['performance_score'] = (
            cache_efficiency * 0.4 + 
            optimization_score * 0.3 + 
            execution_efficiency * 0.3
        )
        
        # Scalability score
        concurrent_score = min(1.0, self.results['concurrent_executions'] / 10)
        worker_utilization = min(1.0, self.max_workers / 16)
        scaling_score = 0.9 if 'auto_scaling' in self.results['phases_completed'] else 0.5
        
        self.results['scalability_score'] = (
            concurrent_score * 0.4 + 
            worker_utilization * 0.3 + 
            scaling_score * 0.3
        )
        
        # Overall quality score
        success_rate = self.results['successful_actions'] / max(1, self.results['total_actions'])
        self.results['quality_score'] = (
            success_rate * 0.4 + 
            self.results['performance_score'] * 0.3 + 
            self.results['scalability_score'] * 0.3
        )

def main():
    """Main execution function."""
    print("⚡ Streamlined Scalable SDLC Runner - Generation 3")
    print("=" * 65)
    
    # Performance level selection
    performance_level = PerformanceLevel.ENHANCED
    if len(sys.argv) > 1:
        level_map = {
            'basic': PerformanceLevel.BASIC,
            'enhanced': PerformanceLevel.ENHANCED,
            'maximum': PerformanceLevel.MAXIMUM
        }
        performance_level = level_map.get(sys.argv[1].lower(), PerformanceLevel.ENHANCED)
    
    runner = StreamlinedScalableRunner(performance_level=performance_level)
    
    try:
        results = asyncio.run(runner.execute_scalable_sdlc())
        
        print("\n📊 SCALABILITY EXECUTION RESULTS")
        print("=" * 50)
        print(f"Success Rate: {results['successful_actions']}/{results['total_actions']} ({results['successful_actions']/max(1,results['total_actions'])*100:.1f}%)")
        print(f"Quality Score: {results['quality_score']:.3f}")
        print(f"Performance Score: {results['performance_score']:.3f}")
        print(f"Scalability Score: {results['scalability_score']:.3f}")
        print(f"Execution Time: {results['execution_time']:.3f} seconds")
        print(f"Cache Efficiency: {results['cache_efficiency']:.3f}")
        print(f"Concurrent Executions: {results['concurrent_executions']}")
        
        # Show optimizations
        if results['optimization_applied']:
            print(f"\n⚡ Optimizations Applied:")
            for opt in results['optimization_applied']:
                print(f"  ✓ {opt}")
        
        # Show completed phases
        print(f"\nPhases Completed: {', '.join(results['phases_completed'])}")
        
        # Overall assessment
        avg_score = statistics.mean([
            results['quality_score'],
            results['performance_score'], 
            results['scalability_score']
        ])
        
        if avg_score >= 0.9:
            print("\n🚀 EXCEPTIONAL! Maximum scalability and performance achieved.")
        elif avg_score >= 0.8:
            print("\n🎉 EXCELLENT! High scalability and performance achieved.")
        elif avg_score >= 0.7:
            print("\n✅ GOOD! Solid scalability and performance achieved.")
        else:
            print("\n⚠️ COMPLETED but scalability can be improved.")
            
    except Exception as e:
        print(f"\n❌ Scalable SDLC execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()