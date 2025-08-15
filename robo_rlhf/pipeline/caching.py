"""
Pipeline Caching: Advanced caching strategies and optimization for pipeline operations.

Implements intelligent caching, cache warming, and multi-tier cache management.
"""

import asyncio
import hashlib
import logging
import time
import json
import pickle
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
import statistics

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # Adaptive based on access patterns
    QUANTUM_OPTIMAL = "quantum"    # Quantum-optimized caching


class CacheLevel(Enum):
    """Cache levels in multi-tier architecture."""
    L1_MEMORY = "l1_memory"        # In-memory cache (fastest)
    L2_REDIS = "l2_redis"          # Redis cache (fast, distributed)
    L3_DISK = "l3_disk"            # Disk cache (slower, persistent)
    L4_REMOTE = "l4_remote"        # Remote cache (slowest, shared)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    total_size_bytes: int = 0
    avg_response_time: float = 0.0


class IntelligentCache:
    """
    Intelligent caching system with adaptive strategies and optimization.
    
    Features:
    - Multiple eviction strategies
    - Automatic cache warming
    - Performance-based optimization
    - Multi-tier caching support
    - Quantum-enhanced cache management
    """
    
    def __init__(
        self,
        name: str,
        max_size_mb: int = 100,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        default_ttl: Optional[float] = 3600.0  # 1 hour
    ):
        self.name = name
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.strategy = strategy
        self.default_ttl = default_ttl
        
        # Cache storage
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict = OrderedDict()  # For LRU
        self.access_frequency: Dict[str, int] = defaultdict(int)  # For LFU
        
        # Performance tracking
        self.stats = CacheStats()
        self.response_times: List[float] = []
        self.hit_rate_history: List[float] = []
        
        # Adaptive optimization
        self.optimization_enabled = True
        self.last_optimization = 0.0
        self.optimization_interval = 300.0  # 5 minutes
        
        # Cache warming
        self.warming_patterns: Dict[str, Callable] = {}
        self.warming_enabled = True
        
        # Quantum integration if available
        try:
            from robo_rlhf.quantum import QuantumOptimizer
            self.quantum_optimizer = QuantumOptimizer()
            self.quantum_enabled = True
            logger.info(f"Quantum-enhanced cache initialized: {name}")
        except ImportError:
            self.quantum_enabled = False
            logger.info(f"Standard cache initialized: {name}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.time()
        self.stats.total_requests += 1
        
        try:
            # Check if key exists and is not expired
            if key in self.entries:
                entry = self.entries[key]
                
                # Check TTL expiration
                if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                    await self._evict_key(key, reason="ttl_expired")
                    self.stats.misses += 1
                    return None
                
                # Update access metadata
                entry.access_count += 1
                entry.last_access = time.time()
                self.access_frequency[key] += 1
                
                # Update LRU order
                if key in self.access_order:
                    self.access_order.move_to_end(key)
                
                self.stats.hits += 1
                
                # Record response time
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                if len(self.response_times) > 1000:
                    self.response_times = self.response_times[-1000:]
                
                return entry.value
            
            else:
                self.stats.misses += 1
                return None
                
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.stats.misses += 1
            return None
        
        finally:
            # Update average response time
            response_time = time.time() - start_time
            alpha = 0.1  # Exponential moving average factor
            self.stats.avg_response_time = (
                alpha * response_time + (1 - alpha) * self.stats.avg_response_time
            )
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set value in cache."""
        try:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Check if we need to make space
            if not await self._ensure_space(size_bytes):
                logger.warning(f"Could not make space for cache entry: {key}")
                return False
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes,
                metadata=metadata or {}
            )
            
            # Remove old entry if exists
            if key in self.entries:
                await self._evict_key(key, reason="overwrite")
            
            # Add new entry
            self.entries[key] = entry
            self.access_order[key] = True
            self.stats.total_size_bytes += size_bytes
            
            # Trigger optimization if needed
            if self.optimization_enabled:
                await self._maybe_optimize()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.entries:
            await self._evict_key(key, reason="manual_delete")
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        self.entries.clear()
        self.access_order.clear()
        self.access_frequency.clear()
        self.stats.total_size_bytes = 0
        logger.info(f"Cache {self.name} cleared")
    
    async def _ensure_space(self, required_bytes: int) -> bool:
        """Ensure there's enough space for new entry."""
        available_space = self.max_size_bytes - self.stats.total_size_bytes
        
        if available_space >= required_bytes:
            return True
        
        # Need to evict entries
        bytes_to_free = required_bytes - available_space
        return await self._evict_entries(bytes_to_free)
    
    async def _evict_entries(self, bytes_to_free: int) -> bool:
        """Evict entries to free up space."""
        freed_bytes = 0
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used entries
            while freed_bytes < bytes_to_free and self.access_order:
                key = next(iter(self.access_order))
                entry = self.entries.get(key)
                if entry:
                    freed_bytes += entry.size_bytes
                    await self._evict_key(key, reason="lru_eviction")
                else:
                    self.access_order.pop(key, None)
        
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used entries
            sorted_keys = sorted(
                self.access_frequency.keys(),
                key=lambda k: self.access_frequency[k]
            )
            
            for key in sorted_keys:
                if freed_bytes >= bytes_to_free:
                    break
                entry = self.entries.get(key)
                if entry:
                    freed_bytes += entry.size_bytes
                    await self._evict_key(key, reason="lfu_eviction")
        
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first, then oldest
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self.entries.items():
                if entry.ttl and current_time - entry.timestamp > entry.ttl:
                    expired_keys.append(key)
            
            # Evict expired entries
            for key in expired_keys:
                if freed_bytes >= bytes_to_free:
                    break
                entry = self.entries.get(key)
                if entry:
                    freed_bytes += entry.size_bytes
                    await self._evict_key(key, reason="ttl_eviction")
            
            # If not enough, evict oldest entries
            if freed_bytes < bytes_to_free:
                sorted_keys = sorted(
                    self.entries.keys(),
                    key=lambda k: self.entries[k].timestamp
                )
                
                for key in sorted_keys:
                    if freed_bytes >= bytes_to_free:
                        break
                    entry = self.entries.get(key)
                    if entry:
                        freed_bytes += entry.size_bytes
                        await self._evict_key(key, reason="age_eviction")
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Use adaptive eviction based on access patterns
            await self._adaptive_eviction(bytes_to_free)
        
        elif self.strategy == CacheStrategy.QUANTUM_OPTIMAL and self.quantum_enabled:
            # Use quantum optimization for eviction
            await self._quantum_eviction(bytes_to_free)
        
        return freed_bytes >= bytes_to_free
    
    async def _adaptive_eviction(self, bytes_to_free: int) -> None:
        """Adaptive eviction strategy based on access patterns."""
        freed_bytes = 0
        current_time = time.time()
        
        # Score entries based on multiple factors
        entry_scores = {}
        
        for key, entry in self.entries.items():
            # Factors for eviction score (higher = more likely to evict)
            recency_score = current_time - entry.last_access  # Older = higher score
            frequency_score = 1.0 / (entry.access_count + 1)  # Less frequent = higher score
            size_score = entry.size_bytes / 1024  # Larger = higher score
            ttl_score = 0.0
            
            if entry.ttl:
                time_until_expiry = entry.ttl - (current_time - entry.timestamp)
                if time_until_expiry <= 0:
                    ttl_score = 1000  # Expired entries get highest score
                else:
                    ttl_score = 1.0 / time_until_expiry  # Soon to expire = higher score
            
            # Weighted combination
            total_score = (
                recency_score * 0.3 +
                frequency_score * 0.3 +
                size_score * 0.2 +
                ttl_score * 0.2
            )
            
            entry_scores[key] = total_score
        
        # Sort by score (highest first) and evict
        sorted_keys = sorted(entry_scores.keys(), key=lambda k: entry_scores[k], reverse=True)
        
        for key in sorted_keys:
            if freed_bytes >= bytes_to_free:
                break
            entry = self.entries.get(key)
            if entry:
                freed_bytes += entry.size_bytes
                await self._evict_key(key, reason="adaptive_eviction")
    
    async def _quantum_eviction(self, bytes_to_free: int) -> None:
        """Quantum-optimized eviction strategy."""
        try:
            # Prepare data for quantum optimization
            entries_data = {}
            
            for key, entry in self.entries.items():
                entries_data[key] = {
                    "size_bytes": entry.size_bytes,
                    "access_count": entry.access_count,
                    "last_access": entry.last_access,
                    "timestamp": entry.timestamp,
                    "ttl": entry.ttl
                }
            
            # Get quantum-optimized eviction list
            eviction_list = await self.quantum_optimizer.optimize_cache_eviction(
                entries=entries_data,
                bytes_to_free=bytes_to_free,
                cache_stats=self._get_stats_dict()
            )
            
            # Evict according to quantum recommendations
            freed_bytes = 0
            for key in eviction_list:
                if freed_bytes >= bytes_to_free:
                    break
                entry = self.entries.get(key)
                if entry:
                    freed_bytes += entry.size_bytes
                    await self._evict_key(key, reason="quantum_eviction")
            
        except Exception as e:
            logger.error(f"Quantum eviction failed: {e}")
            # Fallback to adaptive eviction
            await self._adaptive_eviction(bytes_to_free)
    
    async def _evict_key(self, key: str, reason: str) -> None:
        """Evict a specific key from cache."""
        if key in self.entries:
            entry = self.entries[key]
            self.stats.total_size_bytes -= entry.size_bytes
            self.stats.evictions += 1
            
            del self.entries[key]
            self.access_order.pop(key, None)
            self.access_frequency.pop(key, None)
            
            logger.debug(f"Evicted cache key {key} ({reason})")
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, dict)):
                return len(pickle.dumps(value))
            else:
                # Generic serialization
                return len(pickle.dumps(value))
        except Exception:
            # Fallback estimate
            return 1024  # 1KB default
    
    async def _maybe_optimize(self) -> None:
        """Trigger optimization if conditions are met."""
        current_time = time.time()
        
        if current_time - self.last_optimization < self.optimization_interval:
            return
        
        self.last_optimization = current_time
        
        # Calculate current hit rate
        hit_rate = self.get_hit_rate()
        self.hit_rate_history.append(hit_rate)
        
        # Keep only recent history
        if len(self.hit_rate_history) > 100:
            self.hit_rate_history = self.hit_rate_history[-100:]
        
        # Optimization logic
        await self._optimize_cache_parameters()
    
    async def _optimize_cache_parameters(self) -> None:
        """Optimize cache parameters based on performance."""
        if len(self.hit_rate_history) < 5:
            return
        
        recent_hit_rate = statistics.mean(self.hit_rate_history[-5:])
        
        # Adaptive strategy switching
        if self.strategy == CacheStrategy.ADAPTIVE:
            if recent_hit_rate < 0.5:
                # Low hit rate, maybe TTL is too short
                if self.default_ttl and self.default_ttl < 7200:  # Less than 2 hours
                    self.default_ttl *= 1.2
                    logger.info(f"Increased cache TTL to {self.default_ttl:.0f}s (hit rate: {recent_hit_rate:.2f})")
            
            elif recent_hit_rate > 0.9:
                # Very high hit rate, maybe we can reduce TTL to save memory
                if self.default_ttl and self.default_ttl > 300:  # More than 5 minutes
                    self.default_ttl *= 0.9
                    logger.info(f"Decreased cache TTL to {self.default_ttl:.0f}s (hit rate: {recent_hit_rate:.2f})")
    
    def register_warming_pattern(self, pattern_name: str, warming_func: Callable) -> None:
        """Register a cache warming pattern."""
        self.warming_patterns[pattern_name] = warming_func
        logger.info(f"Registered cache warming pattern: {pattern_name}")
    
    async def warm_cache(self, pattern_name: Optional[str] = None) -> Dict[str, int]:
        """Warm cache using registered patterns."""
        if not self.warming_enabled:
            return {}
        
        results = {}
        
        patterns_to_run = []
        if pattern_name:
            if pattern_name in self.warming_patterns:
                patterns_to_run = [pattern_name]
        else:
            patterns_to_run = list(self.warming_patterns.keys())
        
        for pattern in patterns_to_run:
            try:
                warming_func = self.warming_patterns[pattern]
                
                # Execute warming function
                start_time = time.time()
                items = await warming_func()
                
                # Cache the items
                cached_count = 0
                if isinstance(items, dict):
                    for key, value in items.items():
                        if await self.set(key, value):
                            cached_count += 1
                elif isinstance(items, list):
                    for item in items:
                        if isinstance(item, tuple) and len(item) >= 2:
                            key, value = item[0], item[1]
                            if await self.set(key, value):
                                cached_count += 1
                
                duration = time.time() - start_time
                results[pattern] = cached_count
                
                logger.info(
                    f"Cache warming pattern '{pattern}' completed: "
                    f"{cached_count} items cached in {duration:.2f}s"
                )
                
            except Exception as e:
                logger.error(f"Cache warming pattern '{pattern}' failed: {e}")
                results[pattern] = 0
        
        return results
    
    def get_hit_rate(self) -> float:
        """Calculate current hit rate."""
        if self.stats.total_requests == 0:
            return 0.0
        return self.stats.hits / self.stats.total_requests
    
    def get_size_mb(self) -> float:
        """Get current cache size in MB."""
        return self.stats.total_size_bytes / (1024 * 1024)
    
    def _get_stats_dict(self) -> Dict[str, Any]:
        """Get cache statistics as dictionary."""
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.get_hit_rate(),
            "evictions": self.stats.evictions,
            "total_requests": self.stats.total_requests,
            "size_mb": self.get_size_mb(),
            "entry_count": len(self.entries),
            "avg_response_time": self.stats.avg_response_time
        }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        stats = self._get_stats_dict()
        
        # Add additional details
        if self.response_times:
            stats["response_time_p95"] = sorted(self.response_times)[int(0.95 * len(self.response_times))]
            stats["response_time_p99"] = sorted(self.response_times)[int(0.99 * len(self.response_times))]
        
        if self.hit_rate_history:
            stats["hit_rate_trend"] = statistics.mean(self.hit_rate_history[-10:]) if len(self.hit_rate_history) >= 10 else self.get_hit_rate()
        
        stats.update({
            "name": self.name,
            "strategy": self.strategy.value,
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "default_ttl": self.default_ttl,
            "quantum_enabled": self.quantum_enabled,
            "warming_patterns": len(self.warming_patterns)
        })
        
        return stats


class MultiTierCache:
    """
    Multi-tier caching system with intelligent data placement.
    
    Manages multiple cache levels with automatic promotion/demotion.
    """
    
    def __init__(self, name: str):
        self.name = name
        
        # Initialize cache tiers
        self.tiers: Dict[CacheLevel, IntelligentCache] = {
            CacheLevel.L1_MEMORY: IntelligentCache(f"{name}_L1", max_size_mb=50),
            CacheLevel.L2_REDIS: IntelligentCache(f"{name}_L2", max_size_mb=200),
            CacheLevel.L3_DISK: IntelligentCache(f"{name}_L3", max_size_mb=1000)
        }
        
        # Access pattern tracking
        self.access_patterns: Dict[str, List[float]] = {}
        self.promotion_threshold = 5  # Promote after 5 accesses in L2/L3
        self.demotion_threshold = 300  # Demote after 5 minutes without access
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-tier cache."""
        access_time = time.time()
        
        # Track access pattern
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        self.access_patterns[key].append(access_time)
        
        # Keep only recent accesses (last hour)
        cutoff_time = access_time - 3600
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff_time
        ]
        
        # Try tiers in order (L1 -> L2 -> L3)
        for level in [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DISK]:
            cache = self.tiers[level]
            value = await cache.get(key)
            
            if value is not None:
                # Consider promotion to higher tier
                await self._consider_promotion(key, value, level)
                return value
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        target_level: Optional[CacheLevel] = None
    ) -> bool:
        """Set value in multi-tier cache."""
        # Determine target level
        if target_level is None:
            target_level = await self._determine_optimal_level(key, value)
        
        # Set in target level
        success = await self.tiers[target_level].set(key, value, ttl)
        
        if success:
            # Also set in higher tiers if frequently accessed
            access_frequency = len(self.access_patterns.get(key, []))
            
            if access_frequency >= self.promotion_threshold:
                # Set in L1 as well
                if target_level != CacheLevel.L1_MEMORY:
                    await self.tiers[CacheLevel.L1_MEMORY].set(key, value, ttl)
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache tiers."""
        results = []
        for cache in self.tiers.values():
            results.append(await cache.delete(key))
        
        # Clean up access patterns
        self.access_patterns.pop(key, None)
        
        return any(results)
    
    async def _consider_promotion(self, key: str, value: Any, current_level: CacheLevel) -> None:
        """Consider promoting entry to higher tier."""
        access_frequency = len(self.access_patterns.get(key, []))
        
        if access_frequency >= self.promotion_threshold:
            if current_level == CacheLevel.L2_REDIS:
                # Promote to L1
                await self.tiers[CacheLevel.L1_MEMORY].set(key, value)
                logger.debug(f"Promoted {key} from L2 to L1 (access_frequency: {access_frequency})")
            
            elif current_level == CacheLevel.L3_DISK:
                # Promote to L2
                await self.tiers[CacheLevel.L2_REDIS].set(key, value)
                logger.debug(f"Promoted {key} from L3 to L2 (access_frequency: {access_frequency})")
    
    async def _determine_optimal_level(self, key: str, value: Any) -> CacheLevel:
        """Determine optimal cache level for new entry."""
        # Simple heuristic based on value size and access patterns
        value_size = len(pickle.dumps(value))
        access_frequency = len(self.access_patterns.get(key, []))
        
        if access_frequency >= self.promotion_threshold and value_size < 1024 * 10:  # 10KB
            return CacheLevel.L1_MEMORY
        elif value_size < 1024 * 100:  # 100KB
            return CacheLevel.L2_REDIS
        else:
            return CacheLevel.L3_DISK
    
    async def cleanup_stale_entries(self) -> Dict[str, int]:
        """Clean up stale entries across all tiers."""
        cleanup_results = {}
        current_time = time.time()
        
        for level, cache in self.tiers.items():
            cleaned = 0
            
            # Find stale entries
            stale_keys = []
            for key, entry in cache.entries.items():
                time_since_access = current_time - entry.last_access
                
                if time_since_access > self.demotion_threshold:
                    stale_keys.append(key)
            
            # Remove stale entries
            for key in stale_keys:
                await cache.delete(key)
                cleaned += 1
            
            cleanup_results[level.value] = cleaned
        
        return cleanup_results
    
    def get_tier_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all cache tiers."""
        return {
            level.value: cache.get_detailed_stats()
            for level, cache in self.tiers.items()
        }


class CacheManager:
    """
    Central cache management system for pipeline components.
    
    Coordinates multiple caches and provides unified cache operations.
    """
    
    def __init__(self):
        self.caches: Dict[str, Union[IntelligentCache, MultiTierCache]] = {}
        self.global_stats = CacheStats()
        self.warming_schedule: Dict[str, float] = {}  # cache_name -> next_warming_time
    
    def create_cache(
        self,
        name: str,
        cache_type: str = "intelligent",
        **kwargs
    ) -> Union[IntelligentCache, MultiTierCache]:
        """Create a new cache instance."""
        if cache_type == "intelligent":
            cache = IntelligentCache(name, **kwargs)
        elif cache_type == "multi_tier":
            cache = MultiTierCache(name)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
        
        self.caches[name] = cache
        logger.info(f"Created {cache_type} cache: {name}")
        return cache
    
    def get_cache(self, name: str) -> Optional[Union[IntelligentCache, MultiTierCache]]:
        """Get cache instance by name."""
        return self.caches.get(name)
    
    async def warm_all_caches(self) -> Dict[str, Dict[str, int]]:
        """Warm all caches according to their patterns."""
        results = {}
        
        for name, cache in self.caches.items():
            if isinstance(cache, IntelligentCache):
                cache_results = await cache.warm_cache()
                if cache_results:
                    results[name] = cache_results
        
        return results
    
    async def cleanup_all_caches(self) -> Dict[str, Any]:
        """Cleanup stale entries from all caches."""
        results = {}
        
        for name, cache in self.caches.items():
            if isinstance(cache, MultiTierCache):
                cleanup_results = await cache.cleanup_stale_entries()
                results[name] = cleanup_results
            elif isinstance(cache, IntelligentCache):
                # Basic cleanup for intelligent cache
                initial_count = len(cache.entries)
                current_time = time.time()
                
                expired_keys = []
                for key, entry in cache.entries.items():
                    if entry.ttl and current_time - entry.timestamp > entry.ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    await cache.delete(key)
                
                results[name] = {"expired_entries": len(expired_keys)}
        
        return results
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics for all caches."""
        total_stats = {
            "total_caches": len(self.caches),
            "total_hits": 0,
            "total_misses": 0,
            "total_requests": 0,
            "total_size_mb": 0.0,
            "average_hit_rate": 0.0,
            "cache_details": {}
        }
        
        hit_rates = []
        
        for name, cache in self.caches.items():
            if isinstance(cache, IntelligentCache):
                stats = cache.get_detailed_stats()
            elif isinstance(cache, MultiTierCache):
                # Aggregate stats from all tiers
                tier_stats = cache.get_tier_stats()
                stats = {
                    "hits": sum(t["hits"] for t in tier_stats.values()),
                    "misses": sum(t["misses"] for t in tier_stats.values()),
                    "total_requests": sum(t["total_requests"] for t in tier_stats.values()),
                    "size_mb": sum(t["size_mb"] for t in tier_stats.values()),
                    "hit_rate": 0.0,
                    "tiers": tier_stats
                }
                
                if stats["total_requests"] > 0:
                    stats["hit_rate"] = stats["hits"] / stats["total_requests"]
            else:
                continue
            
            total_stats["total_hits"] += stats["hits"]
            total_stats["total_misses"] += stats["misses"]
            total_stats["total_requests"] += stats["total_requests"]
            total_stats["total_size_mb"] += stats["size_mb"]
            
            if stats["hit_rate"] > 0:
                hit_rates.append(stats["hit_rate"])
            
            total_stats["cache_details"][name] = stats
        
        # Calculate average hit rate
        if hit_rates:
            total_stats["average_hit_rate"] = statistics.mean(hit_rates)
        
        # Calculate overall hit rate
        if total_stats["total_requests"] > 0:
            total_stats["overall_hit_rate"] = total_stats["total_hits"] / total_stats["total_requests"]
        else:
            total_stats["overall_hit_rate"] = 0.0
        
        return total_stats