"""
Optimized data collection with performance enhancements.
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass
import queue

from robo_rlhf.core.logging import get_logger
from robo_rlhf.core.performance import (
    measure_time, timer, cached, ThreadPool, BatchProcessor,
    ResourcePool, get_performance_monitor
)
from robo_rlhf.collectors.base import DemonstrationData, TeleOpCollector
from robo_rlhf.collectors.recorder import DemonstrationRecorder


logger = get_logger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for processing operations."""
    total_processed: int = 0
    processing_time: float = 0.0
    avg_processing_time: float = 0.0
    throughput: float = 0.0  # items per second
    errors: int = 0


class OptimizedDataProcessor:
    """Optimized data processor with batching and parallel processing."""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_workers: int = None,
        cache_size: int = 256,
        enable_compression: bool = True
    ):
        """
        Initialize optimized data processor.
        
        Args:
            batch_size: Size of processing batches
            max_workers: Number of worker threads
            cache_size: Size of result cache
            enable_compression: Whether to compress processed data
        """
        self.batch_size = batch_size
        self.thread_pool = ThreadPool(max_workers)
        self.batch_processor = BatchProcessor(batch_size, max_wait_time=0.5)
        self.enable_compression = enable_compression
        
        # Performance monitoring
        self.stats = ProcessingStats()
        self.perf_monitor = get_performance_monitor()
        
        logger.info("OptimizedDataProcessor initialized", extra={
            "batch_size": batch_size,
            "max_workers": max_workers or "auto",
            "cache_size": cache_size
        })
    
    @cached(maxsize=256, ttl=300)  # Cache for 5 minutes
    def preprocess_observations(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess observations with caching.
        
        Args:
            observations: Raw observations
            
        Returns:
            Preprocessed observations
        """
        with timer("preprocess_observations"):
            processed = {}
            
            for modality, data in observations.items():
                if modality in ['rgb', 'depth']:
                    # Image preprocessing
                    processed[modality] = self._preprocess_images(data)
                elif modality == 'proprioception':
                    # Proprioception preprocessing
                    processed[modality] = self._preprocess_proprioception(data)
                else:
                    # Default: normalize
                    processed[modality] = self._normalize_data(data)
            
            return processed
    
    def _preprocess_images(self, images: np.ndarray) -> np.ndarray:
        """Preprocess image data."""
        # Ensure uint8 format
        if images.dtype != np.uint8:
            images = (images * 255).astype(np.uint8)
        
        # Optional: resize, normalize, etc.
        return images
    
    def _preprocess_proprioception(self, data: np.ndarray) -> np.ndarray:
        """Preprocess proprioception data."""
        # Normalize to [-1, 1] range
        return np.tanh(data)
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Generic data normalization."""
        if data.dtype in [np.float32, np.float64]:
            return (data - np.mean(data)) / (np.std(data) + 1e-8)
        return data
    
    @measure_time
    def process_demonstrations_parallel(
        self,
        demonstrations: List[DemonstrationData],
        process_func: Callable
    ) -> List[Any]:
        """
        Process demonstrations in parallel.
        
        Args:
            demonstrations: List of demonstrations to process
            process_func: Function to apply to each demonstration
            
        Returns:
            List of processed results
        """
        start_time = time.time()
        
        # Submit all tasks
        futures = []
        for demo in demonstrations:
            future = self.thread_pool.submit(process_func, demo)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                self.stats.total_processed += 1
            except Exception as e:
                logger.error(f"Error processing demonstration: {e}")
                self.stats.errors += 1
        
        # Update statistics  
        processing_time = time.time() - start_time
        self.stats.processing_time += processing_time
        if self.stats.total_processed > 0:
            self.stats.avg_processing_time = self.stats.processing_time / self.stats.total_processed
            self.stats.throughput = self.stats.total_processed / self.stats.processing_time
        
        self.perf_monitor.increment_counter("demonstrations_processed", len(results))
        
        logger.info("Parallel processing completed", extra={
            "processed": len(results),
            "errors": self.stats.errors,
            "processing_time": processing_time,
            "throughput": len(results) / processing_time if processing_time > 0 else 0
        })
        
        return results
    
    def process_batch(self, items: List[Any], process_func: Callable) -> List[Any]:
        """Process a batch of items efficiently."""
        return self.batch_processor.submit(process_func, items)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_processed": self.stats.total_processed,
            "processing_time": self.stats.processing_time,
            "avg_processing_time": self.stats.avg_processing_time,
            "throughput": self.stats.throughput,
            "errors": self.stats.errors,
            "cache_stats": self.preprocess_observations.cache_stats()
        }
    
    def shutdown(self) -> None:
        """Shutdown the processor."""
        self.thread_pool.shutdown()
        self.batch_processor.shutdown()


class StreamingDataCollector(TeleOpCollector):
    """Streaming data collector with real-time processing."""
    
    def __init__(
        self,
        env,
        modalities: List[str] = ["rgb", "proprioception"],
        device: str = "keyboard",
        recording_fps: int = 30,
        stream_buffer_size: int = 1000,
        enable_real_time_processing: bool = True,
        **kwargs
    ):
        """
        Initialize streaming data collector.
        
        Args:
            env: Environment to collect from
            modalities: Observation modalities to collect
            device: Input device type
            recording_fps: Target recording framerate
            stream_buffer_size: Size of streaming buffer
            enable_real_time_processing: Enable real-time data processing
            **kwargs: Additional arguments for base collector
        """
        super().__init__(env, modalities, device, recording_fps, **kwargs)
        
        self.stream_buffer_size = stream_buffer_size
        self.enable_real_time_processing = enable_real_time_processing
        
        # Streaming components
        self.stream_buffer = queue.Queue(maxsize=stream_buffer_size)
        self.processed_buffer = queue.Queue(maxsize=stream_buffer_size)
        
        # Async processing
        self.processor = OptimizedDataProcessor()
        self.processing_thread = None
        self.streaming = False
        
        logger.info("StreamingDataCollector initialized", extra={
            "modalities": modalities,
            "fps": recording_fps,
            "buffer_size": stream_buffer_size,
            "real_time_processing": enable_real_time_processing
        })
    
    def start_streaming(self) -> None:
        """Start streaming data collection."""
        self.streaming = True
        
        if self.enable_real_time_processing:
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
        
        logger.info("Started streaming data collection")
    
    def stop_streaming(self) -> None:
        """Stop streaming data collection."""
        self.streaming = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Stopped streaming data collection")
    
    def _processing_loop(self) -> None:
        """Background processing loop for real-time data processing."""
        while self.streaming:
            try:
                # Get data from stream buffer
                data = self.stream_buffer.get(timeout=0.1)
                
                # Process data
                with timer("stream_processing"):
                    processed_data = self.processor.preprocess_observations(data['observations'])
                    
                    # Add to processed buffer
                    processed_item = {
                        'timestamp': data['timestamp'],
                        'observations': processed_data,
                        'action': data['action'],
                        'metadata': data.get('metadata', {})
                    }
                    
                    self.processed_buffer.put(processed_item)
                
                self.stream_buffer.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    def stream_step(self, observation: Dict[str, np.ndarray], action: np.ndarray) -> None:
        """Add a step to the streaming buffer."""
        if not self.streaming:
            return
        
        stream_data = {
            'timestamp': time.time(),
            'observations': observation,
            'action': action,
            'metadata': {'step': self.current_episode['frame_count'] if self.current_episode else 0}
        }
        
        try:
            self.stream_buffer.put_nowait(stream_data)
        except queue.Full:
            logger.warning("Stream buffer full, dropping frame")
    
    def get_processed_data(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get processed data from buffer."""
        try:
            return self.processed_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            "stream_buffer_size": self.stream_buffer.qsize(),
            "processed_buffer_size": self.processed_buffer.qsize(),
            "streaming": self.streaming,
            "processor_stats": self.processor.get_stats()
        }


class OptimizedRecorder(DemonstrationRecorder):
    """Optimized demonstration recorder with performance enhancements."""
    
    def __init__(
        self,
        output_dir: str,
        compression: bool = True,
        buffer_size: int = 10000,
        enable_parallel_saving: bool = True,
        save_thread_count: int = 2,
        **kwargs
    ):
        """
        Initialize optimized recorder.
        
        Args:
            output_dir: Directory to save demonstrations
            compression: Enable data compression
            buffer_size: Buffer size for async operations
            enable_parallel_saving: Enable parallel file saving
            save_thread_count: Number of threads for parallel saving
            **kwargs: Additional arguments for base recorder
        """
        super().__init__(output_dir, compression, buffer_size, **kwargs)
        
        self.enable_parallel_saving = enable_parallel_saving
        self.save_thread_pool = ThreadPool(save_thread_count) if enable_parallel_saving else None
        
        # Resource pool for file handles
        self.file_handle_pool = ResourcePool(
            create_func=lambda: {},  # Empty dict to store file handles
            reset_func=lambda handles: handles.clear(),
            max_size=save_thread_count * 2
        )
        
        logger.info("OptimizedRecorder initialized", extra={
            "parallel_saving": enable_parallel_saving,
            "save_threads": save_thread_count
        })
    
    @measure_time
    def save_demonstration_async(self, demo: DemonstrationData) -> None:
        """Save demonstration asynchronously."""
        if self.enable_parallel_saving and self.save_thread_pool:
            future = self.save_thread_pool.submit(self._save_demonstration_optimized, demo)
            return future
        else:
            return self._save_demonstration_optimized(demo)
    
    def _save_demonstration_optimized(self, demo: DemonstrationData) -> None:
        """Optimized demonstration saving with resource pooling."""
        with self.file_handle_pool.get_resource() as file_handles:
            with timer("save_demonstration"):
                # Save with resource pooling
                episode_path = self.output_dir / demo.episode_id
                episode_path.mkdir(parents=True, exist_ok=True)
                
                # Use compressed numpy format if enabled
                save_func = np.savez_compressed if self.compression else np.savez
                
                # Save all arrays in single compressed file
                arrays_to_save = {
                    'actions': demo.actions,
                    'metadata': demo.metadata
                }
                
                # Add observations
                for key, obs in demo.observations.items():
                    arrays_to_save[f'obs_{key}'] = obs
                
                if demo.rewards is not None:
                    arrays_to_save['rewards'] = demo.rewards
                
                # Save to compressed archive
                save_func(episode_path / "data.npz", **arrays_to_save)
                
                # Save metadata as JSON
                metadata_path = episode_path / "metadata.json"
                with open(metadata_path, 'w') as f:
                    import json
                    json.dump({
                        "episode_id": demo.episode_id,
                        "timestamp": demo.timestamp,
                        "success": demo.success,
                        "duration": demo.duration,
                        "metadata": demo.metadata or {}
                    }, f, indent=2)
    
    def batch_save_demonstrations(self, demonstrations: List[DemonstrationData]) -> None:
        """Save multiple demonstrations in parallel."""
        if not demonstrations:
            return
        
        start_time = time.time()
        
        # Submit all save tasks
        futures = []
        for demo in demonstrations:
            future = self.save_demonstration_async(demo)
            if future:
                futures.append(future)
        
        # Wait for completion
        if futures:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error saving demonstration: {e}")
        
        save_time = time.time() - start_time
        
        logger.info("Batch save completed", extra={
            "count": len(demonstrations),
            "save_time": save_time,
            "throughput": len(demonstrations) / save_time if save_time > 0 else 0
        })
    
    def shutdown(self) -> None:
        """Shutdown the optimized recorder."""
        super().cleanup()
        if self.save_thread_pool:
            self.save_thread_pool.shutdown()


def optimize_data_loading(
    data_dir: str,
    parallel_loading: bool = True,
    cache_loaded_data: bool = True,
    max_workers: int = None
) -> List[DemonstrationData]:
    """
    Optimized data loading with parallel processing and caching.
    
    Args:
        data_dir: Directory containing demonstration data
        parallel_loading: Enable parallel loading
        cache_loaded_data: Enable caching of loaded data
        max_workers: Number of worker threads
        
    Returns:
        List of loaded demonstrations
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return []
    
    # Find all demonstration directories
    demo_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not demo_dirs:
        logger.warning(f"No demonstration directories found in: {data_dir}")
        return []
    
    logger.info(f"Loading {len(demo_dirs)} demonstrations from {data_dir}")
    
    if parallel_loading:
        # Parallel loading
        with ThreadPool(max_workers) as pool:
            futures = [pool.submit(_load_single_demonstration, demo_dir) for demo_dir in demo_dirs]
            
            demonstrations = []
            for future in as_completed(futures):
                try:
                    demo = future.result()
                    if demo:
                        demonstrations.append(demo)
                except Exception as e:
                    logger.error(f"Error loading demonstration: {e}")
    else:
        # Sequential loading
        demonstrations = []
        for demo_dir in demo_dirs:
            try:
                demo = _load_single_demonstration(demo_dir)
                if demo:
                    demonstrations.append(demo)
            except Exception as e:
                logger.error(f"Error loading demonstration from {demo_dir}: {e}")
    
    logger.info(f"Loaded {len(demonstrations)} demonstrations successfully")
    return demonstrations


@cached(maxsize=128, ttl=600)  # Cache for 10 minutes
def _load_single_demonstration(demo_dir: Path) -> Optional[DemonstrationData]:
    """Load a single demonstration from directory."""
    try:
        # Try to load from compressed format first
        data_file = demo_dir / "data.npz"
        if data_file.exists():
            # Load from compressed archive
            data = np.load(data_file)
            
            # Extract observations
            observations = {}
            for key in data.files:
                if key.startswith('obs_'):
                    obs_key = key[4:]  # Remove 'obs_' prefix
                    observations[obs_key] = data[key]
            
            # Load metadata
            metadata_file = demo_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    import json
                    metadata = json.load(f)
            else:
                metadata = {}
            
            return DemonstrationData(
                episode_id=metadata.get("episode_id", demo_dir.name),
                timestamp=metadata.get("timestamp", ""),
                observations=observations,
                actions=data.get('actions', np.array([])),
                rewards=data.get('rewards'),
                success=metadata.get("success", False),
                duration=metadata.get("duration", 0.0),
                metadata=metadata.get("metadata", {})
            )
        
        # Fall back to original format
        metadata_file = demo_dir / "metadata.json"
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            import json
            metadata = json.load(f)
        
        # Load observations
        observations = {}
        for obs_file in demo_dir.glob("*.npy"):
            if obs_file.stem not in ["actions", "rewards"]:
                observations[obs_file.stem] = np.load(obs_file)
        
        # Load actions and rewards
        actions_file = demo_dir / "actions.npy"
        actions = np.load(actions_file) if actions_file.exists() else np.array([])
        
        rewards_file = demo_dir / "rewards.npy"
        rewards = np.load(rewards_file) if rewards_file.exists() else None
        
        return DemonstrationData(
            episode_id=metadata["episode_id"],
            timestamp=metadata["timestamp"],
            observations=observations,
            actions=actions,
            rewards=rewards,
            success=metadata.get("success", False),
            duration=metadata.get("duration", 0.0),
            metadata=metadata.get("metadata", {})
        )
    
    except Exception as e:
        logger.error(f"Failed to load demonstration from {demo_dir}: {e}")
        return None