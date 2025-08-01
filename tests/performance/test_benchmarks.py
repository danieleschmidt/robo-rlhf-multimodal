"""Performance benchmarks and tests."""

import pytest
import time
from typing import Dict, Any

import numpy as np
import torch


@pytest.mark.benchmark
class TestModelPerformance:
    """Performance benchmarks for model components."""
    
    @pytest.mark.unit
    def test_vision_encoder_throughput(self, device, performance_benchmark):
        """Benchmark vision encoder throughput."""
        # Mock vision encoder (simple CNN)
        encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 512),
        ).to(device)
        
        batch_sizes = [1, 4, 8, 16, 32]
        throughput_results = {}
        
        for batch_size in batch_sizes:
            # Warm up
            for _ in range(10):
                dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
                _ = encoder(dummy_input)
            
            # Benchmark
            num_iterations = 100
            performance_benchmark.start()
            
            for _ in range(num_iterations):
                input_batch = torch.randn(batch_size, 3, 224, 224, device=device)
                with torch.no_grad():
                    _ = encoder(input_batch)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
            
            elapsed_time = performance_benchmark.stop()
            
            # Calculate throughput (images per second)
            total_images = num_iterations * batch_size
            throughput = total_images / elapsed_time
            throughput_results[batch_size] = throughput
            
            print(f"Batch size {batch_size}: {throughput:.1f} images/sec")
        
        # Verify reasonable performance
        assert throughput_results[1] > 50  # At least 50 fps for single image
        assert throughput_results[32] > throughput_results[1]  # Batching should improve throughput
    
    @pytest.mark.unit
    def test_policy_inference_latency(self, device, performance_benchmark):
        """Benchmark policy inference latency."""
        # Mock policy network
        policy = torch.nn.Sequential(
            torch.nn.Linear(512 + 7, 256),  # vision + proprioception
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 7),  # action output
            torch.nn.Tanh(),
        ).to(device)
        
        policy.eval()
        
        # Warm up
        for _ in range(50):
            dummy_input = torch.randn(1, 519, device=device)
            with torch.no_grad():
                _ = policy(dummy_input)
        
        # Benchmark single inference latency
        latencies = []
        num_inferences = 1000
        
        for _ in range(num_inferences):
            input_data = torch.randn(1, 519, device=device)
            
            performance_benchmark.start()
            with torch.no_grad():
                output = policy(input_data)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            latency = performance_benchmark.stop()
            latencies.append(latency * 1000)  # Convert to milliseconds
        
        # Calculate statistics
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Mean latency: {mean_latency:.2f} ms")
        print(f"P95 latency: {p95_latency:.2f} ms")
        print(f"P99 latency: {p99_latency:.2f} ms")
        
        # Verify real-time performance requirements
        assert mean_latency < 50  # Less than 50ms mean latency
        assert p95_latency < 100   # Less than 100ms P95 latency
        
        # Verify output correctness
        assert output.shape == (1, 7)
        assert torch.all(torch.abs(output) <= 1.0)  # Tanh output range
    
    @pytest.mark.slow
    def test_training_step_performance(self, device, performance_benchmark):
        """Benchmark training step performance."""
        # Mock model for training
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        batch_sizes = [16, 32, 64, 128]
        training_times = {}
        
        for batch_size in batch_sizes:
            # Warm up
            for _ in range(10):
                dummy_input = torch.randn(batch_size, 512, device=device)
                dummy_target = torch.randn(batch_size, 1, device=device)
                
                optimizer.zero_grad()
                output = model(dummy_input)
                loss = criterion(output, dummy_target)
                loss.backward()
                optimizer.step()
            
            # Benchmark training steps
            num_steps = 100
            performance_benchmark.start()
            
            for _ in range(num_steps):
                batch_input = torch.randn(batch_size, 512, device=device)
                batch_target = torch.randn(batch_size, 1, device=device)
                
                optimizer.zero_grad()
                output = model(batch_input)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
            
            elapsed_time = performance_benchmark.stop()
            
            # Calculate steps per second
            steps_per_sec = num_steps / elapsed_time
            training_times[batch_size] = steps_per_sec
            
            print(f"Batch size {batch_size}: {steps_per_sec:.1f} steps/sec")
        
        # Verify reasonable training performance
        assert training_times[16] > 100  # At least 100 steps/sec for small batches


@pytest.mark.benchmark
class TestDataProcessingPerformance:
    """Performance benchmarks for data processing."""
    
    @pytest.mark.unit
    def test_image_preprocessing_speed(self, performance_benchmark):
        """Benchmark image preprocessing pipeline."""
        import cv2
        
        # Generate test images
        num_images = 1000
        test_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(num_images)
        ]
        
        # Benchmark preprocessing pipeline
        performance_benchmark.start()
        
        processed_images = []
        for img in test_images:
            # Typical preprocessing steps
            img_float = img.astype(np.float32) / 255.0
            img_resized = cv2.resize(img_float, (224, 224))
            img_normalized = (img_resized - 0.5) / 0.5  # Normalize to [-1, 1]
            processed_images.append(img_normalized)
        
        elapsed_time = performance_benchmark.stop()
        
        # Calculate processing rate
        images_per_sec = num_images / elapsed_time
        print(f"Image preprocessing: {images_per_sec:.1f} images/sec")
        
        # Verify reasonable performance
        assert images_per_sec > 500  # At least 500 images/sec
        assert len(processed_images) == num_images
    
    @pytest.mark.unit
    def test_trajectory_loading_speed(self, test_data_dir, performance_benchmark):
        """Benchmark trajectory data loading."""
        from tests.fixtures.data_fixtures import create_sample_demonstration_data
        
        # Create large dataset
        demo_dir = create_sample_demonstration_data(test_data_dir, num_episodes=100)
        
        # Benchmark loading
        performance_benchmark.start()
        
        loaded_episodes = []
        import pickle
        
        for demo_file in demo_dir.glob("*.pkl"):
            with open(demo_file, "rb") as f:
                episode_data = pickle.load(f)
                loaded_episodes.append(episode_data)
        
        elapsed_time = performance_benchmark.stop()
        
        # Calculate loading rate
        episodes_per_sec = len(loaded_episodes) / elapsed_time
        print(f"Episode loading: {episodes_per_sec:.1f} episodes/sec")
        
        # Verify reasonable performance
        assert episodes_per_sec > 50  # At least 50 episodes/sec
        assert len(loaded_episodes) == 100


@pytest.mark.benchmark 
class TestMemoryUsage:
    """Memory usage benchmarks."""
    
    @pytest.mark.unit
    def test_model_memory_usage(self, device):
        """Benchmark model memory usage."""
        if device.type != "cuda":
            pytest.skip("GPU memory test requires CUDA")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Create model
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 7),
        ).to(device)
        
        model_memory = torch.cuda.memory_allocated(device) - initial_memory
        
        # Create batch for forward pass
        batch_input = torch.randn(32, 512, device=device)
        output = model(batch_input)
        
        forward_memory = torch.cuda.memory_allocated(device) - initial_memory
        
        # Create gradients
        loss = torch.sum(output)
        loss.backward()
        
        backward_memory = torch.cuda.memory_allocated(device) - initial_memory
        
        print(f"Model parameters: {model_memory / 1024**2:.2f} MB")
        print(f"Forward pass: {forward_memory / 1024**2:.2f} MB")  
        print(f"Backward pass: {backward_memory / 1024**2:.2f} MB")
        
        # Verify reasonable memory usage
        assert model_memory < 50 * 1024**2  # Less than 50MB for model
        assert backward_memory < 200 * 1024**2  # Less than 200MB total
    
    @pytest.mark.unit
    def test_batch_size_memory_scaling(self, device):
        """Test memory scaling with batch size."""
        if device.type != "cuda":
            pytest.skip("GPU memory test requires CUDA")
        
        model = torch.nn.Linear(512, 7).to(device)
        batch_sizes = [1, 8, 16, 32, 64]
        memory_usage = {}
        
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)
            
            batch_input = torch.randn(batch_size, 512, device=device)
            output = model(batch_input)
            loss = torch.sum(output)
            loss.backward()
            
            peak_memory = torch.cuda.memory_allocated(device) - initial_memory
            memory_usage[batch_size] = peak_memory
            
            print(f"Batch size {batch_size}: {peak_memory / 1024**2:.2f} MB")
        
        # Memory should scale roughly linearly with batch size
        assert memory_usage[64] > memory_usage[1]
        assert memory_usage[32] > memory_usage[16]


@pytest.mark.benchmark
class TestScalabilityTests:
    """Scalability tests for large-scale scenarios."""
    
    @pytest.mark.slow
    def test_large_dataset_handling(self, test_data_dir, performance_benchmark):
        """Test handling of large datasets."""
        from tests.fixtures.data_fixtures import create_mock_dataset, MockDataLoader
        
        # Create large mock dataset
        large_dataset = create_mock_dataset(num_samples=10000)
        data_loader = MockDataLoader(large_dataset, batch_size=32)
        
        # Benchmark dataset iteration
        performance_benchmark.start()
        
        total_samples = 0
        for batch in data_loader:
            total_samples += len(batch)
            
            # Simulate some processing
            for sample in batch:
                _ = sample["observation"]["rgb"].mean()
        
        elapsed_time = performance_benchmark.stop()
        
        # Calculate processing rate
        samples_per_sec = total_samples / elapsed_time
        print(f"Dataset processing: {samples_per_sec:.1f} samples/sec")
        
        # Verify performance and correctness
        assert total_samples == 10000
        assert samples_per_sec > 1000  # At least 1000 samples/sec
    
    @pytest.mark.slow
    def test_concurrent_inference(self, device, performance_benchmark):
        """Test concurrent inference scenarios."""
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        # Mock model for inference
        model = torch.nn.Linear(512, 7).to(device)
        model.eval()
        
        def inference_worker(worker_id: int, num_inferences: int) -> Dict[str, Any]:
            """Worker function for concurrent inference."""
            latencies = []
            
            for _ in range(num_inferences):
                input_data = torch.randn(1, 512, device=device)
                
                start_time = time.perf_counter()
                with torch.no_grad():
                    output = model(input_data)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                
                latency = time.perf_counter() - start_time
                latencies.append(latency * 1000)  # Convert to ms
            
            return {
                "worker_id": worker_id,
                "mean_latency": np.mean(latencies),
                "num_inferences": num_inferences,
            }
        
        # Test concurrent inference with multiple threads
        num_workers = 4
        inferences_per_worker = 100
        
        performance_benchmark.start()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(inference_worker, i, inferences_per_worker)
                for i in range(num_workers)
            ]
            
            results = [future.result() for future in futures]
        
        elapsed_time = performance_benchmark.stop()
        
        # Calculate overall throughput
        total_inferences = sum(r["num_inferences"] for r in results)
        overall_throughput = total_inferences / elapsed_time
        
        print(f"Concurrent inference: {overall_throughput:.1f} inferences/sec")
        print(f"Average latency: {np.mean([r['mean_latency'] for r in results]):.2f} ms")
        
        # Verify concurrent performance
        assert len(results) == num_workers
        assert total_inferences == num_workers * inferences_per_worker
        assert overall_throughput > 100  # At least 100 inferences/sec total