"""Performance tests for MovieLens Analysis Platform"""
import time
from pathlib import Path

import pandas as pd
import pytest

from src.analyzer import MovieAnalyzer
from src.data_processor import DataProcessor
from src.visualizer import InsightsVisualizer


class TestPerformance:
    """Performance tests for the system"""

    @pytest.fixture
    def large_dataset(self, sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame):
        """Create a larger dataset for performance testing"""
        # Scale up the sample data
        scale_factor = 1000
        
        large_movies = pd.concat([sample_movies_data] * scale_factor, ignore_index=True)
        large_ratings = pd.concat([sample_ratings_data] * scale_factor, ignore_index=True)
        
        # Adjust IDs to maintain relationships
        large_movies['movieId'] = range(len(large_movies))
        large_ratings['movieId'] = large_ratings['movieId'] % len(large_movies)
        large_ratings['userId'] = large_ratings['userId'] + (large_ratings.index // len(sample_ratings_data)) * sample_ratings_data['userId'].max()
        
        return large_movies, large_ratings

    @pytest.mark.performance
    def test_data_processing_performance(self, large_dataset):
        """Test data processing performance with large dataset"""
        large_movies, large_ratings = large_dataset
        
        processor = DataProcessor()
        
        start_time = time.time()
        processed_data = processor.process_data(large_movies, large_ratings)
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 60  # Should complete within 60 seconds
        assert len(processed_data['movies_clean']) > 0
        assert len(processed_data['ratings_clean']) > 0
        
        print(f"Data processing time: {processing_time:.2f} seconds")
        print(f"Movies processed: {len(processed_data['movies_clean'])}")
        print(f"Ratings processed: {len(processed_data['ratings_clean'])}")

    @pytest.mark.performance
    def test_analysis_performance(self, large_dataset):
        """Test analysis performance with large dataset"""
        large_movies, large_ratings = large_dataset
        
        # Use smaller subset for analysis to keep test reasonable
        movies_subset = large_movies.head(1000)
        ratings_subset = large_ratings.head(10000)
        
        analyzer = MovieAnalyzer(movies_subset, ratings_subset)
        
        # Test various analysis operations
        operations = {
            'top_movies': lambda: analyzer.get_top_rated_movies(n=10),
            'popular_movies': lambda: analyzer.get_most_popular_movies(n=10),
            'genre_analysis': lambda: analyzer.analyze_genres(),
            'statistics': lambda: analyzer.get_statistics_summary(),
        }
        
        performance_results = {}
        
        for operation_name, operation in operations.items():
            start_time = time.time()
            result = operation()
            operation_time = time.time() - start_time
            
            performance_results[operation_name] = operation_time
            
            # Each operation should complete within reasonable time
            assert operation_time < 10  # 10 seconds max per operation
            assert result is not None
            
            print(f"{operation_name} time: {operation_time:.2f} seconds")
        
        # Total analysis time should be reasonable
        total_time = sum(performance_results.values())
        assert total_time < 30  # Total should be under 30 seconds

    @pytest.mark.performance
    def test_visualization_performance(self, temp_output_dir: Path, sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame):
        """Test visualization performance"""
        analyzer = MovieAnalyzer(sample_movies_data, sample_ratings_data)
        visualizer = InsightsVisualizer(temp_output_dir)
        
        # Generate data for visualizations
        top_movies = analyzer.get_top_rated_movies(n=10)
        genre_stats = analyzer.analyze_genres()
        
        visualization_operations = {
            'rating_distribution': lambda: visualizer.plot_rating_distribution(sample_ratings_data),
            'top_movies_plot': lambda: visualizer.plot_top_movies(top_movies),
            'genre_popularity': lambda: visualizer.plot_genre_popularity(genre_stats),
        }
        
        for viz_name, viz_operation in visualization_operations.items():
            start_time = time.time()
            plot_path = viz_operation()
            viz_time = time.time() - start_time
            
            # Each visualization should complete quickly
            assert viz_time < 5  # 5 seconds max per visualization
            assert plot_path.exists()
            
            print(f"{viz_name} time: {viz_time:.2f} seconds")

    @pytest.mark.performance
    def test_memory_usage(self, large_dataset):
        """Test memory usage with large dataset"""
        import psutil
        import os
        
        large_movies, large_ratings = large_dataset
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process data
        processor = DataProcessor()
        processed_data = processor.process_data(large_movies, large_ratings)
        
        # Get memory usage after processing
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Peak memory: {peak_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # Memory usage should be reasonable (adjust threshold as needed)
        assert memory_increase < 1000  # Less than 1GB increase

    @pytest.mark.performance
    def test_concurrent_operations(self, sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame):
        """Test performance of concurrent operations"""
        import threading
        import queue
        
        analyzer = MovieAnalyzer(sample_movies_data, sample_ratings_data)
        results_queue = queue.Queue()
        
        def run_analysis(operation_name, operation):
            start_time = time.time()
            try:
                result = operation()
                end_time = time.time()
                results_queue.put((operation_name, end_time - start_time, True, result))
            except Exception as e:
                end_time = time.time()
                results_queue.put((operation_name, end_time - start_time, False, str(e)))
        
        # Define operations to run concurrently
        operations = [
            ('top_movies', lambda: analyzer.get_top_rated_movies(n=5)),
            ('popular_movies', lambda: analyzer.get_most_popular_movies(n=5)),
            ('genre_analysis', lambda: analyzer.analyze_genres()),
            ('statistics', lambda: analyzer.get_statistics_summary()),
        ]
        
        # Start all operations concurrently
        threads = []
        start_time = time.time()
        
        for operation_name, operation in operations:
            thread = threading.Thread(target=run_analysis, args=(operation_name, operation))
            thread.start()
            threads.append(thread)
        
        # Wait for all operations to complete
        for thread in threads:
            thread.join()
        
        total_concurrent_time = time.time() - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # All operations should complete successfully
        assert len(results) == len(operations)
        
        for operation_name, operation_time, success, result in results:
            assert success, f"Operation {operation_name} failed: {result}"
            assert operation_time < 10  # Each operation should be fast
            print(f"Concurrent {operation_name} time: {operation_time:.2f} seconds")
        
        print(f"Total concurrent execution time: {total_concurrent_time:.2f} seconds")
        
        # Concurrent execution should be faster than sequential
        assert total_concurrent_time < 20  # Should complete within 20 seconds

    @pytest.mark.performance
    def test_cache_performance(self, sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame):
        """Test caching performance improvements"""
        analyzer = MovieAnalyzer(sample_movies_data, sample_ratings_data)
        
        # First run (no cache)
        start_time = time.time()
        result1 = analyzer.get_top_rated_movies(n=10)
        first_run_time = time.time() - start_time
        
        # Second run (should use cache if implemented)
        start_time = time.time()
        result2 = analyzer.get_top_rated_movies(n=10)
        second_run_time = time.time() - start_time
        
        # Results should be identical
        assert len(result1) == len(result2)
        
        # Second run should be faster (if caching is implemented)
        # If no caching, times will be similar
        print(f"First run time: {first_run_time:.4f} seconds")
        print(f"Second run time: {second_run_time:.4f} seconds")
        
        if second_run_time < first_run_time * 0.5:
            print("Caching appears to be working effectively")

    @pytest.mark.performance
    def test_scalability_metrics(self, sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame):
        """Test scalability with different dataset sizes"""
        processor = DataProcessor()
        
        scale_factors = [1, 10, 100]
        processing_times = []
        
        for scale_factor in scale_factors:
            # Create scaled dataset
            scaled_movies = pd.concat([sample_movies_data] * scale_factor, ignore_index=True)
            scaled_ratings = pd.concat([sample_ratings_data] * scale_factor, ignore_index=True)
            
            # Adjust IDs
            scaled_movies['movieId'] = range(len(scaled_movies))
            scaled_ratings['movieId'] = scaled_ratings['movieId'] % len(scaled_movies)
            
            # Measure processing time
            start_time = time.time()
            processed_data = processor.process_data(scaled_movies, scaled_ratings)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            
            print(f"Scale factor {scale_factor}: {processing_time:.2f} seconds")
            print(f"  Movies: {len(scaled_movies)}, Ratings: {len(scaled_ratings)}")
        
        # Processing time should scale reasonably (not exponentially)
        # This is a basic check - adjust thresholds based on expected performance
        for i in range(1, len(processing_times)):
            scale_ratio = scale_factors[i] / scale_factors[i-1]
            time_ratio = processing_times[i] / processing_times[i-1]
            
            # Time increase should not be more than 2x the scale increase
            assert time_ratio <= scale_ratio * 2, f"Poor scalability: {time_ratio} vs {scale_ratio}"

    @pytest.mark.performance
    @pytest.mark.slow
    def test_stress_test(self, sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame):
        """Stress test with intensive operations"""
        # Create a moderately large dataset
        stress_movies = pd.concat([sample_movies_data] * 500, ignore_index=True)
        stress_ratings = pd.concat([sample_ratings_data] * 500, ignore_index=True)
        
        # Adjust IDs
        stress_movies['movieId'] = range(len(stress_movies))
        stress_ratings['movieId'] = stress_ratings['movieId'] % len(stress_movies)
        
        analyzer = MovieAnalyzer(stress_movies, stress_ratings)
        
        # Run multiple intensive operations
        operations_count = 10
        start_time = time.time()
        
        for i in range(operations_count):
            # Vary the operations to stress different parts of the system
            if i % 4 == 0:
                result = analyzer.get_top_rated_movies(n=20)
            elif i % 4 == 1:
                result = analyzer.get_most_popular_movies(n=20)
            elif i % 4 == 2:
                result = analyzer.analyze_genres()
            else:
                result = analyzer.get_statistics_summary()
            
            assert result is not None
        
        total_stress_time = time.time() - start_time
        
        print(f"Stress test completed: {operations_count} operations in {total_stress_time:.2f} seconds")
        print(f"Average time per operation: {total_stress_time/operations_count:.2f} seconds")
        
        # Should handle stress test within reasonable time
        assert total_stress_time < 120  # 2 minutes max for stress test
        assert total_stress_time / operations_count < 15  # 15 seconds max per operation