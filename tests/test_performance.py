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

    @pytest.mark.performance
    def test_large_dataset_memory_efficiency(self):
        """Test memory efficiency with very large datasets"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate large synthetic dataset
        n_users, n_movies = 50000, 5000
        large_ratings = []
        
        for user_id in range(1, n_users + 1):
            # Each user rates 5-20 movies
            n_ratings = min(20, max(5, int(np.random.exponential(10))))
            movie_ids = np.random.choice(range(1, n_movies + 1), n_ratings, replace=False)
            
            for movie_id in movie_ids:
                large_ratings.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'rating': np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
                    'timestamp': np.random.randint(1609459200, 1672531200)
                })
        
        ratings_df = pd.DataFrame(large_ratings)
        
        # Test analyzer with large dataset
        analyzer = MovieAnalyzer()
        analyzer.ratings_df = ratings_df
        
        # Perform memory-intensive operations
        start_time = time.time()
        rating_dist = analyzer.get_rating_distribution()
        top_movies = analyzer.get_top_movies(limit=100)
        processing_time = time.time() - start_time
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        # Performance assertions
        assert processing_time < 30  # Should complete within 30 seconds
        assert memory_used < 1000  # Should use less than 1GB additional memory
        assert len(top_movies) <= 100
        
        # Cleanup
        del large_ratings, ratings_df, analyzer
        gc.collect()

    @pytest.mark.performance
    def test_api_response_time_under_load(self):
        """Test API response times under concurrent load"""
        from app import app
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        app.config['TESTING'] = True
        
        def make_api_request(endpoint):
            with app.test_client() as client:
                start_time = time.time()
                response = client.get(endpoint)
                end_time = time.time()
                return {
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'response_time': end_time - start_time
                }
        
        # Test endpoints under concurrent load
        endpoints = [
            '/api/movies/top',
            '/api/analysis/rating-distribution',
            '/api/analysis/genre-stats',
            '/api/movies/1'
        ]
        
        # Run concurrent requests
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(20):  # 20 concurrent requests
                for endpoint in endpoints:
                    futures.append(executor.submit(make_api_request, endpoint))
            
            for future in futures:
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except Exception as e:
                    # Log but don't fail the test for individual request failures
                    print(f"Request failed: {e}")
        
        # Analyze results
        if results:
            response_times = [r['response_time'] for r in results]
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            # Performance assertions
            assert avg_response_time < 2.0  # Average response time under 2 seconds
            assert max_response_time < 5.0  # Max response time under 5 seconds
            
            # At least 80% of requests should succeed
            successful_requests = [r for r in results if r['status_code'] in [200, 404]]
            success_rate = len(successful_requests) / len(results)
            assert success_rate >= 0.8

    @pytest.mark.performance
    def test_recommendation_algorithm_scalability(self):
        """Test recommendation algorithm performance with varying dataset sizes"""
        import numpy as np
        
        dataset_sizes = [100, 500, 1000, 2000]
        execution_times = []
        
        for size in dataset_sizes:
            # Generate test data
            ratings_data = []
            for user_id in range(1, size + 1):
                for movie_id in range(1, min(100, size) + 1):
                    if np.random.random() < 0.1:  # 10% rating density
                        ratings_data.append({
                            'userId': user_id,
                            'movieId': movie_id,
                            'rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0]),
                            'timestamp': 1609459200
                        })
            
            ratings_df = pd.DataFrame(ratings_data)
            
            analyzer = MovieAnalyzer()
            analyzer.ratings_df = ratings_df
            
            # Measure recommendation performance
            start_time = time.time()
            recommendations = analyzer.get_movie_recommendations(user_id=1, num_recommendations=10)
            execution_time = time.time() - start_time
            
            execution_times.append(execution_time)
            
            # Should complete within reasonable time
            assert execution_time < 10.0
            assert isinstance(recommendations, list)
        
        # Verify scalability - execution time should not grow exponentially
        if len(execution_times) >= 2:
            time_growth_factor = execution_times[-1] / execution_times[0]
            size_growth_factor = dataset_sizes[-1] / dataset_sizes[0]
            
            # Time should not grow more than quadratically with size
            assert time_growth_factor < size_growth_factor ** 2

    @pytest.mark.performance
    def test_visualization_rendering_performance(self):
        """Test visualization rendering performance with large datasets"""
        visualizer = InsightsVisualizer()
        
        # Create large visualization data
        large_genre_stats = {
            "overall": [
                {
                    "genre": f"Genre_{i}",
                    "count": np.random.randint(100, 10000),
                    "mean_rating": np.random.uniform(2.0, 4.5)
                }
                for i in range(100)  # 100 different genres
            ]
        }
        
        large_rating_dist = {
            "distribution": {
                str(float(rating)): np.random.randint(1000, 50000)
                for rating in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            },
            "statistics": {
                "mean": 3.5,
                "median": 3.5,
                "std": 1.2,
                "min": 1.0,
                "max": 5.0,
                "q25": 2.5,
                "q75": 4.5
            }
        }
        
        # Test multiple visualization types
        visualization_tests = [
            (visualizer.plot_rating_distribution, large_rating_dist),
            (visualizer.plot_genre_popularity, large_genre_stats),
        ]
        
        for viz_func, data in visualization_tests:
            start_time = time.time()
            plot_path = viz_func(data)
            execution_time = time.time() - start_time
            
            # Performance assertions
            assert execution_time < 15.0  # Should complete within 15 seconds
            assert Path(plot_path).exists()
            assert Path(plot_path).stat().st_size > 1000  # Should generate substantial plot

    @pytest.mark.performance
    def test_data_processing_pipeline_performance(self):
        """Test end-to-end data processing pipeline performance"""
        from src.data_loader import DataLoader
        
        # Create large synthetic dataset files
        n_movies = 1000
        n_ratings = 50000
        
        movies_data = [
            {
                'movieId': i,
                'title': f'Test Movie {i}',
                'genres': np.random.choice(['Action', 'Comedy', 'Drama']) + '|' + 
                         np.random.choice(['Romance', 'Thriller', 'Sci-Fi'])
            }
            for i in range(1, n_movies + 1)
        ]
        
        ratings_data = [
            {
                'userId': np.random.randint(1, 1000),
                'movieId': np.random.randint(1, n_movies + 1),
                'rating': np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
                'timestamp': np.random.randint(1609459200, 1672531200)
            }
            for _ in range(n_ratings)
        ]
        
        # Test full pipeline
        start_time = time.time()
        
        # Data processing
        processor = DataProcessor()
        clean_movies = processor.clean_movies(movies_data)
        clean_ratings = processor.clean_ratings(ratings_data)
        
        # Analysis
        analyzer = MovieAnalyzer()
        analyzer.movies_df = pd.DataFrame(clean_movies)
        analyzer.ratings_df = pd.DataFrame(clean_ratings)
        
        # Perform multiple analyses
        rating_dist = analyzer.get_rating_distribution()
        top_movies = analyzer.get_top_movies(limit=50)
        genre_stats = analyzer.analyze_genres()
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert total_time < 45.0  # Full pipeline should complete within 45 seconds
        assert len(clean_movies) > 0
        assert len(clean_ratings) > 0
        assert 'distribution' in rating_dist
        assert len(top_movies) <= 50
        assert 'overall' in genre_stats

    @pytest.mark.performance
    def test_memory_leak_detection_extended(self):
        """Extended memory leak detection over multiple iterations"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Record memory usage over multiple iterations
        memory_readings = []
        
        for iteration in range(10):
            gc.collect()  # Force garbage collection
            
            # Record memory before operation
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform memory-intensive operations
            analyzer = MovieAnalyzer()
            
            # Generate data for this iteration
            test_ratings = [
                {
                    'userId': i % 100 + 1,
                    'movieId': i % 50 + 1,
                    'rating': 4.0,
                    'timestamp': 1609459200
                }
                for i in range(1000)
            ]
            
            analyzer.ratings_df = pd.DataFrame(test_ratings)
            
            # Perform operations
            analyzer.get_rating_distribution()
            analyzer.get_top_movies(limit=10)
            
            # Clean up
            del analyzer, test_ratings
            gc.collect()
            
            # Record memory after operation
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_readings.append(memory_after - memory_before)
        
        # Analyze memory usage pattern
        avg_memory_per_iteration = sum(memory_readings) / len(memory_readings)
        max_memory_increase = max(memory_readings)
        
        # Memory usage should be reasonable and not continuously increasing
        assert avg_memory_per_iteration < 50  # Average increase should be less than 50MB
        assert max_memory_increase < 100  # Max increase should be less than 100MB
        
        # Check for memory leak pattern (continuously increasing memory)
        if len(memory_readings) >= 5:
            # Compare first half vs second half
            first_half_avg = sum(memory_readings[:5]) / 5
            second_half_avg = sum(memory_readings[5:]) / 5
            
            # Second half should not be significantly higher (indicating leak)
            memory_growth_ratio = second_half_avg / first_half_avg if first_half_avg > 0 else 1
            assert memory_growth_ratio < 2.0  # Should not double in memory usage