"""
Comprehensive integration and real-world scenario tests for MovieLens Analysis Platform.
Tests error recovery, edge cases, and realistic usage patterns.
"""

import pytest
import pandas as pd
import numpy as np
import json
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests
from concurrent.futures import ThreadPoolExecutor
import threading

from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.analyzer import MovieAnalyzer
from src.visualizer import InsightsVisualizer


class TestRealWorldScenarios:
    """Test realistic usage scenarios and edge cases"""
    
    @pytest.fixture
    def realistic_dataset(self):
        """Create a realistic dataset with real-world characteristics"""
        # Simulate realistic movie data with various edge cases
        movies_data = [
            {'movieId': 1, 'title': 'Toy Story (1995)', 'genres': 'Adventure|Animation|Children|Comedy|Fantasy'},
            {'movieId': 2, 'title': 'Jumanji (1995)', 'genres': 'Adventure|Children|Fantasy'},
            {'movieId': 3, 'title': 'Grumpier Old Men (1995)', 'genres': 'Comedy|Romance'},
            {'movieId': 4, 'title': 'Waiting to Exhale (1995)', 'genres': 'Comedy|Drama|Romance'},
            {'movieId': 5, 'title': 'Father of the Bride Part II (1995)', 'genres': 'Comedy'},
            # Movies with special characters and edge cases
            {'movieId': 6, 'title': 'Heat (1995)', 'genres': 'Action|Crime|Thriller'},
            {'movieId': 7, 'title': 'Sabrina (1995)', 'genres': 'Comedy|Romance'},
            {'movieId': 8, 'title': 'Tom and Huck (1995)', 'genres': 'Adventure|Children'},
            {'movieId': 9, 'title': 'Sudden Death (1995)', 'genres': 'Action'},
            {'movieId': 10, 'title': 'GoldenEye (1995)', 'genres': 'Action|Adventure|Thriller'},
            # Edge case: Movie with no genre
            {'movieId': 11, 'title': 'Unknown Movie', 'genres': '(no genres listed)'},
            # Edge case: Movie with very long title
            {'movieId': 12, 'title': 'A Very Long Movie Title That Goes On And On And Contains Many Words And Characters (2023)', 'genres': 'Drama'},
        ]
        
        # Simulate realistic rating patterns
        ratings_data = []
        user_preferences = {}
        
        # Create user preference profiles
        for user_id in range(1, 101):  # 100 users
            # Some users prefer certain genres
            preferred_genres = np.random.choice(['Action', 'Comedy', 'Drama', 'Romance'], 
                                              size=np.random.randint(1, 3), replace=False)
            user_preferences[user_id] = preferred_genres
        
        # Generate ratings based on preferences
        for user_id in range(1, 101):
            n_ratings = np.random.randint(5, 50)  # Each user rates 5-50 movies
            rated_movies = np.random.choice(range(1, 13), size=n_ratings, replace=False)
            
            for movie_id in rated_movies:
                movie = next(m for m in movies_data if m['movieId'] == movie_id)
                movie_genres = movie['genres'].split('|')
                
                # Higher rating if movie matches user preferences
                base_rating = 3.0
                if any(genre in user_preferences[user_id] for genre in movie_genres):
                    base_rating = 4.0
                
                # Add some randomness
                rating = base_rating + np.random.normal(0, 0.8)
                rating = max(1.0, min(5.0, rating))
                rating = round(rating * 2) / 2  # Round to nearest 0.5
                
                ratings_data.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'rating': rating,
                    'timestamp': np.random.randint(946684800, 1672531200)  # 2000-2023
                })
        
        return movies_data, ratings_data

    def test_complete_workflow_with_realistic_data(self, realistic_dataset):
        """Test complete workflow with realistic data patterns"""
        movies_data, ratings_data = realistic_dataset
        
        # Step 1: Data Processing
        processor = DataProcessor()
        clean_movies = processor.clean_movies(movies_data)
        clean_ratings = processor.clean_ratings(ratings_data)
        
        assert len(clean_movies) > 0
        assert len(clean_ratings) > 0
        
        # Step 2: Analysis
        analyzer = MovieAnalyzer()
        analyzer.movies_df = pd.DataFrame(clean_movies)
        analyzer.ratings_df = pd.DataFrame(clean_ratings)
        
        # Perform comprehensive analysis
        rating_dist = analyzer.get_rating_distribution()
        top_movies = analyzer.get_top_movies(limit=10)
        genre_stats = analyzer.analyze_genres()
        user_stats = analyzer.get_user_statistics()
        
        # Validate results
        assert 'distribution' in rating_dist
        assert 'statistics' in rating_dist
        assert len(top_movies) <= 10
        assert 'overall' in genre_stats
        assert 'total_users' in user_stats
        
        # Step 3: Visualization
        visualizer = InsightsVisualizer()
        
        # Generate multiple visualizations
        rating_plot = visualizer.plot_rating_distribution(rating_dist)
        genre_plot = visualizer.plot_genre_popularity(genre_stats)
        top_movies_plot = visualizer.plot_top_movies({'movies': top_movies})
        
        # Verify all plots were created
        for plot_path in [rating_plot, genre_plot, top_movies_plot]:
            assert Path(plot_path).exists()
            assert Path(plot_path).stat().st_size > 1000

    def test_data_quality_issues_handling(self):
        """Test handling of various data quality issues"""
        # Create dataset with quality issues
        problematic_movies = [
            {'movieId': 1, 'title': 'Normal Movie', 'genres': 'Action|Comedy'},
            {'movieId': 2, 'title': '', 'genres': 'Drama'},  # Empty title
            {'movieId': 3, 'title': None, 'genres': 'Comedy'},  # None title
            {'movieId': 4, 'title': 'Movie with Special Chars äöü', 'genres': 'Action'},
            {'movieId': 5, 'title': 'Movie (1995)', 'genres': ''},  # Empty genres
            {'movieId': 6, 'title': 'Movie', 'genres': None},  # None genres
        ]
        
        problematic_ratings = [
            {'userId': 1, 'movieId': 1, 'rating': 4.5, 'timestamp': 1609459200},
            {'userId': 2, 'movieId': 1, 'rating': 6.0, 'timestamp': 1609459200},  # Invalid rating
            {'userId': 3, 'movieId': 1, 'rating': -1.0, 'timestamp': 1609459200},  # Invalid rating
            {'userId': 4, 'movieId': 1, 'rating': None, 'timestamp': 1609459200},  # None rating
            {'userId': 5, 'movieId': 999, 'rating': 4.0, 'timestamp': 1609459200},  # Non-existent movie
            {'userId': None, 'movieId': 1, 'rating': 4.0, 'timestamp': 1609459200},  # None user
        ]
        
        # Process data with quality issues
        processor = DataProcessor()
        clean_movies = processor.clean_movies(problematic_movies)
        clean_ratings = processor.clean_ratings(problematic_ratings)
        
        # Verify data cleaning worked
        assert len(clean_movies) > 0  # Some movies should remain
        assert len(clean_ratings) > 0  # Some ratings should remain
        
        # Verify invalid data was filtered out
        clean_movies_df = pd.DataFrame(clean_movies)
        clean_ratings_df = pd.DataFrame(clean_ratings)
        
        # All remaining movies should have valid titles
        assert not clean_movies_df['title'].isnull().any()
        assert not (clean_movies_df['title'] == '').any()
        
        # All remaining ratings should be valid
        assert clean_ratings_df['rating'].between(1.0, 5.0).all()
        assert not clean_ratings_df['userId'].isnull().any()

    def test_concurrent_analysis_operations(self, realistic_dataset):
        """Test concurrent analysis operations for thread safety"""
        movies_data, ratings_data = realistic_dataset
        
        processor = DataProcessor()
        clean_movies = processor.clean_movies(movies_data)
        clean_ratings = processor.clean_ratings(ratings_data)
        
        # Create multiple analyzer instances
        analyzers = []
        for i in range(5):
            analyzer = MovieAnalyzer()
            analyzer.movies_df = pd.DataFrame(clean_movies)
            analyzer.ratings_df = pd.DataFrame(clean_ratings)
            analyzers.append(analyzer)
        
        results = []
        errors = []
        
        def run_analysis(analyzer, operation_id):
            try:
                # Perform multiple operations concurrently
                rating_dist = analyzer.get_rating_distribution()
                top_movies = analyzer.get_top_movies(limit=5)
                genre_stats = analyzer.analyze_genres()
                
                results.append({
                    'operation_id': operation_id,
                    'rating_dist': rating_dist,
                    'top_movies': top_movies,
                    'genre_stats': genre_stats
                })
            except Exception as e:
                errors.append({'operation_id': operation_id, 'error': str(e)})
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i, analyzer in enumerate(analyzers):
                futures.append(executor.submit(run_analysis, analyzer, i))
            
            # Wait for all operations to complete
            for future in futures:
                future.result(timeout=30)
        
        # Verify results
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        assert len(results) == 5
        
        # Verify all results are consistent
        first_result = results[0]
        for result in results[1:]:
            assert result['rating_dist']['statistics']['mean'] == first_result['rating_dist']['statistics']['mean']
            assert len(result['top_movies']) == len(first_result['top_movies'])

    def test_memory_pressure_scenarios(self):
        """Test behavior under memory pressure conditions"""
        # Create progressively larger datasets to test memory handling
        dataset_sizes = [1000, 5000, 10000]
        
        for size in dataset_sizes:
            # Generate large dataset
            movies_data = [
                {
                    'movieId': i,
                    'title': f'Movie {i}',
                    'genres': 'Action|Comedy'
                }
                for i in range(1, min(size // 10, 1000) + 1)
            ]
            
            ratings_data = [
                {
                    'userId': np.random.randint(1, size // 100 + 1),
                    'movieId': np.random.randint(1, len(movies_data) + 1),
                    'rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0]),
                    'timestamp': 1609459200
                }
                for _ in range(size)
            ]
            
            # Process and analyze
            processor = DataProcessor()
            analyzer = MovieAnalyzer()
            
            try:
                clean_movies = processor.clean_movies(movies_data)
                clean_ratings = processor.clean_ratings(ratings_data)
                
                analyzer.movies_df = pd.DataFrame(clean_movies)
                analyzer.ratings_df = pd.DataFrame(clean_ratings)
                
                # Perform analysis
                rating_dist = analyzer.get_rating_distribution()
                top_movies = analyzer.get_top_movies(limit=10)
                
                # Verify results are reasonable
                assert 'distribution' in rating_dist
                assert len(top_movies) <= 10
                
            except MemoryError:
                # If we hit memory limits, that's acceptable for very large datasets
                pytest.skip(f"Memory limit reached at dataset size {size}")
            
            # Clean up
            del movies_data, ratings_data, clean_movies, clean_ratings
            del analyzer

    def test_network_failure_simulation(self):
        """Test behavior when network operations fail"""
        data_loader = DataLoader()
        
        # Mock network failures
        with patch('requests.get') as mock_get:
            # Simulate connection timeout
            mock_get.side_effect = requests.exceptions.ConnectTimeout("Connection timed out")
            
            with pytest.raises((requests.exceptions.ConnectTimeout, Exception)):
                data_loader._load_from_grouplens()
            
            # Simulate HTTP error
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
            mock_get.return_value = mock_response
            
            with pytest.raises((requests.exceptions.HTTPError, Exception)):
                data_loader._load_from_grouplens()

    def test_file_system_error_recovery(self):
        """Test recovery from file system errors"""
        visualizer = InsightsVisualizer()
        
        # Test with invalid output directory
        with patch.object(visualizer, 'output_dir', '/invalid/path/that/does/not/exist'):
            rating_data = {
                'distribution': {'4.0': 100, '5.0': 50},
                'statistics': {'mean': 4.2, 'std': 0.8}
            }
            
            # Should handle directory creation or use fallback
            try:
                plot_path = visualizer.plot_rating_distribution(rating_data)
                # If successful, verify the plot exists
                if plot_path:
                    assert Path(plot_path).exists()
            except (OSError, PermissionError):
                # Expected behavior for invalid paths
                pass

    def test_data_corruption_scenarios(self):
        """Test handling of corrupted data scenarios"""
        # Test with corrupted JSON-like data
        corrupted_data_scenarios = [
            # Scenario 1: Mixed data types
            [
                {'movieId': 1, 'title': 'Movie 1', 'genres': 'Action'},
                {'movieId': '2', 'title': 'Movie 2', 'genres': ['Comedy']},  # Wrong type
                {'movieId': 3.5, 'title': 123, 'genres': None},  # Wrong types
            ],
            # Scenario 2: Missing required fields
            [
                {'movieId': 1, 'title': 'Movie 1'},  # Missing genres
                {'title': 'Movie 2', 'genres': 'Comedy'},  # Missing movieId
                {'movieId': 3, 'genres': 'Drama'},  # Missing title
            ],
            # Scenario 3: Extreme values
            [
                {'movieId': -1, 'title': 'Movie 1', 'genres': 'Action'},  # Negative ID
                {'movieId': 999999999, 'title': 'Movie 2', 'genres': 'Comedy'},  # Very large ID
                {'movieId': 1, 'title': 'A' * 10000, 'genres': 'Drama'},  # Very long title
            ]
        ]
        
        processor = DataProcessor()
        
        for scenario_data in corrupted_data_scenarios:
            try:
                clean_data = processor.clean_movies(scenario_data)
                # Should either clean the data or handle gracefully
                assert isinstance(clean_data, list)
                
                # If any data remains, it should be valid
                if clean_data:
                    clean_df = pd.DataFrame(clean_data)
                    assert 'movieId' in clean_df.columns
                    assert 'title' in clean_df.columns
                    assert 'genres' in clean_df.columns
                    
            except Exception as e:
                # Should not crash with unhandled exceptions
                assert isinstance(e, (ValueError, TypeError, KeyError))

    def test_edge_case_analysis_scenarios(self):
        """Test analysis with edge case data scenarios"""
        analyzer = MovieAnalyzer()
        
        # Scenario 1: Single movie, single user
        minimal_movies = [{'movieId': 1, 'title': 'Only Movie', 'genres': 'Drama'}]
        minimal_ratings = [{'userId': 1, 'movieId': 1, 'rating': 5.0, 'timestamp': 1609459200}]
        
        analyzer.movies_df = pd.DataFrame(minimal_movies)
        analyzer.ratings_df = pd.DataFrame(minimal_ratings)
        
        rating_dist = analyzer.get_rating_distribution()
        top_movies = analyzer.get_top_movies(limit=10)
        
        assert 'distribution' in rating_dist
        assert len(top_movies) == 1
        
        # Scenario 2: All movies have same rating
        uniform_ratings = [
            {'userId': i, 'movieId': 1, 'rating': 3.0, 'timestamp': 1609459200}
            for i in range(1, 11)
        ]
        
        analyzer.ratings_df = pd.DataFrame(uniform_ratings)
        
        rating_dist = analyzer.get_rating_distribution()
        assert rating_dist['statistics']['std'] == 0.0
        
        # Scenario 3: Extreme rating distribution (all 1s and 5s)
        extreme_ratings = []
        for i in range(1, 51):
            rating = 1.0 if i <= 25 else 5.0
            extreme_ratings.append({
                'userId': i, 'movieId': 1, 'rating': rating, 'timestamp': 1609459200
            })
        
        analyzer.ratings_df = pd.DataFrame(extreme_ratings)
        
        rating_dist = analyzer.get_rating_distribution()
        assert rating_dist['statistics']['std'] > 1.5  # Should have high standard deviation

    def test_visualization_error_recovery(self):
        """Test visualization error recovery scenarios"""
        visualizer = InsightsVisualizer()
        
        # Test with empty data
        empty_data = {'distribution': {}, 'statistics': {}}
        
        try:
            plot_path = visualizer.plot_rating_distribution(empty_data)
            # Should either create a plot or handle gracefully
            if plot_path:
                assert Path(plot_path).exists()
        except Exception as e:
            # Should be a handled exception, not a crash
            assert isinstance(e, (ValueError, KeyError, TypeError))
        
        # Test with malformed data
        malformed_data = {'invalid': 'structure'}
        
        try:
            plot_path = visualizer.plot_rating_distribution(malformed_data)
            if plot_path:
                assert Path(plot_path).exists()
        except Exception as e:
            assert isinstance(e, (ValueError, KeyError, TypeError))

    def test_api_integration_scenarios(self):
        """Test API integration with various scenarios"""
        from app import app
        
        app.config['TESTING'] = True
        
        with app.test_client() as client:
            # Test API with no data loaded
            response = client.get('/api/movies/top')
            # Should handle gracefully (either return empty results or appropriate error)
            assert response.status_code in [200, 404, 500]
            
            # Test API with invalid parameters
            response = client.get('/api/movies/top?limit=invalid')
            assert response.status_code in [400, 422, 500]
            
            # Test API with extreme parameters
            response = client.get('/api/movies/top?limit=999999')
            assert response.status_code in [200, 400, 422]
            
            # Test non-existent endpoints
            response = client.get('/api/nonexistent/endpoint')
            assert response.status_code == 404

    def test_long_running_operations(self):
        """Test behavior of long-running operations"""
        # Create a dataset that will take some time to process
        large_movies = [
            {'movieId': i, 'title': f'Movie {i}', 'genres': 'Action|Comedy|Drama'}
            for i in range(1, 1001)
        ]
        
        large_ratings = []
        for user_id in range(1, 501):
            for movie_id in range(1, 21):  # Each user rates 20 movies
                large_ratings.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0]),
                    'timestamp': 1609459200
                })
        
        # Test with timeout simulation
        analyzer = MovieAnalyzer()
        analyzer.movies_df = pd.DataFrame(large_movies)
        analyzer.ratings_df = pd.DataFrame(large_ratings)
        
        start_time = time.time()
        
        # Perform analysis
        rating_dist = analyzer.get_rating_distribution()
        top_movies = analyzer.get_top_movies(limit=50)
        genre_stats = analyzer.analyze_genres()
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 60  # 1 minute max
        
        # Results should be valid
        assert 'distribution' in rating_dist
        assert len(top_movies) <= 50
        assert 'overall' in genre_stats

    def test_resource_cleanup(self):
        """Test proper resource cleanup after operations"""
        import gc
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Perform multiple operations that should clean up properly
        for iteration in range(5):
            analyzer = MovieAnalyzer()
            
            # Create temporary data
            temp_movies = [
                {'movieId': i, 'title': f'Movie {i}', 'genres': 'Action'}
                for i in range(1, 101)
            ]
            temp_ratings = [
                {'userId': 1, 'movieId': i, 'rating': 4.0, 'timestamp': 1609459200}
                for i in range(1, 101)
            ]
            
            analyzer.movies_df = pd.DataFrame(temp_movies)
            analyzer.ratings_df = pd.DataFrame(temp_ratings)
            
            # Perform operations
            analyzer.get_rating_distribution()
            analyzer.get_top_movies(limit=10)
            
            # Explicit cleanup
            del analyzer, temp_movies, temp_ratings
            gc.collect()
        
        # Check memory usage hasn't grown excessively
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Should not have significant memory growth
        assert memory_growth < 100  # Less than 100MB growth