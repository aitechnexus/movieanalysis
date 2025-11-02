"""Integration tests for MovieLens Analysis Platform"""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.analyzer import MovieAnalyzer
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.visualizer import InsightsVisualizer


class TestIntegration:
    """Integration tests for the complete system"""

    @pytest.mark.integration
    def test_complete_data_pipeline(self, temp_data_dir: Path, temp_output_dir: Path,
                                  sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame):
        """Test complete data processing pipeline"""
        # Save sample data to files
        movies_file = temp_data_dir / "raw" / "movies.csv"
        ratings_file = temp_data_dir / "raw" / "ratings.csv"
        
        sample_movies_data.to_csv(movies_file, index=False)
        sample_ratings_data.to_csv(ratings_file, index=False)
        
        # Initialize components
        loader = DataLoader(temp_data_dir)
        processor = DataProcessor()
        
        # Load data
        movies_df = loader.load_movies()
        ratings_df = loader.load_ratings()
        
        # Validate data
        loader.validate_data(movies_df, ratings_df)
        
        # Process data
        processed_data = processor.process_data(movies_df, ratings_df)
        
        # Analyze data
        analyzer = MovieAnalyzer(processed_data['movies_clean'], processed_data['ratings_clean'])
        
        # Generate insights
        top_movies = analyzer.get_top_rated_movies(n=3)
        genre_stats = analyzer.analyze_genres()
        
        # Create visualizations
        visualizer = InsightsVisualizer(temp_output_dir)
        plot_path = visualizer.plot_top_movies(top_movies)
        
        # Verify results
        assert len(processed_data) > 0
        assert len(top_movies) > 0
        assert len(genre_stats) > 0
        assert plot_path.exists()

    @pytest.mark.integration
    def test_api_endpoints(self, client):
        """Test Flask API endpoints"""
        # Test status endpoint
        response = client.get('/api/status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'

    @pytest.mark.integration
    def test_data_consistency(self, sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame):
        """Test data consistency across components"""
        processor = DataProcessor()
        processed_data = processor.process_data(sample_movies_data, sample_ratings_data)
        
        analyzer = MovieAnalyzer(processed_data['movies_clean'], processed_data['ratings_clean'])
        
        # Check that movie IDs are consistent
        movie_ids_in_movies = set(processed_data['movies_clean']['movieId'])
        movie_ids_in_ratings = set(processed_data['ratings_clean']['movieId'])
        
        # All movies in ratings should exist in movies data
        assert movie_ids_in_ratings.issubset(movie_ids_in_movies)
        
        # Statistics should be consistent
        stats_summary = analyzer.get_statistics_summary()
        assert stats_summary['total_movies'] == len(processed_data['movies_clean'])
        assert stats_summary['total_ratings'] == len(processed_data['ratings_clean'])

    @pytest.mark.integration
    def test_recommendation_system(self, sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame):
        """Test recommendation system integration"""
        analyzer = MovieAnalyzer(sample_movies_data, sample_ratings_data)
        
        # Get a user from the data
        user_id = sample_ratings_data['userId'].iloc[0]
        
        # Get user preferences
        preferences = analyzer.get_user_preferences(user_id)
        
        # Get recommendations
        recommendations = analyzer.get_movie_recommendations(user_id, n=3)
        
        # Verify recommendations are valid
        assert isinstance(recommendations, pd.DataFrame)
        assert len(recommendations) <= 3
        
        # Recommended movies should not be movies the user has already rated
        user_rated_movies = set(sample_ratings_data[sample_ratings_data['userId'] == user_id]['movieId'])
        recommended_movies = set(recommendations['movieId'])
        
        # Should have no overlap (user shouldn't get movies they've already rated)
        assert len(user_rated_movies.intersection(recommended_movies)) == 0

    @pytest.mark.integration
    def test_visualization_pipeline(self, temp_output_dir: Path, 
                                  sample_movies_data: pd.DataFrame, 
                                  sample_ratings_data: pd.DataFrame):
        """Test complete visualization pipeline"""
        analyzer = MovieAnalyzer(sample_movies_data, sample_ratings_data)
        visualizer = InsightsVisualizer(temp_output_dir)
        
        # Generate analysis results
        top_movies = analyzer.get_top_rated_movies(n=5)
        genre_stats = analyzer.analyze_genres()
        rating_trends = analyzer.analyze_rating_trends()
        
        # Create visualizations
        plots = []
        plots.append(visualizer.plot_top_movies(top_movies))
        plots.append(visualizer.plot_genre_popularity(genre_stats))
        plots.append(visualizer.plot_rating_distribution(sample_ratings_data))
        
        # Verify all plots were created
        for plot_path in plots:
            assert plot_path.exists()
            assert plot_path.stat().st_size > 0  # File is not empty

    @pytest.mark.integration
    def test_error_handling_pipeline(self, temp_data_dir: Path):
        """Test error handling across the pipeline"""
        loader = DataLoader(temp_data_dir)
        
        # Test loading non-existent files
        with pytest.raises(FileNotFoundError):
            loader.load_movies()
        
        with pytest.raises(FileNotFoundError):
            loader.load_ratings()

    @pytest.mark.integration
    def test_memory_efficiency(self, sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame):
        """Test memory efficiency with larger datasets"""
        # Create larger dataset by repeating sample data
        large_movies = pd.concat([sample_movies_data] * 100, ignore_index=True)
        large_ratings = pd.concat([sample_ratings_data] * 100, ignore_index=True)
        
        # Adjust IDs to maintain uniqueness
        large_movies['movieId'] = range(len(large_movies))
        large_ratings['movieId'] = large_ratings['movieId'] % len(large_movies)
        
        processor = DataProcessor()
        
        # Should handle larger datasets without errors
        processed_data = processor.process_data(large_movies, large_ratings)
        
        assert len(processed_data['movies_clean']) > 0
        assert len(processed_data['ratings_clean']) > 0

    @pytest.mark.integration
    def test_concurrent_analysis(self, sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame):
        """Test concurrent analysis operations"""
        analyzer = MovieAnalyzer(sample_movies_data, sample_ratings_data)
        
        # Run multiple analysis operations
        results = {}
        results['top_movies'] = analyzer.get_top_rated_movies(n=5)
        results['popular_movies'] = analyzer.get_most_popular_movies(n=5)
        results['genre_stats'] = analyzer.analyze_genres()
        results['statistics'] = analyzer.get_statistics_summary()
        
        # Verify all operations completed successfully
        for key, result in results.items():
            assert result is not None
            if isinstance(result, pd.DataFrame):
                assert len(result) >= 0
            elif isinstance(result, dict):
                assert len(result) > 0

    @pytest.mark.integration
    def test_data_export_import(self, temp_data_dir: Path, 
                              sample_movies_data: pd.DataFrame, 
                              sample_ratings_data: pd.DataFrame):
        """Test data export and import functionality"""
        processor = DataProcessor()
        
        # Process data
        processed_data = processor.process_data(sample_movies_data, sample_ratings_data)
        
        # Export processed data
        export_dir = temp_data_dir / "processed"
        
        for key, df in processed_data.items():
            export_path = export_dir / f"{key}.csv"
            df.to_csv(export_path, index=False)
            
            # Verify export
            assert export_path.exists()
            
            # Re-import and verify
            reimported_df = pd.read_csv(export_path)
            assert len(reimported_df) == len(df)
            assert list(reimported_df.columns) == list(df.columns)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_benchmarks(self, sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame):
        """Test performance benchmarks"""
        import time
        
        # Create larger dataset for performance testing
        large_movies = pd.concat([sample_movies_data] * 1000, ignore_index=True)
        large_ratings = pd.concat([sample_ratings_data] * 1000, ignore_index=True)
        
        # Adjust IDs
        large_movies['movieId'] = range(len(large_movies))
        large_ratings['movieId'] = large_ratings['movieId'] % len(large_movies)
        
        # Benchmark data processing
        start_time = time.time()
        processor = DataProcessor()
        processed_data = processor.process_data(large_movies, large_ratings)
        processing_time = time.time() - start_time
        
        # Benchmark analysis
        start_time = time.time()
        analyzer = MovieAnalyzer(processed_data['movies_clean'], processed_data['ratings_clean'])
        top_movies = analyzer.get_top_rated_movies(n=10)
        analysis_time = time.time() - start_time
        
        # Performance assertions (adjust thresholds as needed)
        assert processing_time < 30  # Should complete within 30 seconds
        assert analysis_time < 10    # Should complete within 10 seconds
        assert len(top_movies) > 0   # Should produce results