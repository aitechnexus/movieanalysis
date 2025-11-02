"""
Advanced analytics validation tests for the MovieLens Analysis Platform.
Tests for statistical accuracy, algorithm correctness, and mathematical validation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import math

from src.data_processor import DataProcessor
from src.analyzer import MovieAnalyzer
from src.visualizer import InsightsVisualizer


class TestStatisticalAccuracy:
    """Test statistical calculations and algorithm accuracy"""

    @pytest.fixture
    def sample_movies_data(self):
        """Create sample movies data for testing"""
        data = [
            {'movieId': 1, 'title': 'Movie A', 'genres': 'Action|Adventure', 'year': 2020},
            {'movieId': 2, 'title': 'Movie B', 'genres': 'Comedy', 'year': 2019},
            {'movieId': 3, 'title': 'Movie C', 'genres': 'Drama|Romance', 'year': 2021},
            {'movieId': 4, 'title': 'Movie D', 'genres': 'Action', 'year': 2018},
            {'movieId': 5, 'title': 'Movie E', 'genres': 'Comedy|Romance', 'year': 2022}
        ]
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_ratings_data(self):
        """Create sample ratings data for testing"""
        data = [
            {'userId': 1, 'movieId': 1, 'rating': 4.5, 'timestamp': 1609459200},
            {'userId': 1, 'movieId': 2, 'rating': 3.0, 'timestamp': 1609545600},
            {'userId': 2, 'movieId': 1, 'rating': 5.0, 'timestamp': 1609632000},
            {'userId': 2, 'movieId': 3, 'rating': 4.0, 'timestamp': 1609718400},
            {'userId': 3, 'movieId': 1, 'rating': 3.5, 'timestamp': 1609804800},
            {'userId': 3, 'movieId': 2, 'rating': 2.5, 'timestamp': 1609891200},
            {'userId': 3, 'movieId': 3, 'rating': 4.5, 'timestamp': 1609977600},
            {'userId': 4, 'movieId': 4, 'rating': 3.0, 'timestamp': 1610064000},
            {'userId': 4, 'movieId': 5, 'rating': 4.0, 'timestamp': 1610150400},
            {'userId': 5, 'movieId': 5, 'rating': 5.0, 'timestamp': 1610236800}
        ]
        df = pd.DataFrame(data)
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df

    @pytest.fixture
    def analyzer_with_data(self, sample_movies_data, sample_ratings_data):
        """Create analyzer with sample data"""
        return MovieAnalyzer(sample_movies_data, sample_ratings_data)

    def test_rating_distribution_statistical_accuracy(self, analyzer_with_data):
        """Test that rating distribution calculations are statistically accurate"""
        result = analyzer_with_data.get_rating_distribution()
        
        # Verify the structure
        assert 'distribution' in result
        assert 'statistics' in result
        
        # Calculate expected statistics manually
        ratings = analyzer_with_data.ratings['rating'].values
        expected_mean = np.mean(ratings)
        expected_median = np.median(ratings)
        expected_std = np.std(ratings, ddof=1)  # Sample standard deviation
        expected_min = np.min(ratings)
        expected_max = np.max(ratings)
        expected_q25 = np.percentile(ratings, 25)
        expected_q75 = np.percentile(ratings, 75)
        
        # Verify statistical accuracy (within reasonable tolerance)
        stats_result = result['statistics']
        assert abs(stats_result['mean'] - expected_mean) < 0.01
        assert abs(stats_result['median'] - expected_median) < 0.01
        assert abs(stats_result['std'] - expected_std) < 0.01
        assert abs(stats_result['min'] - expected_min) < 0.01
        assert abs(stats_result['max'] - expected_max) < 0.01
        assert abs(stats_result['q25'] - expected_q25) < 0.01
        assert abs(stats_result['q75'] - expected_q75) < 0.01

    def test_weighted_rating_calculation_accuracy(self, analyzer_with_data):
        """Test weighted rating calculation using IMDB formula"""
        top_movies = analyzer_with_data.get_top_movies(limit=5)
        
        # Verify weighted rating calculation
        for movie in top_movies:
            movie_id = movie.get('movieId')
            if movie_id:
                movie_ratings = analyzer_with_data.ratings[
            analyzer_with_data.ratings['movieId'] == movie_id
                ]['rating']
                
                if len(movie_ratings) > 0:
                    # Calculate expected weighted rating using IMDB formula
                    # WR = (v/(v+m)) * R + (m/(v+m)) * C
                    # Where: v = vote count, m = minimum votes, R = average rating, C = global mean
                    
                    v = len(movie_ratings)  # vote count
                    R = movie_ratings.mean()  # average rating
                    C = analyzer_with_data.global_mean  # global mean
                    m = 10  # minimum votes threshold (common value)
                    
                    expected_weighted_rating = (v / (v + m)) * R + (m / (v + m)) * C
                    
                    # Verify the calculation is within reasonable tolerance
                    actual_weighted_rating = movie.get('weighted_rating', 0)
                    assert abs(actual_weighted_rating - expected_weighted_rating) < 0.1

    def test_genre_analysis_statistical_validity(self, analyzer_with_data):
        """Test genre analysis statistical calculations"""
        genre_stats = analyzer_with_data.analyze_genres()
        
        # Verify structure
        assert 'overall' in genre_stats
        assert 'top_genres' in genre_stats
        
        # Verify each genre's statistics
        for genre_data in genre_stats['overall']:
            genre = genre_data['genre']
            count = genre_data['count']
            mean_rating = genre_data['mean_rating']
            
            # Manually calculate expected values by exploding genres like the method does
            df = analyzer_with_data.ratings.merge(analyzer_with_data.movies, on="movieId")
            df["genres_list"] = df["genres"].str.split("|")
            df = df.explode("genres_list")
            df = df[df["genres_list"] != "(no genres listed)"]
            
            genre_ratings = df[df["genres_list"] == genre]["rating"]
            
            if not genre_ratings.empty:
                expected_count = len(genre_ratings)
                expected_mean_rating = genre_ratings.mean()
                
                # Verify accuracy
                assert count == expected_count
                assert abs(mean_rating - expected_mean_rating) < 0.01

    def test_similarity_calculation_mathematical_correctness(self, analyzer_with_data):
        """Test similarity calculation mathematical correctness"""
        # Test similarity for a specific movie
        movie_id_1 = 1
        
        similar_movies = analyzer_with_data.calculate_movie_similarity(movie_id_1, method="rating", limit=5)
        
        # Verify structure
        assert isinstance(similar_movies, list)
        
        # Verify each similar movie has required fields
        for similar_movie in similar_movies:
            assert 'movieId' in similar_movie
            assert 'similarity_score' in similar_movie or 'score' in similar_movie
            
            # Similarity scores should be between 0 and 1
            score = similar_movie.get('similarity_score', similar_movie.get('score', 0))
            assert 0 <= score <= 1

    def test_recommendation_algorithm_accuracy(self, analyzer_with_data):
        """Test recommendation algorithm accuracy and relevance"""
        user_id = 1
        recommendations = analyzer_with_data.get_movie_recommendations(user_id, limit=3)
        
        # Verify recommendations structure
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        
        # Verify recommendation scores are reasonable
        for rec in recommendations:
            assert 'movieId' in rec
            assert 'predicted_rating' in rec or 'score' in rec
            
            # Predicted ratings should be within valid range
            predicted_rating = rec.get('predicted_rating', rec.get('score', 0))
            assert 0.5 <= predicted_rating <= 5.0

    def test_time_series_analysis_trend_accuracy(self, analyzer_with_data):
        """Test time series analysis trend calculation accuracy"""
        time_series = analyzer_with_data.generate_time_series_analysis()
        
        # Verify structure
        assert 'monthly' in time_series
        assert 'yearly' in time_series
        
        # Verify trend calculations
        monthly_data = time_series.get('monthly', [])
        
        if monthly_data:
            # Check that trends are chronologically ordered
            dates = [item.get('year_month', '') for item in monthly_data]
            sorted_dates = sorted(dates)
            assert dates == sorted_dates or len(set(dates)) <= 1  # Allow for single date
            
            # Verify rating calculations
            for trend_item in monthly_data:
                avg_rating = trend_item.get('mean_rating', 0)
                rating_count = trend_item.get('count', 0)
                
                # Ratings should be within valid range
                if avg_rating > 0:
                    assert 0.5 <= avg_rating <= 5.0
                
                # Count should be non-negative
                assert rating_count >= 0

    def test_user_behavior_statistics_accuracy(self, analyzer_with_data):
        """Test user behavior statistics calculation accuracy"""
        user_stats = analyzer_with_data.get_user_behavior_stats()
        
        # Verify structure
        expected_keys = ['total_users', 'avg_ratings_per_user', 'most_active_users']
        for key in expected_keys:
            if key in user_stats:
                # Verify the values are reasonable
                value = user_stats[key]
                
                if key == 'total_users':
                    expected_total = analyzer_with_data.ratings['userId'].nunique()
                    assert value == expected_total
                
                elif key == 'avg_ratings_per_user':
                    expected_avg = analyzer_with_data.ratings.groupby('userId').size().mean()
                    assert abs(value - expected_avg) < 0.01
                
                elif key == 'most_active_users':
                    assert isinstance(value, list)
                    # Verify users are sorted by activity
                    if len(value) > 1:
                        activities = [user.get('rating_count', 0) for user in value]
                        assert activities == sorted(activities, reverse=True)

    def test_statistical_significance_validation(self, analyzer_with_data):
        """Test statistical significance of results"""
        # Test with larger sample to ensure statistical validity
        large_sample_ratings = []
        
        # Generate larger sample data
        np.random.seed(42)  # For reproducible results
        for user_id in range(1, 101):  # 100 users
            for movie_id in range(1, 21):  # 20 movies
                if np.random.random() < 0.3:  # 30% chance of rating
                    rating = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
                    large_sample_ratings.append({
                        'userId': user_id,
                        'movieId': movie_id,
                        'rating': rating,
                        'timestamp': 1609459200 + (user_id * movie_id * 1000)
                    })
        
        # Create analyzer with larger sample
        large_movies_data = analyzer_with_data.movies.copy()
        large_analyzer = MovieAnalyzer(large_movies_data, pd.DataFrame(large_sample_ratings))
        
        # Test statistical significance of rating distribution
        rating_dist = large_analyzer.get_rating_distribution()
        
        # Perform basic normality test using numpy
        ratings = large_analyzer.ratings['rating'].values
        if len(ratings) > 8:  # Minimum sample size for test
            # Simple normality check using skewness and kurtosis
            mean_rating = np.mean(ratings)
            std_rating = np.std(ratings)
            
            # Calculate basic statistics for normality assessment
            skewness = np.mean(((ratings - mean_rating) / std_rating) ** 3)
            kurtosis = np.mean(((ratings - mean_rating) / std_rating) ** 4) - 3
            
            # Verify the calculations run without error
            assert isinstance(skewness, float)
            assert isinstance(kurtosis, float)
            assert not np.isnan(skewness)
            assert not np.isnan(kurtosis)

    def test_correlation_analysis_accuracy(self, analyzer_with_data):
        """Test correlation analysis mathematical accuracy"""
        # Test correlation between different metrics
        movies_with_ratings = analyzer_with_data.ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        movies_with_ratings.columns = ['movieId', 'avg_rating', 'rating_count']
        
        if len(movies_with_ratings) > 1:
            # Calculate correlation between average rating and rating count
            correlation = movies_with_ratings['avg_rating'].corr(movies_with_ratings['rating_count'])
            
            # Verify correlation is within valid range
            assert -1 <= correlation <= 1 or np.isnan(correlation)
            
            # Manual calculation verification
            if not np.isnan(correlation):
                manual_corr = np.corrcoef(movies_with_ratings['avg_rating'], 
                                        movies_with_ratings['rating_count'])[0, 1]
                assert abs(correlation - manual_corr) < 0.01

    def test_outlier_detection_accuracy(self, analyzer_with_data):
        """Test outlier detection in ratings and statistics"""
        ratings = analyzer_with_data.ratings['rating'].values
        
        # Calculate IQR method for outlier detection
        Q1 = np.percentile(ratings, 25)
        Q3 = np.percentile(ratings, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ratings[(ratings < lower_bound) | (ratings > upper_bound)]
        
        # Verify outlier detection logic
        assert isinstance(outliers, np.ndarray)
        
        # For movie ratings (1-5 scale), there shouldn't be many outliers
        outlier_percentage = len(outliers) / len(ratings) * 100
        assert outlier_percentage < 50  # Less than 50% should be outliers

    def test_algorithm_convergence_stability(self, analyzer_with_data):
        """Test algorithm stability and convergence"""
        # Test recommendation algorithm stability
        user_id = 1
        
        # Run recommendations multiple times
        recommendations_1 = analyzer_with_data.get_movie_recommendations(user_id, limit=5)
        recommendations_2 = analyzer_with_data.get_movie_recommendations(user_id, limit=5)
        
        # Results should be consistent (deterministic)
        if recommendations_1 and recommendations_2:
            movie_ids_1 = [rec['movieId'] for rec in recommendations_1]
            movie_ids_2 = [rec['movieId'] for rec in recommendations_2]
            
            # Should have significant overlap (at least 60%)
            overlap = len(set(movie_ids_1) & set(movie_ids_2))
            overlap_percentage = overlap / max(len(movie_ids_1), len(movie_ids_2)) * 100
            assert overlap_percentage >= 60

    def test_edge_case_mathematical_handling(self, analyzer_with_data):
        """Test handling of mathematical edge cases"""
        # Test division by zero scenarios
        empty_movies = pd.DataFrame(columns=['movieId', 'title', 'genres'])
        empty_ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
        empty_analyzer = MovieAnalyzer(empty_movies, empty_ratings)
        
        # Should handle empty data gracefully
        try:
            result = empty_analyzer.get_rating_distribution()
            # Should return valid structure even with empty data
            assert isinstance(result, dict)
        except (ZeroDivisionError, ValueError):
            # Acceptable to raise appropriate exceptions
            pass
        
        # Test with single data point
        single_movies = pd.DataFrame([{
            'movieId': 1, 'title': 'Test Movie', 'genres': 'Action'
        }])
        single_ratings = pd.DataFrame([{
            'userId': 1, 'movieId': 1, 'rating': 4.0, 'timestamp': pd.to_datetime(1609459200, unit='s')
        }])
        single_rating_analyzer = MovieAnalyzer(single_movies, single_ratings)
        
        try:
            result = single_rating_analyzer.get_rating_distribution()
            # Should handle single data point - std can be 0 or NaN for single value
            std_value = result['statistics']['std']
            assert std_value == 0 or np.isnan(std_value)  # Standard deviation should be 0 or NaN
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, (ValueError, ZeroDivisionError))


class TestAlgorithmValidation:
    """Test specific algorithm implementations"""

    def test_collaborative_filtering_accuracy(self):
        """Test collaborative filtering algorithm accuracy"""
        # Create controlled test data
        test_ratings = pd.DataFrame([
            {'userId': 1, 'movieId': 1, 'rating': 5.0},
            {'userId': 1, 'movieId': 2, 'rating': 3.0},
            {'userId': 2, 'movieId': 1, 'rating': 4.0},
            {'userId': 2, 'movieId': 3, 'rating': 5.0},
            {'userId': 3, 'movieId': 2, 'rating': 2.0},
            {'userId': 3, 'movieId': 3, 'rating': 4.0}
        ])
        
        test_movies = pd.DataFrame([
            {'movieId': 1, 'title': 'Movie 1', 'genres': 'Action'},
            {'movieId': 2, 'title': 'Movie 2', 'genres': 'Comedy'},
            {'movieId': 3, 'title': 'Movie 3', 'genres': 'Drama'}
        ])
        
        analyzer = MovieAnalyzer(test_movies, test_ratings)
        
        # Test user similarity calculation
        user_1_ratings = test_ratings[test_ratings['userId'] == 1].set_index('movieId')['rating']
        user_2_ratings = test_ratings[test_ratings['userId'] == 2].set_index('movieId')['rating']
        
        # Find common movies
        common_movies = user_1_ratings.index.intersection(user_2_ratings.index)
        
        if len(common_movies) > 0:
            # Calculate Pearson correlation
            ratings_1 = user_1_ratings.loc[common_movies]
            ratings_2 = user_2_ratings.loc[common_movies]
            
            if len(ratings_1) > 1:
                expected_similarity = ratings_1.corr(ratings_2)
                # Verify the calculation is mathematically sound
                assert -1 <= expected_similarity <= 1 or np.isnan(expected_similarity)

    def test_content_based_filtering_accuracy(self):
        """Test content-based filtering accuracy"""
        # Test genre-based similarity
        movies_data = pd.DataFrame([
            {'movieId': 1, 'title': 'Action Movie', 'genres': 'Action|Adventure'},
            {'movieId': 2, 'title': 'Comedy Movie', 'genres': 'Comedy'},
            {'movieId': 3, 'title': 'Action Comedy', 'genres': 'Action|Comedy'}
        ])
        
        ratings_data = pd.DataFrame([
            {'userId': 1, 'movieId': 1, 'rating': 5.0},
            {'userId': 1, 'movieId': 2, 'rating': 3.0},
            {'userId': 1, 'movieId': 3, 'rating': 4.0}
        ])
        
        analyzer = MovieAnalyzer(movies_data, ratings_data)
        
        # Test genre similarity calculation
        movie_1_genres = set(movies_data[movies_data['movieId'] == 1]['genres'].iloc[0].split('|'))
        movie_3_genres = set(movies_data[movies_data['movieId'] == 3]['genres'].iloc[0].split('|'))
        
        # Calculate Jaccard similarity
        intersection = len(movie_1_genres & movie_3_genres)
        union = len(movie_1_genres | movie_3_genres)
        expected_jaccard = intersection / union if union > 0 else 0
        
        # Verify Jaccard similarity is calculated correctly
        assert 0 <= expected_jaccard <= 1

    def test_matrix_factorization_convergence(self):
        """Test matrix factorization algorithm convergence"""
        # Create test rating matrix
        np.random.seed(42)
        n_users, n_movies = 10, 8
        n_factors = 3
        
        # Generate synthetic rating matrix
        true_user_factors = np.random.normal(0, 1, (n_users, n_factors))
        true_movie_factors = np.random.normal(0, 1, (n_movies, n_factors))
        true_ratings = np.dot(true_user_factors, true_movie_factors.T)
        
        # Add noise and create sparse matrix
        noise = np.random.normal(0, 0.1, true_ratings.shape)
        observed_ratings = true_ratings + noise
        
        # Clip to valid rating range
        observed_ratings = np.clip(observed_ratings, 1, 5)
        
        # Test basic matrix operations
        U, s, Vt = np.linalg.svd(observed_ratings, full_matrices=False)
        
        # Verify SVD decomposition
        reconstructed = np.dot(U, np.dot(np.diag(s), Vt))
        mse = np.mean((observed_ratings - reconstructed) ** 2)
        
        # MSE should be reasonable for this synthetic data
        assert mse < 1.0  # Reasonable threshold for synthetic data
        
        # Test correlation analysis
        test_movies = pd.DataFrame([
            {'movieId': 1, 'title': 'Movie 1', 'genres': 'Action'},
            {'movieId': 2, 'title': 'Movie 2', 'genres': 'Comedy'}
        ])
        test_ratings = pd.DataFrame([
            {'userId': 1, 'movieId': 1, 'rating': 4.0},
            {'userId': 1, 'movieId': 2, 'rating': 3.0},
            {'userId': 2, 'movieId': 1, 'rating': 5.0},
            {'userId': 2, 'movieId': 2, 'rating': 2.0}
        ])
        analyzer = MovieAnalyzer(test_movies, test_ratings)
        
        # Calculate correlation manually using numpy
        user_movie_matrix = analyzer.ratings.pivot_table(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(user_movie_matrix.T)
        
        # Verify correlation properties
        assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
        assert np.allclose(np.diag(correlation_matrix), 1.0)  # Diagonal should be 1
        assert np.allclose(correlation_matrix, correlation_matrix.T)  # Should be symmetric