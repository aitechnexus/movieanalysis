"""Unit tests for MovieAnalyzer class"""

import pandas as pd
import pytest

from src.analyzer import MovieAnalyzer


class TestMovieAnalyzer:
    """Test cases for MovieAnalyzer class"""

    def test_init(
        self, sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame
    ):
        """Test MovieAnalyzer initialization"""
        analyzer = MovieAnalyzer(sample_movies_data, sample_ratings_data)

        assert analyzer.movies is not None
        assert analyzer.ratings is not None
        assert len(analyzer.movies) == len(sample_movies_data)
        assert len(analyzer.ratings) == len(sample_ratings_data)

    def test_get_top_rated_movies(self, movie_analyzer: MovieAnalyzer):
        """Test getting top rated movies"""
        top_movies = movie_analyzer.get_top_rated_movies(n=3, min_ratings=1)

        assert isinstance(top_movies, pd.DataFrame)
        assert len(top_movies) <= 3
        assert "avg_rating" in top_movies.columns
        assert "rating_count" in top_movies.columns

        # Check that results are sorted by rating
        if len(top_movies) > 1:
            assert top_movies["avg_rating"].is_monotonic_decreasing

    def test_get_most_popular_movies(self, movie_analyzer: MovieAnalyzer):
        """Test getting most popular movies"""
        popular_movies = movie_analyzer.get_most_popular_movies(n=3)

        assert isinstance(popular_movies, pd.DataFrame)
        assert len(popular_movies) <= 3
        assert "rating_count" in popular_movies.columns

        # Check that results are sorted by popularity
        if len(popular_movies) > 1:
            assert popular_movies["rating_count"].is_monotonic_decreasing

    def test_analyze_genres(self, movie_analyzer: MovieAnalyzer):
        """Test genre analysis"""
        genre_stats = movie_analyzer.analyze_genres()

        assert isinstance(genre_stats, pd.DataFrame)
        assert "genre" in genre_stats.columns
        assert "movie_count" in genre_stats.columns
        assert "avg_rating" in genre_stats.columns
        assert len(genre_stats) > 0

    def test_analyze_rating_trends(self, movie_analyzer: MovieAnalyzer):
        """Test rating trends analysis"""
        trends = movie_analyzer.analyze_rating_trends()

        assert isinstance(trends, dict)
        expected_keys = ["yearly_trends", "monthly_trends", "daily_trends"]

        for key in expected_keys:
            assert key in trends
            assert isinstance(trends[key], pd.DataFrame)

    def test_get_user_preferences(self, movie_analyzer: MovieAnalyzer):
        """Test user preferences analysis"""
        # Get a user ID from the sample data
        user_id = movie_analyzer.ratings["userId"].iloc[0]

        preferences = movie_analyzer.get_user_preferences(user_id)

        assert isinstance(preferences, dict)
        expected_keys = [
            "favorite_genres",
            "avg_rating",
            "total_ratings",
            "rating_distribution",
            "activity_level",
        ]

        for key in expected_keys:
            assert key in preferences

    def test_get_movie_recommendations(self, movie_analyzer: MovieAnalyzer):
        """Test movie recommendations"""
        # Get a user ID from the sample data
        user_id = movie_analyzer.ratings["userId"].iloc[0]

        recommendations = movie_analyzer.get_movie_recommendations(
            user_id, method="popularity", limit=3
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        if recommendations:
            assert "movieId" in recommendations[0]
            assert "title" in recommendations[0]
            assert "score" in recommendations[0]

    def test_calculate_movie_similarity(self, movie_analyzer: MovieAnalyzer):
        """Test movie similarity calculation"""
        # Get a movie ID from the sample data
        movie_id = movie_analyzer.movies["movieId"].iloc[0]

        similar_movies = movie_analyzer.calculate_movie_similarity(
            movie_id, method="genre", limit=2
        )

        assert isinstance(similar_movies, list)
        assert len(similar_movies) <= 2
        if similar_movies:
            assert "movieId" in similar_movies[0]
            assert "title" in similar_movies[0]
            assert "similarity_score" in similar_movies[0]

    def test_analyze_rating_patterns(self, movie_analyzer: MovieAnalyzer):
        """Test rating patterns analysis"""
        patterns = movie_analyzer.analyze_rating_patterns()

        assert isinstance(patterns, dict)
        expected_keys = ["rating_distribution", "user_activity", "movie_popularity"]

        for key in expected_keys:
            assert key in patterns

    def test_get_statistics_summary(self, movie_analyzer: MovieAnalyzer):
        """Test statistics summary"""
        summary = movie_analyzer.get_statistics_summary()

        assert isinstance(summary, dict)
        expected_keys = [
            "total_movies",
            "total_users",
            "total_ratings",
            "avg_rating",
            "rating_std",
            "sparsity",
        ]

        for key in expected_keys:
            assert key in summary
            assert isinstance(summary[key], (int, float))

    def test_invalid_user_id(self, movie_analyzer: MovieAnalyzer):
        """Test handling of invalid user ID"""
        invalid_user_id = 99999

        preferences = movie_analyzer.get_user_preferences(invalid_user_id)
        assert preferences is None or len(preferences) == 0

    def test_invalid_movie_id(self, movie_analyzer: MovieAnalyzer):
        """Test handling of invalid movie ID"""
        invalid_movie_id = 99999

        similar_movies = movie_analyzer.calculate_movie_similarity(invalid_movie_id)
        assert len(similar_movies) == 0

    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_movies = pd.DataFrame(columns=["movieId", "title", "genres"])
        empty_ratings = pd.DataFrame(
            columns=["userId", "movieId", "rating", "timestamp"]
        )

        with pytest.raises(ValueError):
            MovieAnalyzer(empty_movies, empty_ratings)

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_top_movies_different_n(self, movie_analyzer: MovieAnalyzer, n: int):
        """Test top movies with different n values"""
        top_movies = movie_analyzer.get_top_rated_movies(n=n, min_ratings=1)

        assert len(top_movies) <= n
        assert len(top_movies) <= len(movie_analyzer.movies_df)

    @pytest.mark.parametrize("min_ratings", [1, 5, 10])
    def test_top_movies_different_min_ratings(
        self, movie_analyzer: MovieAnalyzer, min_ratings: int
    ):
        """Test top movies with different minimum ratings"""
        top_movies = movie_analyzer.get_top_rated_movies(n=5, min_ratings=min_ratings)

        # All movies should have at least min_ratings ratings
        if len(top_movies) > 0:
            assert (top_movies["rating_count"] >= min_ratings).all()

    def test_genre_filtering(self, movie_analyzer: MovieAnalyzer):
        """Test genre-based filtering"""
        # Get available genres
        genre_stats = movie_analyzer.analyze_genres()
        if len(genre_stats) > 0:
            test_genre = genre_stats["genre"].iloc[0]

            genre_movies = movie_analyzer.get_movies_by_genre(test_genre)

            assert isinstance(genre_movies, pd.DataFrame)
            # All movies should contain the specified genre
            assert genre_movies["genres"].str.contains(test_genre, case=False).all()

    def test_time_based_analysis(self, movie_analyzer: MovieAnalyzer):
        """Test time-based analysis functionality"""
        # Ensure timestamp is datetime
        movie_analyzer.ratings_df["timestamp"] = pd.to_datetime(
            movie_analyzer.ratings_df["timestamp"]
        )

        time_analysis = movie_analyzer.analyze_temporal_patterns()

        assert isinstance(time_analysis, dict)
        expected_keys = ["hourly_patterns", "daily_patterns", "monthly_patterns"]

        for key in expected_keys:
            if key in time_analysis:
                assert isinstance(time_analysis[key], pd.DataFrame)
