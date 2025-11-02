"""Unit tests for DataProcessor class"""

import pandas as pd
import pytest

from src.data_processor import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor class"""

    def test_init(self):
        """Test DataProcessor initialization"""
        processor = DataProcessor()
        assert processor is not None

    def test_clean_movies_data(self, data_processor: DataProcessor):
        """Test movies data cleaning"""
        # Create test data with issues
        dirty_movies = pd.DataFrame(
            {
                "movieId": [1, 2, 3, None, 5],
                "title": ["Movie 1", "", "Movie 3", "Movie 4", None],
                "genres": [
                    "Action",
                    "Comedy|Drama",
                    "",
                    "Horror",
                    "(no genres listed)",
                ],
            }
        )

        cleaned = data_processor.clean_movies_data(dirty_movies)

        # Should remove rows with missing movieId or title
        assert len(cleaned) == 3
        assert cleaned["movieId"].notna().all()
        assert cleaned["title"].str.len().gt(0).all()

    def test_clean_ratings_data(self, data_processor: DataProcessor):
        """Test ratings data cleaning"""
        # Create test data with issues
        dirty_ratings = pd.DataFrame(
            {
                "userId": [1, 2, None, 4, 5],
                "movieId": [1, 2, 3, None, 5],
                "rating": [4.0, -1.0, 3.5, 2.5, 6.0],
                "timestamp": [
                    1234567890,
                    1234567891,
                    1234567892,
                    1234567893,
                    1234567894,
                ],
            }
        )

        cleaned = data_processor.clean_ratings_data(dirty_ratings)

        # Should remove rows with missing IDs or invalid ratings
        assert len(cleaned) == 2  # Only valid rows remain
        assert cleaned["userId"].notna().all()
        assert cleaned["movieId"].notna().all()
        assert (cleaned["rating"] >= 0.5).all()
        assert (cleaned["rating"] <= 5.0).all()

    def test_extract_genres(
        self, data_processor: DataProcessor, sample_movies_data: pd.DataFrame
    ):
        """Test genre extraction"""
        genres_df = data_processor.extract_genres(sample_movies_data)

        assert "movieId" in genres_df.columns
        assert "genre" in genres_df.columns
        assert len(genres_df) > len(
            sample_movies_data
        )  # Should have more rows due to genre splitting

    def test_calculate_movie_stats(
        self,
        data_processor: DataProcessor,
        sample_movies_data: pd.DataFrame,
        sample_ratings_data: pd.DataFrame,
    ):
        """Test movie statistics calculation"""
        stats_df = data_processor.calculate_movie_stats(
            sample_movies_data, sample_ratings_data
        )

        expected_columns = [
            "movieId",
            "title",
            "genres",
            "avg_rating",
            "rating_count",
            "rating_std",
            "min_rating",
            "max_rating",
        ]

        for col in expected_columns:
            assert col in stats_df.columns

        # Check data types
        assert stats_df["avg_rating"].dtype in ["float64", "float32"]
        assert stats_df["rating_count"].dtype in ["int64", "int32"]

    def test_calculate_user_stats(
        self, data_processor: DataProcessor, sample_ratings_data: pd.DataFrame
    ):
        """Test user statistics calculation"""
        stats_df = data_processor.calculate_user_stats(sample_ratings_data)

        expected_columns = [
            "userId",
            "avg_rating",
            "rating_count",
            "rating_std",
            "min_rating",
            "max_rating",
        ]

        for col in expected_columns:
            assert col in stats_df.columns

    def test_create_time_features(
        self, data_processor: DataProcessor, sample_ratings_data: pd.DataFrame
    ):
        """Test time feature creation"""
        # Ensure timestamp is datetime
        sample_ratings_data["timestamp"] = pd.to_datetime(
            sample_ratings_data["timestamp"]
        )

        enhanced_df = data_processor.create_time_features(sample_ratings_data)

        expected_columns = ["year", "month", "day_of_week", "hour"]
        for col in expected_columns:
            assert col in enhanced_df.columns

    def test_normalize_ratings(
        self, data_processor: DataProcessor, sample_ratings_data: pd.DataFrame
    ):
        """Test rating normalization"""
        normalized_df = data_processor.normalize_ratings(sample_ratings_data)

        assert "normalized_rating" in normalized_df.columns

        # Check normalization bounds
        assert normalized_df["normalized_rating"].min() >= -1
        assert normalized_df["normalized_rating"].max() <= 1

    def test_detect_outliers(self, data_processor: DataProcessor):
        """Test outlier detection"""
        # Create test data with clear outliers
        test_data = pd.DataFrame(
            {
                "userId": [1, 2, 3, 4, 5],
                "rating_count": [10, 12, 15, 1000, 11],  # 1000 is clearly an outlier
            }
        )

        outliers = data_processor.detect_outliers(test_data, "rating_count")

        assert len(outliers) == 1
        assert outliers.iloc[0]["rating_count"] == 1000

    def test_process_data_pipeline(
        self,
        data_processor: DataProcessor,
        sample_movies_data: pd.DataFrame,
        sample_ratings_data: pd.DataFrame,
    ):
        """Test complete data processing pipeline"""
        processed_data = data_processor.process_data(
            sample_movies_data, sample_ratings_data
        )

        # Check that all expected keys are present
        expected_keys = [
            "movies_clean",
            "ratings_clean",
            "movie_stats",
            "user_stats",
            "genres_df",
            "ratings_enhanced",
        ]

        for key in expected_keys:
            assert key in processed_data
            assert isinstance(processed_data[key], pd.DataFrame)
            assert len(processed_data[key]) > 0

    def test_empty_dataframe_handling(self, data_processor: DataProcessor):
        """Test handling of empty dataframes"""
        empty_movies = pd.DataFrame(columns=["movieId", "title", "genres"])
        empty_ratings = pd.DataFrame(
            columns=["userId", "movieId", "rating", "timestamp"]
        )

        with pytest.raises(ValueError):
            data_processor.process_data(empty_movies, empty_ratings)

    def test_missing_columns_handling(self, data_processor: DataProcessor):
        """Test handling of missing required columns"""
        invalid_movies = pd.DataFrame({"id": [1, 2], "name": ["Movie 1", "Movie 2"]})
        invalid_ratings = pd.DataFrame(
            {"user": [1, 2], "movie": [1, 2], "score": [4.0, 3.5]}
        )

        with pytest.raises(KeyError):
            data_processor.process_data(invalid_movies, invalid_ratings)

    @pytest.mark.parametrize("method", ["iqr", "zscore", "isolation"])
    def test_outlier_detection_methods(
        self, data_processor: DataProcessor, method: str
    ):
        """Test different outlier detection methods"""
        test_data = pd.DataFrame(
            {
                "userId": range(100),
                "rating_count": [10] * 95 + [1000] * 5,  # 5 clear outliers
            }
        )

        outliers = data_processor.detect_outliers(
            test_data, "rating_count", method=method
        )

        # Should detect some outliers
        assert len(outliers) > 0
        assert len(outliers) < len(test_data)
