"""Unit tests for DataLoader class"""
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class"""

    def test_init(self, temp_data_dir: Path):
        """Test DataLoader initialization"""
        loader = DataLoader(temp_data_dir, source="grouplens", dataset="ml-latest-small")
        assert loader.data_dir == temp_data_dir
        assert loader.source == "grouplens"
        assert loader.dataset == "ml-latest-small"

    def test_init_creates_directories(self, temp_data_dir: Path):
        """Test that DataLoader creates necessary directories"""
        loader = DataLoader(temp_data_dir)
        assert (temp_data_dir / "raw").exists()
        assert (temp_data_dir / "processed").exists()

    @patch('src.data_loader.requests.get')
    def test_download_data_success(self, mock_get, temp_data_dir: Path):
        """Test successful data download"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"test data"
        mock_get.return_value = mock_response

        loader = DataLoader(temp_data_dir)
        result = loader.download_data("http://test.com/data.zip")
        
        assert result is True
        mock_get.assert_called_once_with("http://test.com/data.zip", timeout=30)

    @patch('src.data_loader.requests.get')
    def test_download_data_failure(self, mock_get, temp_data_dir: Path):
        """Test failed data download"""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        loader = DataLoader(temp_data_dir)
        result = loader.download_data("http://test.com/nonexistent.zip")
        
        assert result is False

    def test_load_movies_file_not_found(self, temp_data_dir: Path):
        """Test loading movies when file doesn't exist"""
        loader = DataLoader(temp_data_dir)
        
        with pytest.raises(FileNotFoundError):
            loader.load_movies()

    def test_load_ratings_file_not_found(self, temp_data_dir: Path):
        """Test loading ratings when file doesn't exist"""
        loader = DataLoader(temp_data_dir)
        
        with pytest.raises(FileNotFoundError):
            loader.load_ratings()

    def test_load_movies_success(self, temp_data_dir: Path):
        """Test successful movies loading"""
        # Create test movies file
        movies_file = temp_data_dir / "raw" / "movies.csv"
        test_data = pd.DataFrame({
            'movieId': [1, 2],
            'title': ['Movie 1', 'Movie 2'],
            'genres': ['Action', 'Comedy']
        })
        test_data.to_csv(movies_file, index=False)

        loader = DataLoader(temp_data_dir)
        movies_df = loader.load_movies()
        
        assert len(movies_df) == 2
        assert 'movieId' in movies_df.columns
        assert 'title' in movies_df.columns
        assert 'genres' in movies_df.columns

    def test_load_ratings_success(self, temp_data_dir: Path):
        """Test successful ratings loading"""
        # Create test ratings file
        ratings_file = temp_data_dir / "raw" / "ratings.csv"
        test_data = pd.DataFrame({
            'userId': [1, 2],
            'movieId': [1, 2],
            'rating': [4.0, 3.5],
            'timestamp': [1234567890, 1234567891]
        })
        test_data.to_csv(ratings_file, index=False)

        loader = DataLoader(temp_data_dir)
        ratings_df = loader.load_ratings()
        
        assert len(ratings_df) == 2
        assert 'userId' in ratings_df.columns
        assert 'movieId' in ratings_df.columns
        assert 'rating' in ratings_df.columns
        assert 'timestamp' in ratings_df.columns

    def test_validate_data_valid(self, sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame):
        """Test data validation with valid data"""
        loader = DataLoader(Path("/tmp"))
        
        # Should not raise any exceptions
        loader.validate_data(sample_movies_data, sample_ratings_data)

    def test_validate_data_missing_columns(self, temp_data_dir: Path):
        """Test data validation with missing columns"""
        loader = DataLoader(temp_data_dir)
        
        # Missing required columns
        invalid_movies = pd.DataFrame({'id': [1, 2]})
        invalid_ratings = pd.DataFrame({'user': [1, 2]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.validate_data(invalid_movies, invalid_ratings)

    def test_validate_data_empty_dataframes(self, temp_data_dir: Path):
        """Test data validation with empty dataframes"""
        loader = DataLoader(temp_data_dir)
        
        empty_movies = pd.DataFrame(columns=['movieId', 'title', 'genres'])
        empty_ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
        
        with pytest.raises(ValueError, match="Empty dataframes"):
            loader.validate_data(empty_movies, empty_ratings)

    @pytest.mark.parametrize("invalid_rating", [-1, 6, 10.5])
    def test_validate_data_invalid_ratings(self, temp_data_dir: Path, invalid_rating: float):
        """Test data validation with invalid rating values"""
        loader = DataLoader(temp_data_dir)
        
        movies_df = pd.DataFrame({
            'movieId': [1],
            'title': ['Test Movie'],
            'genres': ['Action']
        })
        
        ratings_df = pd.DataFrame({
            'userId': [1],
            'movieId': [1],
            'rating': [invalid_rating],
            'timestamp': [1234567890]
        })
        
        with pytest.raises(ValueError, match="Invalid rating values"):
            loader.validate_data(movies_df, ratings_df)