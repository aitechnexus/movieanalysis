"""Pytest configuration and shared fixtures"""

import os
import tempfile
from pathlib import Path
from typing import Generator, Tuple

import pandas as pd
import pytest

from src.analyzer import MovieAnalyzer
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.visualizer import InsightsVisualizer


@pytest.fixture(scope="session")
def sample_movies_data() -> pd.DataFrame:
    """Create sample movies data for testing"""
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "title": [
                "Toy Story (1995)",
                "Jumanji (1995)",
                "Grumpier Old Men (1995)",
                "Waiting to Exhale (1995)",
                "Father of the Bride Part II (1995)",
            ],
            "genres": [
                "Adventure|Animation|Children|Comedy|Fantasy",
                "Adventure|Children|Fantasy",
                "Comedy|Romance",
                "Comedy|Drama|Romance",
                "Comedy",
            ],
        }
    )


@pytest.fixture(scope="session")
def sample_ratings_data() -> pd.DataFrame:
    """Create sample ratings data for testing"""
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 5,
            "movieId": [1, 2, 1, 3, 2, 4, 3, 5, 4, 1] * 5,
            "rating": [4.0, 3.5, 5.0, 2.5, 4.5, 3.0, 4.0, 3.5, 2.0, 4.5] * 5,
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-01-06",
                    "2020-01-07",
                    "2020-01-08",
                    "2020-01-09",
                    "2020-01-10",
                ]
                * 5
            ),
        }
    )


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        (data_dir / "raw").mkdir(parents=True)
        (data_dir / "processed").mkdir(parents=True)
        yield data_dir


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        (output_dir / "plots").mkdir(parents=True)
        (output_dir / "reports").mkdir(parents=True)
        yield output_dir


@pytest.fixture
def data_processor() -> DataProcessor:
    """Create DataProcessor instance"""
    return DataProcessor()


@pytest.fixture
def movie_analyzer(
    sample_movies_data: pd.DataFrame, sample_ratings_data: pd.DataFrame
) -> MovieAnalyzer:
    """Create MovieAnalyzer instance with sample data"""
    return MovieAnalyzer(sample_movies_data, sample_ratings_data)


@pytest.fixture
def visualizer(temp_output_dir: Path) -> InsightsVisualizer:
    """Create InsightsVisualizer instance"""
    return InsightsVisualizer(temp_output_dir / "plots")


@pytest.fixture
def data_loader(temp_data_dir: Path) -> DataLoader:
    """Create DataLoader instance"""
    return DataLoader(temp_data_dir, source="grouplens", dataset="ml-latest-small")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Set test environment variables
    os.environ["TESTING"] = "1"
    os.environ["LOG_LEVEL"] = "ERROR"

    yield

    # Cleanup
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if "LOG_LEVEL" in os.environ:
        del os.environ["LOG_LEVEL"]


@pytest.fixture
def client():
    """Create Flask test client"""
    import app

    app.app.config["TESTING"] = True
    with app.app.test_client() as client:
        yield client
