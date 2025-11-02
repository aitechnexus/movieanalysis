import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    A comprehensive data loader for MovieLens datasets supporting multiple sources and formats.
    
    This class handles downloading, caching, and loading of MovieLens datasets from various
    sources including GroupLens and HuggingFace. It provides automatic caching mechanisms
    to avoid repeated downloads and supports multiple dataset sizes.
    
    Attributes:
        GROUPLENS_URLS (dict): Mapping of dataset names to their download URLs
        data_dir (Path): Root directory for data storage
        source (str): Data source ('grouplens' or 'huggingface')
        dataset (str): Dataset identifier (e.g., 'ml-25m', 'ml-latest-small')
        raw_dir (Path): Directory for raw downloaded data
        processed_dir (Path): Directory for processed/cached data
    
    Examples:
        Basic usage with GroupLens data:
        >>> loader = DataLoader(Path('data'), source='grouplens', dataset='ml-latest-small')
        >>> movies_df, ratings_df = loader.load_or_download()
        >>> print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
        
        Using HuggingFace datasets:
        >>> loader = DataLoader(Path('data'), source='huggingface')
        >>> movies_df, ratings_df = loader.load_or_download()
        
        Custom data directory:
        >>> from pathlib import Path
        >>> custom_dir = Path('/path/to/custom/data')
        >>> loader = DataLoader(custom_dir, dataset='ml-25m')
        >>> movies_df, ratings_df = loader.load_or_download()
    """
    
    GROUPLENS_URLS = {
        "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    }

    def __init__(
        self, data_dir: Path, source: str = "grouplens", dataset: str = "ml-25m"
    ):
        """
        Initialize the DataLoader with specified configuration.
        
        Args:
            data_dir (Path): Root directory for data storage. Will be created if it doesn't exist.
            source (str, optional): Data source to use. Options: 'grouplens', 'huggingface'. 
                                  Defaults to 'grouplens'.
            dataset (str, optional): Dataset identifier. For GroupLens: 'ml-25m', 'ml-latest-small'.
                                   Defaults to 'ml-25m'.
        
        Raises:
            ValueError: If an unsupported source or dataset is specified.
            
        Examples:
            >>> loader = DataLoader(Path('data'))  # Default: GroupLens ml-25m
            >>> loader = DataLoader(Path('data'), source='huggingface')
            >>> loader = DataLoader(Path('data'), dataset='ml-latest-small')
        """
        self.data_dir, self.source, self.dataset = data_dir, source, dataset
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"

    def load_or_download(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load MovieLens data from cache or download if not available.
        
        This method first checks for cached Parquet files in the processed directory.
        If found, it loads from cache for faster access. Otherwise, it downloads
        and processes the data from the specified source.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - movies_df: DataFrame with columns [movieId, title, genres]
                - ratings_df: DataFrame with columns [userId, movieId, rating, timestamp]
        
        Raises:
            ValueError: If the specified source or dataset is not supported.
            ImportError: If required dependencies are missing (e.g., datasets for HuggingFace).
            ConnectionError: If download fails due to network issues.
            
        Examples:
            >>> loader = DataLoader(Path('data'))
            >>> movies_df, ratings_df = loader.load_or_download()
            >>> print(f"Movies shape: {movies_df.shape}")
            >>> print(f"Ratings shape: {ratings_df.shape}")
            >>> print(f"Columns: {movies_df.columns.tolist()}")
            
            Check data types:
            >>> print(ratings_df.dtypes)
            userId       int64
            movieId      int64
            rating     float64
            timestamp   datetime64[ns]
        """
        movies_parquet = self.processed_dir / "movies.parquet"
        ratings_parquet = self.processed_dir / "ratings.parquet"
        if movies_parquet.exists() and ratings_parquet.exists():
            logger.info("Loading from existing Parquet files...")
            return pd.read_parquet(movies_parquet), pd.read_parquet(ratings_parquet)
        if self.source == "grouplens":
            return self._load_from_grouplens()
        elif self.source == "huggingface":
            return self._load_from_huggingface()
        else:
            raise ValueError(f"Unknown source: {self.source}")

    def _load_from_grouplens(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download and load data from GroupLens MovieLens repository.
        
        Downloads the specified dataset from GroupLens, extracts the ZIP file,
        and loads the CSV files into pandas DataFrames. Automatically converts
        timestamp columns to datetime format.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Movies and ratings DataFrames
            
        Raises:
            ValueError: If the dataset name is not recognized
            urllib.error.URLError: If download fails
            zipfile.BadZipFile: If the downloaded file is corrupted
            
        Note:
            This method automatically creates necessary directories and cleans up
            temporary ZIP files after extraction.
        """
        logger.info(f"Downloading {self.dataset} from GroupLens...")
        url = self.GROUPLENS_URLS.get(self.dataset)
        if not url:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        zip_path = self.raw_dir / f"{self.dataset}.zip"
        extract_dir = self.raw_dir / self.dataset
        if not extract_dir.exists():
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(self.raw_dir)
            zip_path.unlink(missing_ok=True)
        movies_df = pd.read_csv(extract_dir / "movies.csv")
        ratings_df = pd.read_csv(extract_dir / "ratings.csv")
        ratings_df["timestamp"] = pd.to_datetime(ratings_df["timestamp"], unit="s")
        return movies_df, ratings_df

    def _load_from_huggingface(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data from HuggingFace datasets repository.
        
        Uses the HuggingFace datasets library to load MovieLens data. This method
        provides an alternative data source and handles data format conversion
        to match the expected schema.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Movies and ratings DataFrames
            
        Raises:
            ImportError: If the 'datasets' library is not installed
            ConnectionError: If unable to connect to HuggingFace
            
        Note:
            Requires 'datasets' library: pip install datasets
            Handles missing timestamps by filling with current time.
            
        Examples:
            >>> # Ensure datasets library is installed
            >>> # pip install datasets
            >>> loader = DataLoader(Path('data'), source='huggingface')
            >>> movies_df, ratings_df = loader._load_from_huggingface()
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install: pip install datasets")
        ds = load_dataset("reczoo/Movielens1M_m1")
        df = ds["train"].to_pandas()
        movies_df = df[["movieId", "title", "genres"]].drop_duplicates()
        ratings_df = df[["userId", "movieId", "rating", "timestamp"]]
        ratings_df["timestamp"] = pd.to_datetime(
            ratings_df["timestamp"], unit="s", errors="coerce"
        )
        ratings_df["timestamp"] = ratings_df["timestamp"].fillna(pd.Timestamp.now())
        return movies_df, ratings_df

    def load_movies(self) -> pd.DataFrame:
        """
        Load only the movies DataFrame from the dataset.
        
        This method provides a convenient way to load just the movies data
        without loading the ratings data, which can be useful for certain
        analysis tasks that only require movie information.
        
        Returns:
            pd.DataFrame: Movies DataFrame with columns:
                - movieId: Unique movie identifier
                - title: Movie title with year
                - genres: Pipe-separated genre list
        
        Examples:
            >>> loader = DataLoader(Path('data'))
            >>> movies_df = loader.load_movies()
            >>> print(f"Loaded {len(movies_df)} movies")
        """
        logger.info("Loading movies data only...")
        movies_df, _ = self.load_or_download()
        logger.info(f"  ✓ Loaded {len(movies_df)} movies")
        return movies_df

    def load_ratings(self) -> pd.DataFrame:
        """
        Load only the ratings DataFrame from the dataset.
        
        This method provides a convenient way to load just the ratings data
        without loading the movies data, which can be useful for certain
        analysis tasks that only require rating information.
        
        Returns:
            pd.DataFrame: Ratings DataFrame with columns:
                - userId: User identifier
                - movieId: Movie identifier
                - rating: Rating value (0.5-5.0)
                - timestamp: Unix timestamp
        
        Examples:
            >>> loader = DataLoader(Path('data'))
            >>> ratings_df = loader.load_ratings()
            >>> print(f"Loaded {len(ratings_df)} ratings")
        """
        logger.info("Loading ratings data only...")
        _, ratings_df = self.load_or_download()
        logger.info(f"  ✓ Loaded {len(ratings_df)} ratings")
        return ratings_df
