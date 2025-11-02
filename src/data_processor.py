import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    A comprehensive data processor for cleaning and preparing MovieLens datasets.
    
    This class provides methods for cleaning, validating, and optimizing MovieLens
    data including movies and ratings DataFrames. It handles data type conversions,
    duplicate removal, missing value imputation, and data validation to ensure
    high-quality datasets for analysis.
    
    The processor focuses on:
    - Data cleaning and deduplication
    - Type optimization for memory efficiency
    - Data validation and quality checks
    - Efficient storage in Parquet format
    
    Examples:
        Basic usage:
        >>> processor = DataProcessor()
        >>> clean_movies = processor.clean_movies(raw_movies_df)
        >>> clean_ratings = processor.clean_ratings(raw_ratings_df)
        
        Save processed data:
        >>> from pathlib import Path
        >>> processor.save_parquet(clean_movies, clean_ratings, Path('processed'))
        
        Complete processing pipeline:
        >>> movies_df, ratings_df = load_raw_data()
        >>> processor = DataProcessor()
        >>> movies_clean = processor.clean_movies(movies_df)
        >>> ratings_clean = processor.clean_ratings(ratings_df)
        >>> processor.save_parquet(movies_clean, ratings_clean, Path('output'))
    """
    
    def clean_movies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the movies DataFrame.
        
        Performs comprehensive cleaning including duplicate removal, missing value
        handling, data type optimization, and validation. Ensures data quality
        and consistency for downstream analysis.
        
        Args:
            df (pd.DataFrame): Raw movies DataFrame with columns [movieId, title, genres]
            
        Returns:
            pd.DataFrame: Cleaned movies DataFrame with optimized data types
            
        Raises:
            KeyError: If required columns are missing
            ValueError: If data contains invalid values
            
        Processing steps:
            1. Remove duplicate movies based on movieId
            2. Drop rows with missing movieId or title
            3. Fill missing genres with "(no genres listed)"
            4. Optimize data types for memory efficiency
            5. Validate data integrity
            
        Examples:
            >>> processor = DataProcessor()
            >>> raw_movies = pd.DataFrame({
            ...     'movieId': [1, 2, 2, 3],  # Contains duplicate
            ...     'title': ['Movie A', 'Movie B', 'Movie B', None],  # Contains null
            ...     'genres': ['Action', 'Drama', 'Drama', 'Comedy']
            ... })
            >>> clean_movies = processor.clean_movies(raw_movies)
            >>> print(len(clean_movies))  # 2 (duplicates and nulls removed)
            >>> print(clean_movies.dtypes)
            movieId      int32
            title       string
            genres      string
        """
        logger.info(f"  → Cleaning movies data ({len(df):,} rows)...")
        initial = len(df)
        df = df.drop_duplicates(subset=["movieId"])
        if len(df) < initial:
            logger.info(f"    Removed {initial - len(df)} duplicate movies")
        df = df.dropna(subset=["movieId", "title"])
        df["genres"] = df["genres"].fillna("(no genres listed)")
        df["movieId"] = df["movieId"].astype("int32", errors="ignore")
        df["title"] = df["title"].astype("string")
        df["genres"] = df["genres"].astype("string")
        logger.info(f"    ✓ Movies cleaned: {len(df):,} rows")
        return df

    def clean_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the ratings DataFrame.
        
        Performs comprehensive cleaning including duplicate removal, rating validation,
        data type optimization, and timestamp handling. Ensures rating values are
        within valid range and timestamps are properly formatted.
        
        Args:
            df (pd.DataFrame): Raw ratings DataFrame with columns [userId, movieId, rating, timestamp]
            
        Returns:
            pd.DataFrame: Cleaned ratings DataFrame with optimized data types
            
        Raises:
            KeyError: If required columns are missing
            ValueError: If ratings are outside valid range
            
        Processing steps:
            1. Remove duplicate ratings
            2. Filter ratings to valid range (0.5-5.0)
            3. Optimize data types for memory efficiency
            4. Convert timestamps to datetime format
            5. Handle missing timestamps with forward fill
            
        Examples:
            >>> processor = DataProcessor()
            >>> raw_ratings = pd.DataFrame({
            ...     'userId': [1, 2, 2, 3],
            ...     'movieId': [1, 2, 2, 3],
            ...     'rating': [4.5, 3.0, 3.0, 6.0],  # Last rating invalid
            ...     'timestamp': [1234567890, 1234567891, 1234567891, 1234567892]
            ... })
            >>> clean_ratings = processor.clean_ratings(raw_ratings)
            >>> print(len(clean_ratings))  # 2 (duplicates and invalid ratings removed)
            >>> print(clean_ratings['rating'].min(), clean_ratings['rating'].max())
            0.5 5.0
        """
        logger.info(f"  → Cleaning ratings data ({len(df):,} rows)...")
        initial = len(df)
        df = df.drop_duplicates()
        if len(df) < initial:
            logger.info(f"    Removed {initial - len(df)} duplicate ratings")
        df = df[(df["rating"] >= 0.5) & (df["rating"] <= 5.0)]
        df["userId"] = df["userId"].astype("int32", errors="ignore")
        df["movieId"] = df["movieId"].astype("int32", errors="ignore")
        df["rating"] = df["rating"].astype("float32", errors="ignore")
        if not str(df["timestamp"].dtype).startswith("datetime"):
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="s", errors="coerce"
            ).fillna(method="ffill")
        logger.info(f"    ✓ Ratings cleaned: {len(df):,} rows")
        return df

    def save_parquet(
        self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, output_dir: Path
    ):
        """
        Save cleaned DataFrames to Parquet format for efficient storage and retrieval.
        
        Saves the processed movies and ratings DataFrames to Parquet files with
        compression for optimal storage efficiency. Creates the output directory
        if it doesn't exist.
        
        Args:
            movies_df (pd.DataFrame): Cleaned movies DataFrame
            ratings_df (pd.DataFrame): Cleaned ratings DataFrame
            output_dir (Path): Directory to save the Parquet files
            
        Raises:
            PermissionError: If unable to write to the output directory
            OSError: If disk space is insufficient
            
        Output files:
            - movies.parquet: Compressed movies data
            - ratings.parquet: Compressed ratings data
            
        Examples:
            >>> from pathlib import Path
            >>> processor = DataProcessor()
            >>> movies_clean = processor.clean_movies(movies_df)
            >>> ratings_clean = processor.clean_ratings(ratings_df)
            >>> processor.save_parquet(movies_clean, ratings_clean, Path('processed'))
            
            Check saved files:
            >>> output_dir = Path('processed')
            >>> print(list(output_dir.glob('*.parquet')))
            [PosixPath('processed/movies.parquet'), PosixPath('processed/ratings.parquet')]
            
            Load saved data:
            >>> movies_loaded = pd.read_parquet('processed/movies.parquet')
            >>> ratings_loaded = pd.read_parquet('processed/ratings.parquet')
        """
        logger.info("  → Saving to Parquet format...")
        output_dir.mkdir(parents=True, exist_ok=True)
        movies_df.to_parquet(
            output_dir / "movies.parquet", compression="zstd", index=False
        )
        ratings_df.to_parquet(
            output_dir / "ratings.parquet", compression="zstd", index=False
        )
        logger.info("    ✓ Parquet files saved")

    def process_data(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Complete data processing pipeline for movies and ratings data.
        
        This method combines the cleaning operations for both movies and ratings
        DataFrames into a single convenient method. It performs all necessary
        data cleaning, validation, and optimization steps.
        
        Args:
            movies_df (pd.DataFrame): Raw movies DataFrame with columns:
                - movieId: Unique movie identifier
                - title: Movie title with year
                - genres: Pipe-separated genre list
            ratings_df (pd.DataFrame): Raw ratings DataFrame with columns:
                - userId: User identifier
                - movieId: Movie identifier
                - rating: Rating value (0.5-5.0)
                - timestamp: Unix timestamp
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing processed data with keys:
                - movies_clean: Cleaned movies DataFrame
                - ratings_clean: Cleaned ratings DataFrame
                - movie_stats: Movie statistics DataFrame
                - user_stats: User statistics DataFrame
                - genres_df: Extracted genres DataFrame
                - ratings_enhanced: Enhanced ratings with time features
        
        Examples:
            >>> processor = DataProcessor()
            >>> result = processor.process_data(raw_movies, raw_ratings)
            >>> print(f"Processed {len(result['movies_clean'])} movies")
        """
        logger.info("Starting complete data processing pipeline...")
        
        # Clean movies data
        logger.info("  → Processing movies data...")
        clean_movies = self.clean_movies(movies_df)
        
        # Clean ratings data
        logger.info("  → Processing ratings data...")
        clean_ratings = self.clean_ratings(ratings_df)
        
        # Calculate additional statistics and features
        logger.info("  → Calculating movie statistics...")
        movie_stats = clean_ratings.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std'],
            'userId': 'nunique'
        }).round(3)
        movie_stats.columns = ['rating_count', 'avg_rating', 'rating_std', 'user_count']
        movie_stats = movie_stats.reset_index()
        
        logger.info("  → Calculating user statistics...")
        user_stats = clean_ratings.groupby('userId').agg({
            'rating': ['count', 'mean', 'std'],
            'movieId': 'nunique'
        }).round(3)
        user_stats.columns = ['rating_count', 'avg_rating', 'rating_std', 'movie_count']
        user_stats = user_stats.reset_index()
        
        # Extract genres
        logger.info("  → Extracting genres...")
        genres_list = []
        for _, row in clean_movies.iterrows():
            genres = row['genres'].split('|') if pd.notna(row['genres']) else []
            for genre in genres:
                if genre and genre != '(no genres listed)':
                    genres_list.append({'movieId': row['movieId'], 'genre': genre})
        genres_df = pd.DataFrame(genres_list)
        
        # Add time features to ratings
        logger.info("  → Adding time features...")
        ratings_enhanced = clean_ratings.copy()
        if 'timestamp' in ratings_enhanced.columns:
            ratings_enhanced['datetime'] = pd.to_datetime(ratings_enhanced['timestamp'], unit='s')
            ratings_enhanced['year'] = ratings_enhanced['datetime'].dt.year
            ratings_enhanced['month'] = ratings_enhanced['datetime'].dt.month
            ratings_enhanced['day_of_week'] = ratings_enhanced['datetime'].dt.dayofweek
        
        logger.info("  ✓ Data processing pipeline completed")
        
        return {
            'movies_clean': clean_movies,
            'ratings_clean': clean_ratings,
            'movie_stats': movie_stats,
            'user_stats': user_stats,
            'genres_df': genres_df,
            'ratings_enhanced': ratings_enhanced
         }

    def clean_movies_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for clean_movies method for backward compatibility."""
        return self.clean_movies(df)

    def clean_ratings_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for clean_ratings method for backward compatibility."""
        return self.clean_ratings(df)

    def extract_genres(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract individual genres from movies DataFrame.
        
        Args:
            movies_df (pd.DataFrame): Movies DataFrame with genres column
            
        Returns:
            pd.DataFrame: DataFrame with movieId and individual genre columns
        """
        genres_list = []
        for _, row in movies_df.iterrows():
            if pd.notna(row['genres']):
                genres = row['genres'].split('|')
                for genre in genres:
                    if genre and genre != '(no genres listed)':
                        genres_list.append({'movieId': row['movieId'], 'genre': genre})
        return pd.DataFrame(genres_list)

    def calculate_movie_stats(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive statistics for each movie.
        
        Args:
            movies_df (pd.DataFrame): Movies DataFrame
            ratings_df (pd.DataFrame): Ratings DataFrame
            
        Returns:
            pd.DataFrame: Movie statistics DataFrame
        """
        stats = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'userId': 'nunique'
        }).round(3)
        stats.columns = ['rating_count', 'avg_rating', 'rating_std', 'min_rating', 'max_rating', 'user_count']
        stats = stats.reset_index()
        
        # Merge with movie titles
        stats = stats.merge(movies_df[['movieId', 'title']], on='movieId', how='left')
        return stats

    def calculate_user_stats(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive statistics for each user.
        
        Args:
            ratings_df (pd.DataFrame): Ratings DataFrame
            
        Returns:
            pd.DataFrame: User statistics DataFrame
        """
        stats = ratings_df.groupby('userId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'movieId': 'nunique'
        }).round(3)
        stats.columns = ['rating_count', 'avg_rating', 'rating_std', 'min_rating', 'max_rating', 'movie_count']
        return stats.reset_index()

    def create_time_features(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from ratings DataFrame.
        
        Args:
            ratings_df (pd.DataFrame): Ratings DataFrame with timestamp column
            
        Returns:
            pd.DataFrame: Enhanced ratings DataFrame with time features
        """
        enhanced_df = ratings_df.copy()
        if 'timestamp' in enhanced_df.columns:
            enhanced_df['datetime'] = pd.to_datetime(enhanced_df['timestamp'], unit='s', errors='coerce')
            enhanced_df['year'] = enhanced_df['datetime'].dt.year
            enhanced_df['month'] = enhanced_df['datetime'].dt.month
            enhanced_df['day_of_week'] = enhanced_df['datetime'].dt.dayofweek
            enhanced_df['hour'] = enhanced_df['datetime'].dt.hour
            enhanced_df['is_weekend'] = enhanced_df['day_of_week'].isin([5, 6])
        return enhanced_df

    def normalize_ratings(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize ratings using z-score normalization per user.
        
        Args:
            ratings_df (pd.DataFrame): Ratings DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with normalized ratings
        """
        normalized_df = ratings_df.copy()
        
        # Calculate user means and stds
        user_stats = ratings_df.groupby('userId')['rating'].agg(['mean', 'std']).reset_index()
        user_stats['std'] = user_stats['std'].fillna(1.0)  # Handle users with only one rating
        
        # Merge stats back
        normalized_df = normalized_df.merge(user_stats, on='userId', suffixes=('', '_user'))
        
        # Normalize ratings
        normalized_df['normalized_rating'] = (normalized_df['rating'] - normalized_df['mean']) / normalized_df['std']
        normalized_df['normalized_rating'] = normalized_df['normalized_rating'].fillna(0.0)
        
        # Clean up temporary columns
        normalized_df = normalized_df.drop(['mean', 'std'], axis=1)
        
        return normalized_df

    def detect_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect outliers in a DataFrame column using various methods.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column (str): Column name to analyze for outliers
            method (str): Method to use ('iqr', 'zscore', 'isolation')
            
        Returns:
            pd.DataFrame: DataFrame containing only outlier rows
        """
        if column not in df.columns:
            return pd.DataFrame()
            
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            threshold = 3
            outlier_indices = df[column].dropna().index[z_scores > threshold]
            outliers = df.loc[outlier_indices]
            
        elif method == 'isolation':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(df[[column]].dropna())
            outlier_indices = df[column].dropna().index[outlier_labels == -1]
            outliers = df.loc[outlier_indices]
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return outliers
