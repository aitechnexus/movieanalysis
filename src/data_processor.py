import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    def clean_movies(self, df: pd.DataFrame) -> pd.DataFrame:
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
        logger.info("  → Saving to Parquet format...")
        output_dir.mkdir(parents=True, exist_ok=True)
        movies_df.to_parquet(
            output_dir / "movies.parquet", compression="zstd", index=False
        )
        ratings_df.to_parquet(
            output_dir / "ratings.parquet", compression="zstd", index=False
        )
        logger.info("    ✓ Parquet files saved")
