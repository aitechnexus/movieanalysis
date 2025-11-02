import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    GROUPLENS_URLS = {
        "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    }

    def __init__(
        self, data_dir: Path, source: str = "grouplens", dataset: str = "ml-25m"
    ):
        self.data_dir, self.source, self.dataset = data_dir, source, dataset
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"

    def load_or_download(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
