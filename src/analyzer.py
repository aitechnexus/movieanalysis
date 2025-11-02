import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MovieAnalyzer:
    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.movies, self.ratings, self._global_mean = movies_df, ratings_df, None

    @property
    def global_mean(self) -> float:
        if self._global_mean is None:
            self._global_mean = float(self.ratings["rating"].mean())
        return self._global_mean

    def get_top_movies(
        self, limit: int = 20, min_ratings: int = 100, m_percentile: float = 0.7
    ) -> List[Dict[str, Any]]:
        logger.info(
            f"    Calculating top {limit} movies (min_ratings={min_ratings})..."
        )
        stats = (
            self.ratings.groupby("movieId")
            .agg(
                v=("rating", "count"),
                R=("rating", "mean"),
                rating_std=("rating", "std"),
            )
            .reset_index()
        )
        stats = stats[stats["v"] >= min_ratings]
        if stats.empty:
            return []
        m, C = float(stats["v"].quantile(m_percentile)), self.global_mean
        stats["WR"] = (stats["v"] / (stats["v"] + m)) * stats["R"] + (
            m / (stats["v"] + m)
        ) * C
        z, p_hat, n = 1.96, stats["R"] / 5.0, stats["v"]
        stats["wilson_lower"] = (
            (
                p_hat
                + z**2 / (2 * n)
                - z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
            )
            / (1 + z**2 / n)
            * 5.0
        )
        top = stats.nlargest(limit, "WR").merge(self.movies, on="movieId")
        result = []
        for _, r in top.iterrows():
            result.append(
                {
                    "movieId": int(r["movieId"]),
                    "title": str(r["title"]),
                    "genres": str(r["genres"]),
                    "rating_mean": round(float(r["R"]), 3),
                    "rating_std": (
                        round(float(r["rating_std"]), 3)
                        if not pd.isna(r["rating_std"])
                        else 0.0
                    ),
                    "vote_count": int(r["v"]),
                    "weighted_rating": round(float(r["WR"]), 3),
                    "wilson_lower": round(float(r["wilson_lower"]), 3),
                    "m": round(m, 2),
                    "C": round(C, 3),
                }
            )
        return result

    def analyze_genre_trends(self) -> Dict[str, Any]:
        logger.info("    Analyzing genre trends...")
        df = self.ratings.merge(self.movies, on="movieId")
        df["genres_list"] = df["genres"].str.split("|")
        df = df.explode("genres_list")
        df = df[df["genres_list"] != "(no genres listed)"]
        overall = (
            df.groupby("genres_list")
            .agg(
                count=("rating", "count"),
                mean_rating=("rating", "mean"),
                std_rating=("rating", "std"),
                median_rating=("rating", "median"),
            )
            .reset_index()
            .rename(columns={"genres_list": "genre"})
            .sort_values("count", ascending=False)
        )
        df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
        time_series = (
            df.groupby(["year_month", "genres_list"])
            .agg(count=("rating", "count"), mean_rating=("rating", "mean"))
            .reset_index()
            .rename(columns={"genres_list": "genre"})
        )
        return {
            "overall": overall.to_dict("records"),
            "time_series": time_series.to_dict("records"),
            "top_genres": overall.head(10).to_dict("records"),
        }

    def generate_time_series_analysis(self) -> Dict[str, Any]:
        logger.info("    Performing time-series analysis...")
        df = self.ratings.copy()
        df["year_month"] = df["timestamp"].dt.to_period("M")
        monthly = (
            df.groupby("year_month")
            .agg(
                count=("rating", "count"),
                mean_rating=("rating", "mean"),
                std_rating=("rating", "std"),
                unique_users=("userId", "nunique"),
                unique_movies=("movieId", "nunique"),
            )
            .reset_index()
        )
        monthly["year_month"] = monthly["year_month"].astype(str)
        df["year"] = df["timestamp"].dt.year
        yearly = (
            df.groupby("year")
            .agg(count=("rating", "count"), mean_rating=("rating", "mean"))
            .reset_index()
        )
        return {
            "monthly": monthly.to_dict("records"),
            "yearly": yearly.to_dict("records"),
            "summary": {
                "total_months": len(monthly),
                "peak_month": (
                    monthly.loc[monthly["count"].idxmax(), "year_month"]
                    if len(monthly)
                    else None
                ),
                "peak_count": int(monthly["count"].max()) if len(monthly) else 0,
            },
        }

    def get_rating_distribution(self) -> Dict[str, Any]:
        logger.info("    Computing rating distribution...")
        counts = self.ratings["rating"].value_counts().sort_index()
        return {
            "distribution": {float(k): int(v) for k, v in counts.to_dict().items()},
            "statistics": {
                "mean": float(self.ratings["rating"].mean()),
                "median": float(self.ratings["rating"].median()),
                "std": float(self.ratings["rating"].std()),
                "min": float(self.ratings["rating"].min()),
                "max": float(self.ratings["rating"].max()),
                "q25": float(self.ratings["rating"].quantile(0.25)),
                "q75": float(self.ratings["rating"].quantile(0.75)),
            },
        }

    def get_user_behavior_stats(self) -> Dict[str, Any]:
        logger.info("    Analyzing user behavior...")
        us = self.ratings.groupby("userId").agg(
            n_ratings=("rating", "count"),
            mean_rating=("rating", "mean"),
            std_rating=("rating", "std"),
        )
        return {
            "total_users": int(self.ratings["userId"].nunique()),
            "avg_ratings_per_user": float(us["n_ratings"].mean()),
            "median_ratings_per_user": float(us["n_ratings"].median()),
            "user_activity_distribution": {
                "light": int((us["n_ratings"] < 20).sum()),
                "moderate": int(
                    ((us["n_ratings"] >= 20) & (us["n_ratings"] < 100)).sum()
                ),
                "heavy": int((us["n_ratings"] >= 100).sum()),
            },
            "rating_variance_mean": float(us["std_rating"].mean(skipna=True)),
        }
