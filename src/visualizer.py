import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.size"] = 10


class InsightsVisualizer:
    """
    Advanced visualization engine for MovieLens analysis insights.
    
    Creates publication-ready visualizations including statistical plots,
    trend analyses, heatmaps, and comprehensive dashboards. Supports
    multiple output formats and customizable styling options.
    
    Features:
    - Rating distribution and statistical analysis plots
    - Top movies and genre popularity visualizations
    - Time-series trend analysis and temporal patterns
    - User behavior and activity heatmaps
    - Correlation analysis and advanced statistical plots
    - Comprehensive dashboard generation
    
    Attributes:
        output_dir (Path): Directory for saving generated visualizations
        
    Examples:
        Basic usage:
        >>> from pathlib import Path
        >>> visualizer = InsightsVisualizer(Path('outputs/plots'))
        >>> 
        >>> # Generate individual plots
        >>> rating_plot = visualizer.plot_rating_distribution(rating_data)
        >>> genre_plot = visualizer.plot_genre_popularity(genre_data)
        >>> 
        >>> # Create comprehensive dashboard
        >>> dashboard = visualizer.plot_comprehensive_statistics(analyzer)
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize the visualizer with output directory configuration.
        
        Args:
            output_dir (Path): Directory to save generated visualizations
            
        Examples:
            >>> from pathlib import Path
            >>> visualizer = InsightsVisualizer(Path('outputs/visualizations'))
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_rating_distribution(self, rating_dist: Dict[str, Any]) -> str:
        logger.info("    → Generating rating distribution plot...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        distribution = rating_dist["distribution"]
        ratings, counts = list(distribution.keys()), list(distribution.values())
        axes[0].bar(ratings, counts, color="steelblue", edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Rating")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Rating Distribution")
        axes[0].grid(axis="y", alpha=0.3)
        for r, c in zip(ratings, counts):
            axes[0].text(r, c, f"{c:,}", ha="center", va="bottom", fontsize=9)
        s = rating_dist["statistics"]
        stats_text = f"""
Mean: {s['mean']:.3f}
Median: {s['median']:.3f}
Std Dev: {s['std']:.3f}

Min: {s['min']:.1f}
Max: {s['max']:.1f}

Q1: {s['q25']:.3f}
Q3: {s['q75']:.3f}
"""
        axes[1].text(
            0.1,
            0.5,
            stats_text,
            fontsize=12,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        axes[1].axis("off")
        axes[1].set_title("Statistics Summary")
        plt.tight_layout()
        path = self.output_dir / "rating_distribution.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_top_movies(self, top_movies: List[Dict[str, Any]]) -> str:
        logger.info("    → Generating top movies plot...")
        import numpy as np
        import pandas as pd

        df = pd.DataFrame(top_movies).head(15)
        if df.empty:
            path = self.output_dir / "top_movies.png"
            plt.figure()
            plt.text(0.5, 0.5, "No movies meet the threshold.", ha="center")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            return str(path)
        fig, ax = plt.subplots(figsize=(14, 8))
        y_pos = np.arange(len(df))
        ax.barh(y_pos, df["weighted_rating"], color="coral", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [
                f"{row['title'][:40]}..." if len(row["title"]) > 40 else row["title"]
                for _, row in df.iterrows()
            ]
        )
        ax.invert_yaxis()
        ax.set_xlabel("Weighted Rating (WR)")
        ax.set_title("Top 15 Movies by Weighted Rating (IMDb Formula)")
        ax.grid(axis="x", alpha=0.3)
        for i, (_, row) in enumerate(df.iterrows()):
            ax.text(
                row["weighted_rating"] + 0.02,
                i,
                f"{row['weighted_rating']:.2f} ({row['vote_count']:,} votes)",
                va="center",
                fontsize=9,
            )
        plt.tight_layout()
        path = self.output_dir / "top_movies.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_genre_popularity(self, genre_stats: Dict[str, Any]) -> str:
        logger.info("    → Generating genre popularity plot...")
        import pandas as pd

        df = pd.DataFrame(genre_stats["overall"]).head(15)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].barh(df["genre"], df["count"], color="skyblue", alpha=0.8)
        axes[0].set_xlabel("Number of Ratings")
        axes[0].set_title("Top 15 Genres by Number of Ratings")
        axes[0].invert_yaxis()
        axes[0].grid(axis="x", alpha=0.3)
        for i, (_, row) in enumerate(df.iterrows()):
            axes[0].text(row["count"], i, f" {row['count']:,}", va="center", fontsize=9)
        axes[1].scatter(
            df["count"],
            df["mean_rating"],
            s=200,
            c=df["mean_rating"],
            cmap="RdYlGn",
            alpha=0.7,
            edgecolors="black",
        )
        axes[1].set_xlabel("Number of Ratings (log scale)")
        axes[1].set_ylabel("Average Rating")
        axes[1].set_xscale("log")
        axes[1].set_title("Genre Rating vs Popularity")
        axes[1].grid(True, alpha=0.3)
        for _, row in df.head(5).iterrows():
            axes[1].annotate(
                row["genre"],
                (row["count"], row["mean_rating"]),
                xytext=(10, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )
        plt.tight_layout()
        path = self.output_dir / "genre_popularity.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_time_series(self, time_series: Dict[str, Any]) -> str:
        logger.info("    → Generating time-series plot...")
        import pandas as pd

        monthly_df = pd.DataFrame(time_series["monthly"])
        if monthly_df.empty:
            path = self.output_dir / "time_series.png"
            plt.figure()
            plt.text(0.5, 0.5, "No data", ha="center")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            return str(path)
        monthly_df["year_month"] = pd.to_datetime(monthly_df["year_month"])
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        axes[0].plot(
            monthly_df["year_month"],
            monthly_df["count"],
            color="steelblue",
            linewidth=2,
        )
        axes[0].fill_between(
            monthly_df["year_month"], monthly_df["count"], alpha=0.3, color="steelblue"
        )
        axes[0].set_ylabel("Number of Ratings")
        axes[0].set_title("Monthly Rating Activity")
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
        axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
        axes[1].plot(
            monthly_df["year_month"],
            monthly_df["mean_rating"],
            color="coral",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        axes[1].axhline(
            monthly_df["mean_rating"].mean(),
            color="red",
            linestyle="--",
            label="Overall Mean",
        )
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Average Rating")
        axes[1].set_title("Average Rating Over Time")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
        axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
        plt.tight_layout()
        path = self.output_dir / "time_series.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_user_activity(self, user_stats: Dict[str, Any]) -> str:
        logger.info("    → Generating user activity plot...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        activity = user_stats["user_activity_distribution"]
        labels = ["Light (<20)", "Moderate (20-100)", "Heavy (>100)"]
        sizes = [activity["light"], activity["moderate"], activity["heavy"]]
        explode = (0.05, 0.05, 0.05)
        axes[0].pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )
        axes[0].set_title("User Activity Distribution")
        stats_text = f"""
Total Users: {user_stats['total_users']:,}

Avg Ratings/User: {user_stats['avg_ratings_per_user']:.1f}
Median Ratings/User: {user_stats['median_ratings_per_user']:.1f}

Light Users: {activity['light']:,}
Moderate Users: {activity['moderate']:,}
Heavy Users: {activity['heavy']:,}
"""
        axes[1].text(
            0.1,
            0.5,
            stats_text,
            fontsize=11,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )
        axes[1].axis("off")
        axes[1].set_title("User Statistics")
        plt.tight_layout()
        path = self.output_dir / "user_activity.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_rating_heatmap(self, ratings_df: pd.DataFrame) -> str:
        logger.info("    → Generating rating heatmap...")
        df = (
            ratings_df.sample(100000, random_state=42)
            if len(ratings_df) > 100000
            else ratings_df.copy()
        )
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.day_name()
        hm = df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
        order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        hm = hm.reindex(order)
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            hm,
            cmap="YlOrRd",
            annot=False,
            fmt="d",
            cbar_kws={"label": "Number of Ratings"},
            ax=ax,
        )
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")
        ax.set_title("Rating Activity Heatmap: Day vs Hour")
        plt.tight_layout()
        path = self.output_dir / "rating_heatmap.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_genre_trends_over_time(self, genre_stats: Dict[str, Any]) -> str:
        logger.info("    → Generating genre trends over time...")
        import pandas as pd

        ts = pd.DataFrame(genre_stats["time_series"])
        if ts.empty:
            path = self.output_dir / "genre_trends_time.png"
            plt.figure()
            plt.text(0.5, 0.5, "No data", ha="center")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            return str(path)
        ts["year_month"] = pd.to_datetime(ts["year_month"])
        top_genres = pd.DataFrame(genre_stats["overall"]).head(8)["genre"].tolist()
        filt = ts[ts["genre"].isin(top_genres)]
        fig, ax = plt.subplots(figsize=(14, 8))
        for g in top_genres:
            gd = filt[filt["genre"] == g]
            ax.plot(
                gd["year_month"],
                gd["count"],
                label=g,
                linewidth=2,
                marker="o",
                markersize=2,
                alpha=0.7,
            )
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Ratings")
        ax.set_title("Top Genres Popularity Over Time")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
        plt.tight_layout()
        path = self.output_dir / "genre_trends_time.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_correlation_analysis(self, analyzer) -> str:
        logger.info("    → Generating correlation analysis...")
        ms = (
            analyzer.ratings.groupby("movieId")
            .agg(
                count=("rating", "count"),
                mean=("rating", "mean"),
                std=("rating", "std"),
                median=("rating", "median"),
            )
            .reset_index()
        )
        ms = ms[ms["count"] >= 50]
        if ms.empty:
            path = self.output_dir / "correlation_matrix.png"
            plt.figure()
            plt.text(0.5, 0.5, "Not enough data", ha="center")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            return str(path)
        corr = ms[["count", "mean", "std", "median"]].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            fmt=".3f",
            ax=ax,
        )
        ax.set_title("Correlation Matrix: Movie Rating Metrics")
        plt.tight_layout()
        path = self.output_dir / "correlation_matrix.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_comprehensive_statistics(self, analyzer) -> str:
        """Generate comprehensive statistical analysis plots"""
        logger.info("    → Generating comprehensive statistical analysis...")

        # Prepare data for statistical analysis
        ratings_df = analyzer.ratings.copy()

        # Calculate movie statistics
        movie_stats = (
            ratings_df.groupby("movieId")
            .agg(
                count=("rating", "count"),
                mean=("rating", "mean"),
                std=("rating", "std"),
                median=("rating", "median"),
                q25=("rating", lambda x: x.quantile(0.25)),
                q75=("rating", lambda x: x.quantile(0.75)),
            )
            .reset_index()
        )

        # Filter movies with sufficient ratings
        movie_stats = movie_stats[movie_stats["count"] >= 20]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Rating Distribution with Statistical Overlay
        axes[0, 0].hist(
            ratings_df["rating"], bins=10, alpha=0.7, color="skyblue", edgecolor="black"
        )
        mean_rating = ratings_df["rating"].mean()
        median_rating = ratings_df["rating"].median()
        std_rating = ratings_df["rating"].std()
        axes[0, 0].axvline(
            mean_rating,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_rating:.2f}",
        )
        axes[0, 0].axvline(
            median_rating,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_rating:.2f}",
        )
        axes[0, 0].axvline(
            mean_rating + std_rating,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label=f"+1σ: {mean_rating + std_rating:.2f}",
        )
        axes[0, 0].axvline(
            mean_rating - std_rating,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label=f"-1σ: {mean_rating - std_rating:.2f}",
        )
        axes[0, 0].set_title("Rating Distribution with Statistical Measures")
        axes[0, 0].set_xlabel("Rating")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Box Plot of Ratings by Rating Value
        rating_groups = [
            ratings_df[ratings_df["rating"] == r]["rating"]
            for r in sorted(ratings_df["rating"].unique())
        ]
        axes[0, 1].boxplot(rating_groups, tick_labels=sorted(ratings_df["rating"].unique()))
        axes[0, 1].set_title("Box Plot: Rating Distribution")
        axes[0, 1].set_xlabel("Rating Value")
        axes[0, 1].set_ylabel("Distribution")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Correlation Heatmap (Enhanced)
        corr_data = movie_stats[["count", "mean", "std", "median", "q25", "q75"]].corr()
        im = axes[0, 2].imshow(corr_data, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        axes[0, 2].set_xticks(range(len(corr_data.columns)))
        axes[0, 2].set_yticks(range(len(corr_data.columns)))
        axes[0, 2].set_xticklabels(corr_data.columns, rotation=45)
        axes[0, 2].set_yticklabels(corr_data.columns)

        # Add correlation values to heatmap
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                axes[0, 2].text(
                    j,
                    i,
                    f"{corr_data.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black" if abs(corr_data.iloc[i, j]) < 0.5 else "white",
                )

        axes[0, 2].set_title("Enhanced Correlation Matrix")
        plt.colorbar(im, ax=axes[0, 2], shrink=0.8)

        # 4. Standard Deviation Analysis
        axes[1, 0].scatter(
            movie_stats["mean"],
            movie_stats["std"],
            alpha=0.6,
            s=movie_stats["count"] / 10,
            c=movie_stats["count"],
            cmap="viridis",
        )
        axes[1, 0].set_xlabel("Mean Rating")
        axes[1, 0].set_ylabel("Standard Deviation")
        axes[1, 0].set_title("Rating Variability Analysis\n(Size = Number of Ratings)")
        axes[1, 0].grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(movie_stats["mean"], movie_stats["std"], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(
            movie_stats["mean"], p(movie_stats["mean"]), "r--", alpha=0.8, linewidth=2
        )

        # 5. Percentile Analysis
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = [
            np.percentile(ratings_df["rating"], p) for p in percentiles
        ]
        axes[1, 1].bar(
            range(len(percentiles)), percentile_values, color="lightcoral", alpha=0.7
        )
        axes[1, 1].set_xticks(range(len(percentiles)))
        axes[1, 1].set_xticklabels([f"{p}th" for p in percentiles])
        axes[1, 1].set_title("Rating Percentiles")
        axes[1, 1].set_xlabel("Percentile")
        axes[1, 1].set_ylabel("Rating Value")
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(percentile_values):
            axes[1, 1].text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom")

        # 6. Rating Count Distribution (Log Scale)
        axes[1, 2].hist(
            movie_stats["count"], bins=50, alpha=0.7, color="gold", edgecolor="black"
        )
        axes[1, 2].set_yscale("log")
        axes[1, 2].set_title("Movie Rating Count Distribution (Log Scale)")
        axes[1, 2].set_xlabel("Number of Ratings per Movie")
        axes[1, 2].set_ylabel("Number of Movies (Log Scale)")
        axes[1, 2].grid(True, alpha=0.3)

        # Add statistical annotations
        mean_count = movie_stats["count"].mean()
        median_count = movie_stats["count"].median()
        axes[1, 2].axvline(
            mean_count, color="red", linestyle="--", label=f"Mean: {mean_count:.0f}"
        )
        axes[1, 2].axvline(
            median_count,
            color="green",
            linestyle="--",
            label=f"Median: {median_count:.0f}",
        )
        axes[1, 2].legend()

        plt.tight_layout()
        path = self.output_dir / "comprehensive_statistics.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_advanced_heatmaps(self, analyzer) -> str:
        """Generate advanced heatmap visualizations"""
        logger.info("    → Generating advanced heatmaps...")

        ratings_df = analyzer.ratings.copy()

        # More aggressive sampling for better performance
        if len(ratings_df) > 50000:
            ratings_df = ratings_df.sample(50000, random_state=42)
            logger.info(f"    → Sampled {len(ratings_df)} ratings for performance")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Hour vs Day of Week Heatmap (Enhanced)
        ratings_df["hour"] = ratings_df["timestamp"].dt.hour
        ratings_df["day_of_week"] = ratings_df["timestamp"].dt.day_name()
        ratings_df["day_num"] = ratings_df["timestamp"].dt.dayofweek

        # Rating count heatmap
        count_pivot = (
            ratings_df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
        )
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        count_pivot = count_pivot.reindex(day_order)

        sns.heatmap(
            count_pivot,
            cmap="YlOrRd",
            annot=False,
            fmt="d",
            cbar_kws={"label": "Number of Ratings"},
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Rating Activity Heatmap: Day vs Hour")
        axes[0, 0].set_xlabel("Hour of Day")
        axes[0, 0].set_ylabel("Day of Week")

        # 2. Average Rating Heatmap
        avg_pivot = (
            ratings_df.groupby(["day_of_week", "hour"])["rating"]
            .mean()
            .unstack(fill_value=0)
        )
        avg_pivot = avg_pivot.reindex(day_order)

        sns.heatmap(
            avg_pivot,
            cmap="RdYlBu_r",
            annot=False,
            fmt=".2f",
            cbar_kws={"label": "Average Rating"},
            ax=axes[0, 1],
            center=3.0,
        )
        axes[0, 1].set_title("Average Rating Heatmap: Day vs Hour")
        axes[0, 1].set_xlabel("Hour of Day")
        axes[0, 1].set_ylabel("Day of Week")

        # 3. User-Movie Rating Matrix (Sample)
        # Get top users and movies for visualization
        top_users = ratings_df["userId"].value_counts().head(20).index
        top_movies = ratings_df["movieId"].value_counts().head(20).index

        sample_ratings = ratings_df[
            (ratings_df["userId"].isin(top_users))
            & (ratings_df["movieId"].isin(top_movies))
        ]

        user_movie_matrix = sample_ratings.pivot_table(
            index="userId", columns="movieId", values="rating", fill_value=0
        )

        sns.heatmap(
            user_movie_matrix,
            cmap="viridis",
            cbar_kws={"label": "Rating"},
            xticklabels=False,
            yticklabels=False,
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("User-Movie Rating Matrix (Top 20x20)")
        axes[1, 0].set_xlabel("Movies")
        axes[1, 0].set_ylabel("Users")

        # 4. Genre Correlation Heatmap (Optimized)
        # Extract genres more efficiently and limit to top genres
        movies_df = analyzer.movies.copy()

        # Get all genres and count their frequency
        all_genres = []
        for genres_str in movies_df["genres"].dropna():
            all_genres.extend(genres_str.split("|"))

        # Only use top 15 most common genres for performance
        top_genres = pd.Series(all_genres).value_counts().head(15).index.tolist()
        logger.info(f"    → Using top {len(top_genres)} genres for correlation analysis")

        # Create optimized genre matrix using only top genres
        genre_data = []
        for _, row in movies_df.iterrows():
            if pd.notna(row["genres"]):
                movie_genres = row["genres"].split("|")
                genre_row = {genre: 1 if genre in movie_genres else 0 for genre in top_genres}
                genre_row['movieId'] = row["movieId"]
                genre_data.append(genre_row)

        if genre_data:
            genre_df = pd.DataFrame(genre_data).set_index('movieId')
            genre_corr = genre_df.corr()

            # Create correlation heatmap
            mask = np.triu(np.ones_like(genre_corr, dtype=bool))
            sns.heatmap(
                genre_corr,
                mask=mask,
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=axes[1, 1],
                annot=False  # Disable annotations for better performance
            )
            axes[1, 1].set_title(f"Genre Correlation Matrix (Top {len(top_genres)})")
            plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha="right")
            plt.setp(axes[1, 1].get_yticklabels(), rotation=0)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No genre data\navailable",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Genre Correlation Matrix")

        plt.tight_layout()
        path = self.output_dir / "advanced_heatmaps.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def plot_percentage_analysis(self, analyzer) -> str:
        """Generate percentage-based analysis charts"""
        logger.info("    → Generating percentage analysis...")

        ratings_df = analyzer.ratings.copy()
        movies_df = analyzer.movies.copy()

        # Sample data for better performance
        if len(ratings_df) > 100000:
            ratings_df = ratings_df.sample(100000, random_state=42)
            logger.info(f"    → Sampled {len(ratings_df)} ratings for percentage analysis")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Rating Distribution Percentages
        rating_counts = ratings_df["rating"].value_counts().sort_index()
        rating_percentages = (rating_counts / rating_counts.sum() * 100).round(1)

        colors = plt.cm.Set3(np.linspace(0, 1, len(rating_percentages)))
        wedges, texts, autotexts = axes[0, 0].pie(
            rating_percentages.values,
            labels=[f"{r} ★" for r in rating_percentages.index],
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
        )
        axes[0, 0].set_title("Rating Distribution Percentages")

        # 2. Genre Popularity Percentages (Optimized)
        # Use vectorized operations for better performance
        genre_series = movies_df["genres"].dropna().str.split("|").explode()
        genre_counts = genre_series.value_counts().head(10)
        genre_percentages = (genre_counts / genre_counts.sum() * 100).round(1)
        logger.info(f"    → Processed {len(genre_series)} genre entries")

        bars = axes[0, 1].bar(
            range(len(genre_percentages)),
            genre_percentages.values,
            color="lightblue",
            alpha=0.8,
        )
        axes[0, 1].set_xticks(range(len(genre_percentages)))
        axes[0, 1].set_xticklabels(genre_percentages.index, rotation=45, ha="right")
        axes[0, 1].set_title("Top 10 Genre Popularity (%)")
        axes[0, 1].set_ylabel("Percentage of Movies")

        # Add percentage labels on bars
        for bar, pct in zip(bars, genre_percentages.values):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
            )

        # 3. User Activity Distribution
        user_rating_counts = ratings_df["userId"].value_counts()

        # Define activity levels
        light_users = (user_rating_counts < 20).sum()
        moderate_users = (
            (user_rating_counts >= 20) & (user_rating_counts <= 100)
        ).sum()
        heavy_users = (user_rating_counts > 100).sum()

        total_users = len(user_rating_counts)
        activity_data = {
            "Light (<20 ratings)": light_users / total_users * 100,
            "Moderate (20-100)": moderate_users / total_users * 100,
            "Heavy (>100)": heavy_users / total_users * 100,
        }

        colors = ["lightcoral", "gold", "lightgreen"]
        wedges, texts, autotexts = axes[1, 0].pie(
            activity_data.values(),
            labels=activity_data.keys(),
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            explode=(0.05, 0.05, 0.05),
        )
        axes[1, 0].set_title("User Activity Distribution")

        # 4. Rating Trends Over Time (Percentage Change)
        ratings_df["year"] = ratings_df["timestamp"].dt.year
        yearly_counts = ratings_df.groupby("year").size()

        if len(yearly_counts) > 1:
            yearly_change = yearly_counts.pct_change() * 100
            yearly_change = yearly_change.dropna()

            colors = ["red" if x < 0 else "green" for x in yearly_change.values]
            bars = axes[1, 1].bar(
                yearly_change.index, yearly_change.values, color=colors, alpha=0.7
            )
            axes[1, 1].set_title("Year-over-Year Rating Activity Change (%)")
            axes[1, 1].set_xlabel("Year")
            axes[1, 1].set_ylabel("Percentage Change")
            axes[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
            axes[1, 1].grid(True, alpha=0.3)

            # Add percentage labels
            for bar, pct in zip(bars, yearly_change.values):
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (1 if bar.get_height() > 0 else -3),
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom" if bar.get_height() > 0 else "top",
                )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Insufficient data\nfor trend analysis",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Year-over-Year Rating Activity Change (%)")

        plt.tight_layout()
        path = self.output_dir / "percentage_analysis.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return str(path)

    def create_correlation_heatmap(self, analyzer, method: str = "pearson") -> str:
        """
        Create a correlation heatmap for movie ratings and features.
        
        Args:
            analyzer: MovieAnalyzer instance with loaded data
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            str: Path to the saved heatmap image
        """
        logger.info("    → Generating correlation heatmap...")
        
        try:
            # Prepare data for correlation analysis
            ratings_df = analyzer.ratings.copy()
            movies_df = analyzer.movies.copy()
            
            # Create movie features matrix
            movie_stats = ratings_df.groupby('movieId').agg({
                'rating': ['mean', 'std', 'count'],
                'userId': 'nunique'
            }).round(3)
            
            movie_stats.columns = ['avg_rating', 'rating_std', 'rating_count', 'unique_users']
            movie_stats = movie_stats.reset_index()
            
            # Add genre features
            movies_with_genres = movies_df.copy()
            
            # Extract top genres for correlation
            all_genres = []
            for genres in movies_with_genres['genres']:
                all_genres.extend(genres.split('|'))
            
            top_genres = pd.Series(all_genres).value_counts().head(10).index.tolist()
            
            # Create binary genre features
            for genre in top_genres:
                movies_with_genres[f'is_{genre}'] = movies_with_genres['genres'].str.contains(genre).astype(int)
            
            # Merge movie stats with genre features
            correlation_data = movie_stats.merge(
                movies_with_genres[['movieId'] + [f'is_{genre}' for genre in top_genres]], 
                on='movieId'
            )
            
            # Select numeric columns for correlation
            numeric_cols = ['avg_rating', 'rating_std', 'rating_count', 'unique_users'] + [f'is_{genre}' for genre in top_genres]
            corr_matrix = correlation_data[numeric_cols].corr(method=method)
            
            # Create the heatmap
            plt.figure(figsize=(14, 12))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Generate heatmap
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8},
                linewidths=0.5
            )
            
            plt.title(f'Movie Features Correlation Heatmap ({method.title()})', fontsize=16, pad=20)
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            path = self.output_dir / f"correlation_heatmap_{method}.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"    ✓ Correlation heatmap saved to {path}")
            return str(path)
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            # Create a simple fallback heatmap
            plt.figure(figsize=(10, 8))
            
            # Simple correlation of basic rating statistics
            simple_stats = analyzer.ratings.groupby('movieId').agg({
                'rating': ['mean', 'std', 'count']
            }).round(3)
            simple_stats.columns = ['avg_rating', 'rating_std', 'rating_count']
            
            corr_matrix = simple_stats.corr(method=method)
            
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f'
            )
            
            plt.title(f'Basic Rating Statistics Correlation ({method.title()})', fontsize=14)
            plt.tight_layout()
            
            path = self.output_dir / f"simple_correlation_heatmap_{method}.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            
            return str(path)

    def plot_movie_similarity_network(self, analyzer, movie_id: int = None, 
                                    similarity_threshold: float = 0.3, max_nodes: int = 20) -> str:
        """
        Create a network visualization of movie similarities.
        
        Args:
            analyzer: MovieAnalyzer instance with loaded data
            movie_id: Central movie ID for the network (if None, uses top-rated movie)
            similarity_threshold: Minimum similarity score to show connections
            max_nodes: Maximum number of nodes to display
            
        Returns:
            str: Path to the saved network visualization
        """
        logger.info("    → Generating movie similarity network...")
        
        try:
            # If no movie_id specified, use the top-rated movie
            if movie_id is None:
                top_movies = analyzer.get_top_movies(limit=1)
                if top_movies:
                    movie_id = top_movies[0]['movieId']
                else:
                    # Fallback to first movie in dataset
                    movie_id = analyzer.movies.iloc[0]['movieId']
            
            # Get similar movies
            similar_movies = analyzer.calculate_movie_similarity(
                movie_id, method="hybrid", limit=max_nodes-1
            )
            
            if not similar_movies:
                # Fallback to genre similarity if hybrid fails
                similar_movies = analyzer.calculate_movie_similarity(
                    movie_id, method="genre", limit=max_nodes-1
                )
            
            # Get central movie info
            central_movie = analyzer.movies[analyzer.movies['movieId'] == movie_id].iloc[0]
            
            # Create network data
            nodes = [{'id': movie_id, 'title': central_movie['title'], 'similarity': 1.0, 'is_central': True}]
            edges = []
            
            for movie in similar_movies:
                if movie['similarity_score'] >= similarity_threshold:
                    nodes.append({
                        'id': movie['movieId'],
                        'title': movie['title'][:30] + ('...' if len(movie['title']) > 30 else ''),
                        'similarity': movie['similarity_score'],
                        'is_central': False
                    })
                    edges.append({
                        'source': movie_id,
                        'target': movie['movieId'],
                        'weight': movie['similarity_score']
                    })
            
            # Create the network visualization
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Calculate positions using a circular layout
            n_nodes = len(nodes)
            if n_nodes > 1:
                angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
                
                # Central node at center
                positions = {movie_id: (0, 0)}
                
                # Other nodes in circle
                radius = 3
                for i, node in enumerate(nodes[1:], 1):
                    angle = angles[i-1]
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    positions[node['id']] = (x, y)
                
                # Draw edges
                for edge in edges:
                    source_pos = positions[edge['source']]
                    target_pos = positions[edge['target']]
                    
                    # Line width based on similarity
                    line_width = edge['weight'] * 5
                    alpha = min(edge['weight'] + 0.3, 1.0)
                    
                    ax.plot([source_pos[0], target_pos[0]], 
                           [source_pos[1], target_pos[1]], 
                           'gray', linewidth=line_width, alpha=alpha)
                    
                    # Add similarity score label
                    mid_x = (source_pos[0] + target_pos[0]) / 2
                    mid_y = (source_pos[1] + target_pos[1]) / 2
                    ax.text(mid_x, mid_y, f'{edge["weight"]:.2f}', 
                           fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
                
                # Draw nodes
                for node in nodes:
                    pos = positions[node['id']]
                    
                    if node['is_central']:
                        # Central node - larger and different color
                        circle = plt.Circle(pos, 0.3, color='red', alpha=0.8, zorder=3)
                        ax.add_patch(circle)
                        ax.text(pos[0], pos[1]-0.6, node['title'], 
                               fontsize=12, ha='center', va='top', weight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
                    else:
                        # Similar nodes - size based on similarity
                        size = 0.1 + (node['similarity'] * 0.2)
                        circle = plt.Circle(pos, size, color='steelblue', alpha=0.7, zorder=2)
                        ax.add_patch(circle)
                        ax.text(pos[0], pos[1]-size-0.2, node['title'], 
                               fontsize=9, ha='center', va='top',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
                
                # Set equal aspect ratio and limits
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                ax.set_aspect('equal')
                ax.axis('off')
                
                # Add title and legend
                ax.set_title(f'Movie Similarity Network\nCentral Movie: {central_movie["title"]}', 
                           fontsize=16, pad=20)
                
                # Add legend
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                              markersize=15, label='Central Movie'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', 
                              markersize=10, label='Similar Movies'),
                    plt.Line2D([0], [0], color='gray', linewidth=3, label='Similarity Connection')
                ]
                ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
                
            else:
                ax.text(0.5, 0.5, 'No similar movies found\nwith sufficient similarity', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'Movie Similarity Network\nCentral Movie: {central_movie["title"]}', 
                           fontsize=16)
                ax.axis('off')
            
            plt.tight_layout()
            path = self.output_dir / f"movie_similarity_network_{movie_id}.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"    ✓ Movie similarity network saved to {path}")
            return str(path)
            
        except Exception as e:
            logger.error(f"Error creating movie similarity network: {e}")
            
            # Create a simple fallback visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f'Error creating similarity network\nfor movie ID: {movie_id}\n\nError: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Movie Similarity Network - Error', fontsize=14)
            ax.axis('off')
            
            plt.tight_layout()
            path = self.output_dir / f"movie_similarity_network_error_{movie_id}.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            
            return str(path)
