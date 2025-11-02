#!/usr/bin/env python3
"""
Modern Flask API for MovieLens Analysis
Serves analysis data with proper CORS and JSON endpoints
"""

import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.analyzer import MovieAnalyzer
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.visualizer import InsightsVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
CACHE_DURATION = 3600  # 1 hour in seconds
CACHE_FILE = Path("data/cache/analysis_cache.json")
CACHE_TIMESTAMP_FILE = Path("data/cache/cache_timestamp.txt")
VISUALIZATION_CACHE_DIR = Path("data/cache/visualizations")

# Performance optimization: Global data cache to avoid reloading
_global_data_cache = {
    'movies_df': None,
    'ratings_df': None,
    'analyzer': None,
    'last_loaded': None
}

# Memory optimization: Chunk size for large operations
CHUNK_SIZE = 10000


def cache_visualization(cache_key):
    """Decorator to cache expensive visualization operations"""
    def decorator(func):
        from functools import wraps

        @wraps(func)  # This preserves the original function's metadata
        def wrapper(*args, **kwargs):
            # Ensure cache directory exists
            VISUALIZATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            cache_file = VISUALIZATION_CACHE_DIR / f"{cache_key}.json"
            timestamp_file = VISUALIZATION_CACHE_DIR / f"{cache_key}_timestamp.txt"

            current_time = datetime.now().timestamp()

            # Check if cached result exists and is still valid
            if cache_file.exists() and timestamp_file.exists():
                try:
                    with open(timestamp_file, 'r') as f:
                        cached_timestamp = float(f.read().strip())

                    if current_time - cached_timestamp < CACHE_DURATION:
                        with open(cache_file, 'r') as f:
                            cached_result = json.load(f)
                        logger.info(f"Using cached result for {cache_key}")
                        return jsonify(cached_result)
                except (ValueError, FileNotFoundError, json.JSONDecodeError):
                    pass  # Cache invalid, proceed with fresh computation

            # Generate fresh result
            logger.info(f"Generating fresh result for {cache_key}")
            result = func(*args, **kwargs)

            # Cache the result if it's successful
            if hasattr(result, 'get_json') and result.status_code == 200:
                try:
                    result_data = result.get_json()
                    with open(cache_file, 'w') as f:
                        json.dump(result_data, f)
                    with open(timestamp_file, 'w') as f:
                        f.write(str(current_time))
                except Exception as e:
                    logger.warning(f"Failed to cache result for {cache_key}: {e}")

            return result
        return wrapper
    return decorator


def clear_global_cache():
    """Clear the global data cache to force fresh data loading"""
    global _global_data_cache
    _global_data_cache = {
        'movies_df': None,
        'ratings_df': None,
        'analyzer': None,
        'last_loaded': None
    }
    logger.info("Global data cache cleared")


def get_analysis_data():
    """Get or refresh cached analysis data with performance optimizations"""
    current_time = datetime.now().timestamp()
    global _global_data_cache

    # Ensure cache directory exists
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load cached data if it exists
    cached_analysis = None
    cached_timestamp = None

    if CACHE_FILE.exists() and CACHE_TIMESTAMP_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                cached_analysis = json.load(f)
            with open(CACHE_TIMESTAMP_FILE, 'r') as f:
                cached_timestamp = float(f.read().strip())
        except (json.JSONDecodeError, ValueError, FileNotFoundError) as e:
            logger.warning(f"Failed to load cache: {e}")
            cached_analysis = None
            cached_timestamp = None

    # Check if we need to refresh cache
    if (
        cached_analysis is None
        or cached_timestamp is None
        or current_time - cached_timestamp > CACHE_DURATION
    ):

        logger.info("Refreshing analysis cache...")

        try:
            # Performance optimization: Check if data is already loaded in memory
            if (_global_data_cache['last_loaded'] is None or
                    current_time - _global_data_cache['last_loaded'] > CACHE_DURATION):

                logger.info("Loading fresh data from disk...")
                # Load and process data
                loader = DataLoader(
                    data_dir=Path("data"), source="grouplens", dataset="ml-latest-small"
                )
                movies_df, ratings_df = loader.load_or_download()

                processor = DataProcessor()
                movies_df = processor.clean_movies(movies_df)
                ratings_df = processor.clean_ratings(ratings_df)

                # Cache in memory for subsequent requests
                _global_data_cache['movies_df'] = movies_df
                _global_data_cache['ratings_df'] = ratings_df
                _global_data_cache['last_loaded'] = current_time

                logger.info(f"Data loaded: {len(movies_df)} movies, {len(ratings_df)} ratings")
            else:
                logger.info("Using cached data from memory...")
                movies_df = _global_data_cache['movies_df']
                ratings_df = _global_data_cache['ratings_df']

            # Create or reuse analyzer
            if _global_data_cache['analyzer'] is None:
                analyzer = MovieAnalyzer(movies_df, ratings_df)
                _global_data_cache['analyzer'] = analyzer
            else:
                analyzer = _global_data_cache['analyzer']

            # Get all analysis results
            logger.info("Getting top movies...")
            top_movies = analyzer.get_top_movies(limit=20, min_ratings=50)
            logger.info("Getting genre stats...")
            genre_stats = analyzer.analyze_genre_trends()
            logger.info("Getting time series...")
            time_series = analyzer.generate_time_series_analysis()
            logger.info("Getting rating distribution...")
            rating_dist = analyzer.get_rating_distribution()
            logger.info("Getting user stats...")
            user_stats = analyzer.get_user_behavior_stats()
            logger.info("All basic analysis completed, starting comprehensive stats...")

            # Generate comprehensive statistics
            try:
                visualizer = InsightsVisualizer(output_dir=Path("outputs/plots"))
                plot_path = visualizer.plot_comprehensive_statistics(analyzer)
            except Exception as e:
                logger.error(f"Failed to generate comprehensive statistics plot: {e}")
                plot_path = "outputs/plots/comprehensive_statistics.png"  # fallback

            # Calculate detailed statistics for comprehensive analysis (optimized)
            ratings_stats = {
                "mean": float(ratings_df["rating"].mean()),
                "median": float(ratings_df["rating"].median()),
                "std": float(ratings_df["rating"].std()),
                "min": float(ratings_df["rating"].min()),
                "max": float(ratings_df["rating"].max()),
                "q25": float(ratings_df["rating"].quantile(0.25)),
                "q75": float(ratings_df["rating"].quantile(0.75)),
                "skewness": float(ratings_df["rating"].skew()),
                "kurtosis": float(ratings_df["rating"].kurtosis()),
            }

            # Movie-level statistics (memory optimized with chunking for large datasets)
            logger.info("Computing movie-level statistics...")
            if len(ratings_df) > CHUNK_SIZE * 10:  # Only chunk for very large datasets
                movie_stats_list = []
                for chunk in pd.read_csv(ratings_df, chunksize=CHUNK_SIZE):
                    chunk_stats = (
                        chunk.groupby("movieId")
                        .agg(
                            count=("rating", "count"),
                            mean=("rating", "mean"),
                            std=("rating", "std"),
                            median=("rating", "median"),
                        )
                        .reset_index()
                    )
                    movie_stats_list.append(chunk_stats)
                movie_stats = pd.concat(movie_stats_list, ignore_index=True)
            else:
                movie_stats = (
                    ratings_df.groupby("movieId")
                    .agg(
                        count=("rating", "count"),
                        mean=("rating", "mean"),
                        std=("rating", "std"),
                        median=("rating", "median"),
                    )
                    .reset_index()
                )

            movie_stats_summary = {
                "total_movies": len(movie_stats),
                "avg_ratings_per_movie": float(movie_stats["count"].mean()),
                "median_ratings_per_movie": float(movie_stats["count"].median()),
                "movies_with_high_ratings": len(movie_stats[movie_stats["count"] >= 100]),
                "correlation_count_mean": float(
                    movie_stats[["count", "mean"]].corr().iloc[0, 1]
                ),
            }

            comprehensive_stats = {
                "status": "success",
                "plot_path": plot_path,
                "ratings_statistics": ratings_stats,
                "movie_statistics": movie_stats_summary,
                "timestamp": datetime.now().isoformat(),
            }

            # Cache the results
            cached_analysis = {
                "metadata": {
                    "dataset": "ml-1m",
                    "source": "grouplens",
                    "n_movies": len(movies_df),
                    "n_ratings": len(ratings_df),
                    "n_users": ratings_df["userId"].nunique(),
                    "date_range": [
                        ratings_df["timestamp"].min().strftime("%Y-%m-%d"),
                        ratings_df["timestamp"].max().strftime("%Y-%m-%d"),
                    ],
                    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                "top_movies": top_movies,
                "genre_stats": genre_stats,
                "time_series": time_series,
                "rating_distribution": rating_dist,
                "user_stats": user_stats,
                "comprehensive_statistics": comprehensive_stats,
            }

            # Save cache to files
            try:
                with open(CACHE_FILE, 'w') as f:
                    json.dump(cached_analysis, f, indent=2)
                with open(CACHE_TIMESTAMP_FILE, 'w') as f:
                    f.write(str(current_time))
                logger.info("Analysis cache refreshed and saved successfully")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")

            cached_timestamp = current_time

        except Exception as e:
            logger.error(f"Failed to refresh analysis cache: {e}")
            if cached_analysis is None:
                # Return empty data if no cache exists
                cached_analysis = {
                    "error": "Failed to load analysis data",
                    "metadata": {"dataset": "ml-1m", "error": str(e)},
                }

            # Try to save error state to cache files
            try:
                with open(CACHE_FILE, 'w') as f:
                    json.dump(cached_analysis, f, indent=2)
                with open(CACHE_TIMESTAMP_FILE, 'w') as f:
                    f.write(str(current_time))
                logger.info("Error state saved to cache")
            except Exception as cache_error:
                logger.error(f"Failed to save error state to cache: {cache_error}")

    return cached_analysis


@app.route("/")
def index():
    """Serve the main frontend page"""
    return send_from_directory("frontend", "index.html")


@app.route("/api/overview")
def api_overview():
    """Get overview/metadata information"""
    data = get_analysis_data()
    return jsonify({"status": "success", "data": data.get("metadata", {})})


@app.route("/api/rating-distribution")
def api_rating_distribution():
    """Get rating distribution data"""
    data = get_analysis_data()
    rating_dist = data.get("rating_distribution", {})

    # Format for Chart.js
    distribution = rating_dist.get("distribution", {})
    chart_data = {
        "labels": [str(rating) for rating in sorted(distribution.keys())],
        "datasets": [
            {
                "label": "Number of Ratings",
                "data": [
                    distribution[rating] for rating in sorted(distribution.keys())
                ],
                "backgroundColor": "rgba(54, 162, 235, 0.8)",
                "borderColor": "rgba(54, 162, 235, 1)",
                "borderWidth": 1,
            }
        ],
    }

    return jsonify(
        {
            "status": "success",
            "data": {
                "chart": chart_data,
                "statistics": rating_dist.get("statistics", {}),
            },
        }
    )


@app.route("/api/top-movies")
def api_top_movies():
    """Get top movies data"""
    data = get_analysis_data()
    top_movies = data.get("top_movies", [])

    # Format for horizontal bar chart
    chart_data = {
        "labels": [
            movie["title"][:40] + ("..." if len(movie["title"]) > 40 else "")
            for movie in top_movies[:15]
        ],
        "datasets": [
            {
                "label": "Weighted Rating",
                "data": [movie["weighted_rating"] for movie in top_movies[:15]],
                "backgroundColor": "rgba(255, 99, 132, 0.8)",
                "borderColor": "rgba(255, 99, 132, 1)",
                "borderWidth": 1,
            }
        ],
    }

    return jsonify(
        {"status": "success", "data": {"chart": chart_data, "movies": top_movies}}
    )


@app.route("/api/genre-popularity")
def api_genre_popularity():
    """Get genre popularity data"""
    data = get_analysis_data()
    genre_stats = data.get("genre_stats", {})
    overall_genres = genre_stats.get("overall", [])[:15]

    # Format for horizontal bar chart
    chart_data = {
        "labels": [genre["genre"] for genre in overall_genres],
        "datasets": [
            {
                "label": "Number of Ratings",
                "data": [genre["count"] for genre in overall_genres],
                "backgroundColor": "rgba(153, 102, 255, 0.8)",
                "borderColor": "rgba(153, 102, 255, 1)",
                "borderWidth": 1,
            }
        ],
    }

    # Scatter plot data for rating vs popularity
    scatter_data = {
        "datasets": [
            {
                "label": "Genres",
                "data": [
                    {
                        "x": genre["count"],
                        "y": genre["mean_rating"],
                        "label": genre["genre"],
                    }
                    for genre in overall_genres
                ],
                "backgroundColor": "rgba(75, 192, 192, 0.8)",
                "borderColor": "rgba(75, 192, 192, 1)",
                "pointRadius": 8,
            }
        ]
    }

    return jsonify(
        {
            "status": "success",
            "data": {
                "bar_chart": chart_data,
                "scatter_chart": scatter_data,
                "genres": overall_genres,
            },
        }
    )


@app.route("/api/time-series")
def api_time_series():
    """Get time series data"""
    data = get_analysis_data()
    time_series = data.get("time_series", {})
    monthly_data = time_series.get("monthly", [])

    # Format for line chart
    chart_data = {
        "labels": [entry["year_month"] for entry in monthly_data],
        "datasets": [
            {
                "label": "Number of Ratings",
                "data": [entry["count"] for entry in monthly_data],
                "borderColor": "rgba(255, 206, 86, 1)",
                "backgroundColor": "rgba(255, 206, 86, 0.2)",
                "tension": 0.4,
                "fill": True,
            },
            {
                "label": "Average Rating",
                "data": [entry["mean_rating"] for entry in monthly_data],
                "borderColor": "rgba(75, 192, 192, 1)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.4,
                "yAxisID": "y1",
            },
        ],
    }

    return jsonify(
        {
            "status": "success",
            "data": {"chart": chart_data, "summary": time_series.get("summary", {})},
        }
    )


@app.route("/api/user-stats")
def api_user_stats():
    """Get user statistics"""
    data = get_analysis_data()
    user_stats = data.get("user_stats", {})

    activity = user_stats.get("user_activity_distribution", {})
    pie_data = {
        "labels": [
            "Light Users (<20)",
            "Moderate Users (20-100)",
            "Heavy Users (>100)",
        ],
        "datasets": [
            {
                "data": [
                    activity.get("light", 0),
                    activity.get("moderate", 0),
                    activity.get("heavy", 0),
                ],
                "backgroundColor": [
                    "rgba(255, 99, 132, 0.8)",
                    "rgba(54, 162, 235, 0.8)",
                    "rgba(255, 205, 86, 0.8)",
                ],
                "borderColor": [
                    "rgba(255, 99, 132, 1)",
                    "rgba(54, 162, 235, 1)",
                    "rgba(255, 205, 86, 1)",
                ],
                "borderWidth": 1,
            }
        ],
    }

    return jsonify(
        {"status": "success", "data": {"chart": pie_data, "statistics": user_stats}}
    )


@app.route("/api/comprehensive-statistics")
def api_comprehensive_statistics():
    """Get comprehensive statistical analysis data and visualizations"""
    try:
        data = get_analysis_data()

        if "error" in data:
            return jsonify({"error": "No data available"}), 404

        # Use cached comprehensive statistics if available
        if "comprehensive_statistics" in data:
            return jsonify(data["comprehensive_statistics"])

        # Fallback: return basic statistics from cached data
        return jsonify(
            {
                "status": "success",
                "plot_path": "outputs/plots/comprehensive_statistics.png",
                "ratings_statistics": data.get("rating_distribution", {}),
                "movie_statistics": data.get("top_movies", {}),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error generating comprehensive statistics: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/advanced-heatmaps")
@cache_visualization("advanced_heatmaps")
def api_advanced_heatmaps():
    """Get advanced heatmap visualizations"""
    try:
        data = get_analysis_data()

        if "error" in data:
            return jsonify({"error": "No data available"}), 404

        # Generate advanced heatmaps
        loader = DataLoader(
            data_dir=Path("data"), source="grouplens", dataset="ml-latest-small"
        )
        movies_df, ratings_df = loader.load_or_download()

        processor = DataProcessor()
        movies_df = processor.clean_movies(movies_df)
        ratings_df = processor.clean_ratings(ratings_df)

        analyzer = MovieAnalyzer(movies_df, ratings_df)
        visualizer = InsightsVisualizer(output_dir=Path("outputs/plots"))

        # Generate advanced heatmaps
        plot_path = visualizer.plot_advanced_heatmaps(analyzer)

        # Calculate heatmap-related statistics
        ratings_df["hour"] = ratings_df["timestamp"].dt.hour
        ratings_df["day_of_week"] = ratings_df["timestamp"].dt.day_name()

        # Peak activity analysis
        hourly_activity = ratings_df.groupby("hour").size()
        daily_activity = ratings_df.groupby("day_of_week").size()

        peak_hour = int(hourly_activity.idxmax())
        peak_day = daily_activity.idxmax()

        # Genre correlation insights
        all_genres = set()
        for genres in movies_df["genres"].str.split("|"):
            if isinstance(genres, list):
                all_genres.update(genres)

        heatmap_insights = {
            "peak_activity_hour": peak_hour,
            "peak_activity_day": peak_day,
            "total_genres": len(all_genres),
            "hourly_variance": float(hourly_activity.var()),
            "daily_variance": float(daily_activity.var()),
        }

        return jsonify(
            {
                "status": "success",
                "plot_path": plot_path,
                "insights": heatmap_insights,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error generating advanced heatmaps: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/percentage-analysis")
@cache_visualization("percentage_analysis")
def api_percentage_analysis():
    """Get percentage-based analysis data and visualizations"""
    try:
        data = get_analysis_data()

        if "error" in data:
            return jsonify({"error": "No data available"}), 404

        # Generate percentage analysis
        loader = DataLoader(
            data_dir=Path("data"), source="grouplens", dataset="ml-latest-small"
        )
        movies_df, ratings_df = loader.load_or_download()

        processor = DataProcessor()
        movies_df = processor.clean_movies(movies_df)
        ratings_df = processor.clean_ratings(ratings_df)

        analyzer = MovieAnalyzer(movies_df, ratings_df)
        visualizer = InsightsVisualizer(output_dir=Path("outputs/plots"))

        # Generate percentage analysis plot
        plot_path = visualizer.plot_percentage_analysis(analyzer)

        # Calculate percentage statistics
        rating_distribution = ratings_df["rating"].value_counts(normalize=True) * 100

        # Genre percentages
        all_genres = []
        for genres_str in movies_df["genres"].dropna():
            all_genres.extend(genres_str.split("|"))

        genre_distribution = (
            pd.Series(all_genres).value_counts(normalize=True).head(10) * 100
        )

        # User activity percentages
        user_rating_counts = ratings_df["userId"].value_counts()
        light_users_pct = (
            (user_rating_counts < 20).sum() / len(user_rating_counts) * 100
        )
        moderate_users_pct = (
            ((user_rating_counts >= 20) & (user_rating_counts <= 100)).sum()
            / len(user_rating_counts)
            * 100
        )
        heavy_users_pct = (
            (user_rating_counts > 100).sum() / len(user_rating_counts) * 100
        )

        # Year-over-year growth
        ratings_df["year"] = ratings_df["timestamp"].dt.year
        yearly_counts = ratings_df.groupby("year").size()
        yearly_growth = (
            yearly_counts.pct_change() * 100 if len(yearly_counts) > 1 else pd.Series()
        )

        percentage_data = {
            "rating_distribution": rating_distribution.to_dict(),
            "top_genres": genre_distribution.to_dict(),
            "user_activity": {
                "light_users": float(light_users_pct),
                "moderate_users": float(moderate_users_pct),
                "heavy_users": float(heavy_users_pct),
            },
            "yearly_growth": (
                yearly_growth.dropna().to_dict() if not yearly_growth.empty else {}
            ),
        }

        return jsonify(
            {
                "status": "success",
                "plot_path": plot_path,
                "percentage_data": percentage_data,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error generating percentage analysis: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/statistical-summary")
def api_statistical_summary():
    """Get comprehensive statistical summary for all analyses"""
    try:
        data = get_analysis_data()

        if "error" in data:
            return jsonify({"error": "No data available"}), 404

        # Load data for analysis
        loader = DataLoader(
            data_dir=Path("data"), source="grouplens", dataset="ml-latest-small"
        )
        movies_df, ratings_df = loader.load_or_download()

        processor = DataProcessor()
        movies_df = processor.clean_movies(movies_df)
        ratings_df = processor.clean_ratings(ratings_df)

        # Comprehensive statistical summary
        summary = {
            "dataset_overview": {
                "total_movies": len(movies_df),
                "total_ratings": len(ratings_df),
                "total_users": ratings_df["userId"].nunique(),
                "date_range": [
                    ratings_df["timestamp"].min().strftime("%Y-%m-%d"),
                    ratings_df["timestamp"].max().strftime("%Y-%m-%d"),
                ],
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "rating_statistics": {
                "mean": float(ratings_df["rating"].mean()),
                "median": float(ratings_df["rating"].median()),
                "mode": float(ratings_df["rating"].mode().iloc[0]),
                "std_dev": float(ratings_df["rating"].std()),
                "variance": float(ratings_df["rating"].var()),
                "skewness": float(ratings_df["rating"].skew()),
                "kurtosis": float(ratings_df["rating"].kurtosis()),
                "range": float(ratings_df["rating"].max() - ratings_df["rating"].min()),
                "iqr": float(
                    ratings_df["rating"].quantile(0.75)
                    - ratings_df["rating"].quantile(0.25)
                ),
            },
            "distribution_analysis": {
                "percentiles": {
                    "10th": float(ratings_df["rating"].quantile(0.1)),
                    "25th": float(ratings_df["rating"].quantile(0.25)),
                    "50th": float(ratings_df["rating"].quantile(0.5)),
                    "75th": float(ratings_df["rating"].quantile(0.75)),
                    "90th": float(ratings_df["rating"].quantile(0.9)),
                },
                "rating_counts": ratings_df["rating"]
                .value_counts()
                .sort_index()
                .to_dict(),
            },
            "correlation_analysis": {
                "available": True,
                "description": "Correlation matrices available for movie metrics and genre relationships",
            },
        }

        return jsonify({"status": "success", "summary": summary})

    except Exception as e:
        logger.error(f"Error generating statistical summary: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh")
def api_refresh():
    """Force refresh of analysis cache with performance optimizations"""
    # Clear both file cache and memory cache
    try:
        # Clear file cache
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        if CACHE_TIMESTAMP_FILE.exists():
            CACHE_TIMESTAMP_FILE.unlink()

        # Clear visualization cache
        if VISUALIZATION_CACHE_DIR.exists():
            for cache_file in VISUALIZATION_CACHE_DIR.glob("*.json"):
                cache_file.unlink()
            for timestamp_file in VISUALIZATION_CACHE_DIR.glob("*_timestamp.txt"):
                timestamp_file.unlink()

        # Clear global memory cache
        clear_global_cache()

        logger.info("All caches cleared successfully")
    except Exception as e:
        logger.error(f"Failed to clear caches: {e}")

    # Trigger refresh
    get_analysis_data()

    return jsonify({
        "status": "success",
        "message": "Analysis data and all caches refreshed",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/status")
def get_status():
    """Get API status"""
    return jsonify(
        {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
        }
    )


@app.route("/api/health")
def get_health():
    """Health check endpoint for monitoring tools"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "movielens-analysis-api",
        }
    )


@app.route("/api/dataset/status")
def get_dataset_status():
    """Get current dataset status"""
    try:
        data = get_analysis_data()
        if "error" not in data:
            metadata = data.get("metadata", {})
            return jsonify(
                {
                    "loaded": True,
                    "movies_count": metadata.get("n_movies", 0),
                    "ratings_count": metadata.get("n_ratings", 0),
                    "users_count": metadata.get("n_users", 0),
                    "date_range": metadata.get("date_range", []),
                }
            )
        else:
            return jsonify({"loaded": False})
    except Exception as e:
        return jsonify({"error": str(e), "loaded": False}), 500


@app.route("/api/dataset/load-sample", methods=["POST"])
def load_sample_dataset():
    """Load the sample MovieLens dataset"""
    try:
        # Force refresh of analysis cache to reload data
        global cached_analysis, cached_timestamp
        cached_analysis = None
        cached_timestamp = None

        # Trigger data loading
        data = get_analysis_data()

        if "error" not in data:
            metadata = data.get("metadata", {})
            return jsonify(
                {
                    "success": True,
                    "message": "Sample dataset loaded successfully",
                    "movies_count": metadata.get("n_movies", 0),
                    "ratings_count": metadata.get("n_ratings", 0),
                    "users_count": metadata.get("n_users", 0),
                }
            )
        else:
            return (
                jsonify(
                    {"success": False, "error": data.get("error", "Unknown error")}
                ),
                500,
            )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/dataset/upload", methods=["POST"])
def upload_dataset():
    """Upload and process a custom dataset"""
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        # Create a temporary directory for processing
        temp_dir = tempfile.mkdtemp()

        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)

            # Process the uploaded file
            if filename.endswith(".zip"):
                # Extract ZIP file
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Look for CSV files in the extracted content
                csv_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for f in files:
                        if f.endswith(".csv"):
                            csv_files.append(os.path.join(root, f))

                # Try to identify movies and ratings files
                movies_file = None
                ratings_file = None

                for csv_file in csv_files:
                    basename = os.path.basename(csv_file).lower()
                    if "movie" in basename:
                        movies_file = csv_file
                    elif "rating" in basename:
                        ratings_file = csv_file

                if not movies_file or not ratings_file:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "Could not find movies.csv and ratings.csv in the uploaded ZIP file",
                            }
                        ),
                        400,
                    )

            elif filename.endswith(".csv"):
                # Single CSV file - assume it's ratings data
                ratings_file = file_path
                movies_file = None
            else:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Unsupported file format. Please upload a ZIP file or CSV file.",
                        }
                    ),
                    400,
                )

            # Load and validate the data
            movies_df = None
            ratings_df = None

            if movies_file:
                movies_df = pd.read_csv(movies_file)
            if ratings_file:
                ratings_df = pd.read_csv(ratings_file)
                # Convert timestamp if it exists
                if "timestamp" in ratings_df.columns:
                    ratings_df["timestamp"] = pd.to_datetime(
                        ratings_df["timestamp"], unit="s"
                    )

            return jsonify(
                {
                    "success": True,
                    "message": "Dataset uploaded and processed successfully",
                    "movies_count": len(movies_df) if movies_df is not None else 0,
                    "ratings_count": len(ratings_df) if ratings_df is not None else 0,
                    "users_count": (
                        ratings_df["userId"].nunique() if ratings_df is not None else 0
                    ),
                }
            )

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/dataset/load-url", methods=["POST"])
def load_dataset_from_url():
    """Load dataset from a URL"""
    try:
        data = request.get_json()
        url = data.get("url")

        if not url:
            return jsonify({"success": False, "error": "No URL provided"}), 400

        # Create a temporary directory for processing
        temp_dir = tempfile.mkdtemp()

        try:
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Determine filename from URL or content-disposition
            filename = url.split("/")[-1]
            if not filename or "." not in filename:
                filename = "dataset.zip"

            file_path = os.path.join(temp_dir, filename)

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Process the downloaded file (similar to upload logic)
            if filename.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                csv_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for f in files:
                        if f.endswith(".csv"):
                            csv_files.append(os.path.join(root, f))

                movies_file = None
                ratings_file = None

                for csv_file in csv_files:
                    basename = os.path.basename(csv_file).lower()
                    if "movie" in basename:
                        movies_file = csv_file
                    elif "rating" in basename:
                        ratings_file = csv_file

                if not movies_file or not ratings_file:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "Could not find movies.csv and ratings.csv in the downloaded ZIP file",
                            }
                        ),
                        400,
                    )

            elif filename.endswith(".csv"):
                ratings_file = file_path
                movies_file = None
            else:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Unsupported file format. URL must point to a ZIP or CSV file.",
                        }
                    ),
                    400,
                )

            # Load the data
            movies_df = None
            ratings_df = None

            if movies_file:
                movies_df = pd.read_csv(movies_file)
            if ratings_file:
                ratings_df = pd.read_csv(ratings_file)
                if "timestamp" in ratings_df.columns:
                    ratings_df["timestamp"] = pd.to_datetime(
                        ratings_df["timestamp"], unit="s"
                    )

            return jsonify(
                {
                    "success": True,
                    "message": "Dataset loaded from URL successfully",
                    "movies_count": len(movies_df) if movies_df is not None else 0,
                    "ratings_count": len(ratings_df) if ratings_df is not None else 0,
                    "users_count": (
                        ratings_df["userId"].nunique() if ratings_df is not None else 0
                    ),
                }
            )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/dataset/clear", methods=["POST"])
def clear_dataset():
    """Clear the current dataset"""
    try:
        global cached_analysis, cached_timestamp
        cached_analysis = None
        cached_timestamp = None

        return jsonify({"success": True, "message": "Dataset cleared successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/dataset/preview")
def get_dataset_preview():
    """Get a preview of the current dataset"""
    try:
        data = get_analysis_data()

        if "error" in data:
            return jsonify({"error": "No dataset loaded"}), 404

        metadata = data.get("metadata", {})

        # Get basic statistics
        stats = {
            "movies": {
                "count": metadata.get("n_movies", 0),
                "columns": ["movieId", "title", "genres"],
            },
            "ratings": {
                "count": metadata.get("n_ratings", 0),
                "columns": ["userId", "movieId", "rating", "timestamp"],
                "unique_users": metadata.get("n_users", 0),
                "date_range": metadata.get("date_range", []),
            },
        }

        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/frontend/<path:filename>")
def serve_frontend(filename):
    """Serve frontend static files"""
    return send_from_directory("frontend", filename)


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    """Serve output files (plots, reports, etc.)"""
    return send_from_directory("outputs", filename)


if __name__ == "__main__":
    # Ensure frontend directory exists
    frontend_dir = Path("frontend")
    frontend_dir.mkdir(exist_ok=True)

    logger.info("Starting MovieLens Analysis API server...")
    logger.info("API will be available at: http://localhost:8001")

    app.run(host="0.0.0.0", port=8001, debug=True)
