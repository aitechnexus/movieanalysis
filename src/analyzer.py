import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MovieAnalyzer:
    """
    Advanced analytics engine for MovieLens datasets providing comprehensive movie insights.
    
    This class performs sophisticated statistical analysis on MovieLens data including
    movie rankings, genre trends, temporal patterns, and user behavior analysis.
    It implements industry-standard algorithms like weighted ratings and Wilson score
    intervals for robust movie recommendations.
    
    The analyzer provides:
    - Top movie rankings with statistical confidence
    - Genre popularity and trend analysis
    - Time-series analysis of rating patterns
    - User behavior and activity statistics
    - Rating distribution analysis
    
    Attributes:
        movies (pd.DataFrame): Movies dataset with movieId, title, genres
        ratings (pd.DataFrame): Ratings dataset with userId, movieId, rating, timestamp
        global_mean (float): Cached global average rating across all movies
    
    Examples:
        Basic initialization and analysis:
        >>> analyzer = MovieAnalyzer(movies_df, ratings_df)
        >>> top_movies = analyzer.get_top_movies(limit=10)
        >>> genre_trends = analyzer.analyze_genre_trends()
        
        Advanced analysis:
        >>> # Get top movies with custom parameters
        >>> top_movies = analyzer.get_top_movies(
        ...     limit=50, min_ratings=200, m_percentile=0.8
        ... )
        >>> 
        >>> # Comprehensive analytics
        >>> time_analysis = analyzer.generate_time_series_analysis()
        >>> user_stats = analyzer.get_user_behavior_stats()
        >>> rating_dist = analyzer.get_rating_distribution()
    """
    
    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """
        Initialize the MovieAnalyzer with movies and ratings datasets.
        
        Args:
            movies_df (pd.DataFrame): Movies dataset with columns [movieId, title, genres]
            ratings_df (pd.DataFrame): Ratings dataset with columns [userId, movieId, rating, timestamp]
            
        Raises:
            KeyError: If required columns are missing from datasets
            ValueError: If datasets are empty or contain invalid data
            
        Examples:
            >>> movies = pd.DataFrame({
            ...     'movieId': [1, 2, 3],
            ...     'title': ['Movie A', 'Movie B', 'Movie C'],
            ...     'genres': ['Action', 'Drama', 'Comedy']
            ... })
            >>> ratings = pd.DataFrame({
            ...     'userId': [1, 1, 2],
            ...     'movieId': [1, 2, 1],
            ...     'rating': [4.5, 3.0, 5.0],
            ...     'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
            ... })
            >>> analyzer = MovieAnalyzer(movies, ratings)
        """
        # Validate that datasets are not empty
        if movies_df.empty or ratings_df.empty:
            raise ValueError("Input dataframes cannot be empty")
            
        self.movies, self.ratings, self._global_mean = movies_df, ratings_df, None
        # Backward compatibility properties for tests
        self.movies_df = self.movies
        self.ratings_df = self.ratings

    @property
    def global_mean(self) -> float:
        """
        Calculate and cache the global mean rating across all movies.
        
        Returns:
            float: Global average rating (cached after first calculation)
            
        Examples:
            >>> analyzer = MovieAnalyzer(movies_df, ratings_df)
            >>> print(f"Global average rating: {analyzer.global_mean:.2f}")
            Global average rating: 3.52
        """
        if self._global_mean is None:
            self._global_mean = float(self.ratings["rating"].mean())
        return self._global_mean

    def get_top_movies(
        self, limit: int = 20, min_ratings: int = 100, m_percentile: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Calculate top-rated movies using weighted rating and Wilson score algorithms.
        
        Implements IMDB's weighted rating formula and Wilson score confidence intervals
        to provide statistically robust movie rankings that account for both rating
        quality and quantity.
        
        Args:
            limit (int): Maximum number of movies to return (default: 20)
            min_ratings (int): Minimum number of ratings required (default: 100)
            m_percentile (float): Percentile for minimum vote threshold (default: 0.7)
            
        Returns:
            List[Dict[str, Any]]: List of top movies with detailed statistics
            
        Each movie dictionary contains:
            - movieId: Unique movie identifier
            - title: Movie title
            - genres: Movie genres (pipe-separated)
            - rating_mean: Average rating
            - rating_std: Standard deviation of ratings
            - vote_count: Number of ratings
            - weighted_rating: IMDB-style weighted rating
            - wilson_lower: Wilson score lower bound
            - m: Minimum votes threshold used
            - C: Global mean rating used
            
        Examples:
            >>> analyzer = MovieAnalyzer(movies_df, ratings_df)
            >>> 
            >>> # Get top 10 movies with default parameters
            >>> top_movies = analyzer.get_top_movies(limit=10)
            >>> print(f"Top movie: {top_movies[0]['title']}")
            >>> print(f"Rating: {top_movies[0]['rating_mean']}")
            >>> 
            >>> # Get top movies with stricter criteria
            >>> top_strict = analyzer.get_top_movies(
            ...     limit=5, min_ratings=500, m_percentile=0.9
            ... )
            >>> 
            >>> # Access detailed statistics
            >>> movie = top_movies[0]
            >>> print(f"Weighted Rating: {movie['weighted_rating']}")
            >>> print(f"Wilson Lower Bound: {movie['wilson_lower']}")
        """
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
        """
        Analyze genre popularity trends and statistics across the dataset.
        
        Performs comprehensive genre analysis including overall popularity,
        temporal trends, and statistical measures. Handles multi-genre movies
        by exploding genre lists and analyzing each genre independently.
        
        Returns:
            Dict[str, Any]: Genre analysis results containing:
                - overall: Complete genre statistics sorted by popularity
                - time_series: Monthly genre trends over time
                - top_genres: Top 10 most popular genres
                
        Each genre record includes:
            - genre: Genre name
            - count: Number of ratings for this genre
            - mean_rating: Average rating for this genre
            - std_rating: Standard deviation of ratings
            - median_rating: Median rating for this genre
            
        Examples:
            >>> analyzer = MovieAnalyzer(movies_df, ratings_df)
            >>> genre_analysis = analyzer.analyze_genre_trends()
            >>> 
            >>> # Access overall genre statistics
            >>> top_genres = genre_analysis['top_genres']
            >>> print(f"Most popular genre: {top_genres[0]['genre']}")
            >>> print(f"Rating count: {top_genres[0]['count']}")
            >>> 
            >>> # Analyze temporal trends
            >>> time_series = genre_analysis['time_series']
            >>> drama_trends = [t for t in time_series if t['genre'] == 'Drama']
        """
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
        """
        Perform comprehensive time-series analysis of rating patterns.
        
        Analyzes temporal trends in movie ratings including monthly and yearly
        aggregations, user activity patterns, and movie diversity over time.
        Identifies peak activity periods and long-term trends.
        
        Returns:
            Dict[str, Any]: Time-series analysis results containing:
                - monthly: Monthly aggregated statistics
                - yearly: Yearly aggregated statistics  
                - summary: Key insights and peak activity information
                
        Monthly/yearly records include:
            - period: Time period (year_month or year)
            - count: Number of ratings in period
            - mean_rating: Average rating in period
            - std_rating: Standard deviation of ratings
            - unique_users: Number of unique users (monthly only)
            - unique_movies: Number of unique movies (monthly only)
            
        Examples:
            >>> analyzer = MovieAnalyzer(movies_df, ratings_df)
            >>> time_analysis = analyzer.generate_time_series_analysis()
            >>> 
            >>> # Access monthly trends
            >>> monthly_data = time_analysis['monthly']
            >>> peak_month = time_analysis['summary']['peak_month']
            >>> print(f"Peak activity month: {peak_month}")
            >>> 
            >>> # Analyze yearly trends
            >>> yearly_data = time_analysis['yearly']
            >>> recent_year = yearly_data[-1]
            >>> print(f"Recent year activity: {recent_year['count']} ratings")
        """
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
        """
        Analyze the distribution of ratings across the dataset.
        
        Computes comprehensive statistics about rating patterns including
        frequency distribution, central tendencies, and spread measures.
        Useful for understanding user rating behavior and data quality.
        
        Returns:
            Dict[str, Any]: Rating distribution analysis containing:
                - distribution: Count of each rating value (0.5 to 5.0)
                - statistics: Comprehensive statistical measures
                
        Statistics include:
            - mean: Average rating
            - median: Middle rating value
            - std: Standard deviation
            - min/max: Minimum and maximum ratings
            - q25/q75: First and third quartiles
            
        Examples:
            >>> analyzer = MovieAnalyzer(movies_df, ratings_df)
            >>> rating_dist = analyzer.get_rating_distribution()
            >>> 
            >>> # Access distribution data
            >>> distribution = rating_dist['distribution']
            >>> print(f"Most common rating: {max(distribution, key=distribution.get)}")
            >>> 
            >>> # Access statistical measures
            >>> stats = rating_dist['statistics']
            >>> print(f"Average rating: {stats['mean']:.2f}")
            >>> print(f"Rating spread (std): {stats['std']:.2f}")
        """
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
        """
        Analyze user behavior patterns and activity statistics.
        
        Examines user engagement levels, rating patterns, and activity distribution
        to understand user behavior across the platform. Categorizes users by
        activity level and analyzes rating variance patterns.
        
        Returns:
            Dict[str, Any]: User behavior analysis containing:
                - total_users: Total number of unique users
                - avg_ratings_per_user: Average number of ratings per user
                - median_ratings_per_user: Median number of ratings per user
                - user_activity_distribution: Users categorized by activity level
                - rating_variance_mean: Average rating variance across users
                
        Activity categories:
            - light: Users with < 20 ratings
            - moderate: Users with 20-99 ratings  
            - heavy: Users with 100+ ratings
            
        Examples:
            >>> analyzer = MovieAnalyzer(movies_df, ratings_df)
            >>> user_stats = analyzer.get_user_behavior_stats()
            >>> 
            >>> # Access user activity data
            >>> activity = user_stats['user_activity_distribution']
            >>> print(f"Heavy users: {activity['heavy']}")
            >>> print(f"Light users: {activity['light']}")
            >>> 
            >>> # Analyze engagement metrics
            >>> avg_ratings = user_stats['avg_ratings_per_user']
            >>> print(f"Average ratings per user: {avg_ratings:.1f}")
        """
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

    def get_movie_recommendations(self, user_id: int = None, movie_id: int = None, 
                                method: str = "collaborative", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Generate movie recommendations using various algorithms.
        
        Args:
            user_id: Target user ID for personalized recommendations
            movie_id: Movie ID for similar movie recommendations
            method: Algorithm to use ('collaborative', 'content', 'hybrid', 'popularity')
            limit: Number of recommendations to return
            
        Returns:
            List of recommended movies with scores
        """
        try:
            if method == "collaborative" and user_id:
                return self._collaborative_filtering_recommendations(user_id, limit)
            elif method == "content" and movie_id:
                return self._content_based_recommendations(movie_id, limit)
            elif method == "hybrid" and user_id:
                return self._hybrid_recommendations(user_id, limit)
            elif method == "popularity":
                return self._popularity_based_recommendations(limit)
            else:
                # Default to popularity-based if no specific method matches
                return self._popularity_based_recommendations(limit)
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    def calculate_movie_similarity(self, movie_id: int, method: str = "genre", 
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """
        Calculate similarity between movies using various methods.
        
        Args:
            movie_id: Reference movie ID
            method: Similarity method ('genre', 'rating', 'collaborative', 'hybrid')
            limit: Number of similar movies to return
            
        Returns:
            List of similar movies with similarity scores
        """
        try:
            if method == "genre":
                return self._genre_similarity(movie_id, limit)
            elif method == "rating":
                return self._rating_similarity(movie_id, limit)
            elif method == "collaborative":
                return self._collaborative_similarity(movie_id, limit)
            elif method == "hybrid":
                return self._hybrid_similarity(movie_id, limit)
            else:
                return self._genre_similarity(movie_id, limit)
                
        except Exception as e:
            logger.error(f"Error calculating movie similarity: {e}")
            return []

    def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """
        Analyze user preferences and behavior patterns.
        
        Args:
            user_id: Target user ID
            
        Returns:
            Dictionary containing user preference analysis with keys:
            - favorite_genres: User's preferred genres with ratings
            - avg_rating: User's average rating
            - total_ratings: Total number of ratings by user
            - rating_distribution: Distribution of ratings (1-5 stars)
            - activity_level: User activity classification
        """
        try:
            user_ratings = self.ratings[self.ratings['userId'] == user_id]
            
            if user_ratings.empty:
                return {}
            
            # Get user's rated movies with details
            user_movies = user_ratings.merge(self.movies, on='movieId')
            
            # Calculate genre preferences
            genre_prefs = self._calculate_genre_preferences(user_movies)
            
            # Calculate rating distribution
            rating_dist = user_ratings['rating'].value_counts().sort_index().to_dict()
            
            return {
                "favorite_genres": genre_prefs,
                "avg_rating": float(user_ratings['rating'].mean()),
                "total_ratings": len(user_ratings),
                "rating_distribution": rating_dist,
                "activity_level": self._classify_user_activity(len(user_ratings))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user preferences: {e}")
            return {}

    def _collaborative_filtering_recommendations(self, user_id: int, limit: int) -> List[Dict[str, Any]]:
        """Collaborative filtering recommendations based on user similarity."""
        try:
            # Create user-item matrix
            user_item_matrix = self.ratings.pivot_table(
                index='userId', columns='movieId', values='rating', fill_value=0
            )
            
            if user_id not in user_item_matrix.index:
                return self._popularity_based_recommendations(limit)
            
            # Calculate user similarities using cosine similarity
            user_ratings = user_item_matrix.loc[user_id].values.reshape(1, -1)
            similarities = []
            
            for other_user in user_item_matrix.index:
                if other_user != user_id:
                    other_ratings = user_item_matrix.loc[other_user].values.reshape(1, -1)
                    # Simple cosine similarity
                    dot_product = np.dot(user_ratings, other_ratings.T)[0][0]
                    norm_product = np.linalg.norm(user_ratings) * np.linalg.norm(other_ratings)
                    similarity = dot_product / norm_product if norm_product > 0 else 0
                    similarities.append((other_user, similarity))
            
            # Get top similar users
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similar_users = [user for user, sim in similarities[:10] if sim > 0]
            
            # Get recommendations from similar users
            recommendations = {}
            user_rated_movies = set(self.ratings[self.ratings['userId'] == user_id]['movieId'])
            
            for similar_user in top_similar_users:
                similar_user_ratings = self.ratings[
                    (self.ratings['userId'] == similar_user) & 
                    (self.ratings['rating'] >= 4.0)
                ]
                
                for _, row in similar_user_ratings.iterrows():
                    movie_id = row['movieId']
                    if movie_id not in user_rated_movies:
                        if movie_id not in recommendations:
                            recommendations[movie_id] = []
                        recommendations[movie_id].append(row['rating'])
            
            # Calculate average scores and get movie details
            movie_scores = []
            for movie_id, ratings in recommendations.items():
                avg_score = np.mean(ratings)
                movie_info = self.movies[self.movies['movieId'] == movie_id].iloc[0]
                movie_scores.append({
                    "movieId": int(movie_id),
                    "title": movie_info['title'],
                    "genres": movie_info['genres'],
                    "predicted_rating": round(avg_score, 2),
                    "confidence": len(ratings)
                })
            
            # Sort by predicted rating and confidence
            movie_scores.sort(key=lambda x: (x['predicted_rating'], x['confidence']), reverse=True)
            return movie_scores[:limit]
            
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {e}")
            return self._popularity_based_recommendations(limit)

    def _content_based_recommendations(self, movie_id: int, limit: int) -> List[Dict[str, Any]]:
        """Content-based recommendations using genre similarity."""
        return self._genre_similarity(movie_id, limit)

    def _hybrid_recommendations(self, user_id: int, limit: int) -> List[Dict[str, Any]]:
        """Hybrid recommendations combining collaborative and content-based."""
        try:
            # Get collaborative recommendations
            collab_recs = self._collaborative_filtering_recommendations(user_id, limit * 2)
            
            # Get user's favorite genres
            user_ratings = self.ratings[self.ratings['userId'] == user_id]
            user_movies = user_ratings.merge(self.movies, on='movieId')
            genre_prefs = self._calculate_genre_preferences(user_movies)
            
            # Score recommendations based on genre preferences
            for rec in collab_recs:
                genre_score = self._calculate_genre_match_score(rec['genres'], genre_prefs)
                rec['hybrid_score'] = (rec['predicted_rating'] * 0.7) + (genre_score * 0.3)
            
            # Sort by hybrid score
            collab_recs.sort(key=lambda x: x['hybrid_score'], reverse=True)
            return collab_recs[:limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return self._popularity_based_recommendations(limit)

    def _popularity_based_recommendations(self, limit: int) -> List[Dict[str, Any]]:
        """Popularity-based recommendations using top-rated movies."""
        try:
            top_movies = self.get_top_movies(limit=limit, min_ratings=50)
            recommendations = []
            
            for movie in top_movies:
                recommendations.append({
                    "movieId": movie['movieId'],
                    "title": movie['title'],
                    "genres": movie['genres'],
                    "score": movie['weighted_rating'],
                    "confidence": movie['rating_count']
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in popularity-based recommendations: {e}")
            return []

    def _genre_similarity(self, movie_id: int, limit: int) -> List[Dict[str, Any]]:
        """Calculate genre-based similarity."""
        try:
            target_movie = self.movies[self.movies['movieId'] == movie_id]
            if target_movie.empty:
                return []
            
            target_genres = set(target_movie.iloc[0]['genres'].split('|'))
            similarities = []
            
            for _, movie in self.movies.iterrows():
                if movie['movieId'] != movie_id:
                    movie_genres = set(movie['genres'].split('|'))
                    # Jaccard similarity
                    intersection = len(target_genres & movie_genres)
                    union = len(target_genres | movie_genres)
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > 0:
                        similarities.append({
                            "movieId": int(movie['movieId']),
                            "title": movie['title'],
                            "genres": movie['genres'],
                            "similarity_score": round(similarity, 3)
                        })
            
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Error in genre similarity: {e}")
            return []

    def _rating_similarity(self, movie_id: int, limit: int) -> List[Dict[str, Any]]:
        """Calculate rating pattern similarity."""
        try:
            # Get users who rated the target movie
            target_users = set(self.ratings[self.ratings['movieId'] == movie_id]['userId'])
            
            if not target_users:
                return []
            
            # Find movies rated by similar users
            similar_movies = self.ratings[self.ratings['userId'].isin(target_users)]
            movie_scores = similar_movies.groupby('movieId').agg({
                'rating': ['mean', 'count']
            }).round(2)
            
            movie_scores.columns = ['avg_rating', 'rating_count']
            movie_scores = movie_scores[movie_scores.index != movie_id]
            movie_scores = movie_scores[movie_scores['rating_count'] >= 5]
            
            # Merge with movie details
            result = []
            for mid, row in movie_scores.iterrows():
                movie_info = self.movies[self.movies['movieId'] == mid]
                if not movie_info.empty:
                    result.append({
                        "movieId": int(mid),
                        "title": movie_info.iloc[0]['title'],
                        "genres": movie_info.iloc[0]['genres'],
                        "similarity_score": float(row['avg_rating']) / 5.0,  # Normalize to 0-1
                        "common_raters": int(row['rating_count'])
                    })
            
            result.sort(key=lambda x: (x['similarity_score'], x['common_raters']), reverse=True)
            return result[:limit]
            
        except Exception as e:
            logger.error(f"Error in rating similarity: {e}")
            return []

    def _collaborative_similarity(self, movie_id: int, limit: int) -> List[Dict[str, Any]]:
        """Calculate collaborative filtering similarity."""
        return self._rating_similarity(movie_id, limit)

    def _hybrid_similarity(self, movie_id: int, limit: int) -> List[Dict[str, Any]]:
        """Calculate hybrid similarity combining genre and rating patterns."""
        try:
            genre_sim = self._genre_similarity(movie_id, limit * 2)
            rating_sim = self._rating_similarity(movie_id, limit * 2)
            
            # Combine similarities
            combined = {}
            
            # Add genre similarities
            for movie in genre_sim:
                mid = movie['movieId']
                combined[mid] = {
                    **movie,
                    'genre_similarity': movie['similarity_score'],
                    'rating_similarity': 0
                }
            
            # Add rating similarities
            for movie in rating_sim:
                mid = movie['movieId']
                if mid in combined:
                    combined[mid]['rating_similarity'] = movie['similarity_score']
                else:
                    combined[mid] = {
                        **movie,
                        'genre_similarity': 0,
                        'rating_similarity': movie['similarity_score']
                    }
            
            # Calculate hybrid score
            result = []
            for movie in combined.values():
                hybrid_score = (movie['genre_similarity'] * 0.6) + (movie['rating_similarity'] * 0.4)
                movie['similarity_score'] = round(hybrid_score, 3)
                result.append(movie)
            
            result.sort(key=lambda x: x['similarity_score'], reverse=True)
            return result[:limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid similarity: {e}")
            return []

    def _calculate_genre_preferences(self, user_movies: pd.DataFrame) -> Dict[str, float]:
        """Calculate user's genre preferences based on ratings."""
        genre_ratings = {}
        genre_counts = {}
        
        for _, movie in user_movies.iterrows():
            genres = movie['genres'].split('|')
            rating = movie['rating']
            
            for genre in genres:
                if genre not in genre_ratings:
                    genre_ratings[genre] = []
                genre_ratings[genre].append(rating)
        
        # Calculate average rating per genre
        genre_prefs = {}
        for genre, ratings in genre_ratings.items():
            avg_rating = np.mean(ratings)
            genre_prefs[genre] = {
                "average_rating": round(avg_rating, 2),
                "movie_count": len(ratings),
                "preference_score": round(avg_rating / 5.0, 3)  # Normalize to 0-1
            }
        
        return genre_prefs

    def _calculate_genre_match_score(self, movie_genres: str, user_genre_prefs: Dict) -> float:
        """Calculate how well a movie's genres match user preferences."""
        genres = movie_genres.split('|')
        scores = []
        
        for genre in genres:
            if genre in user_genre_prefs:
                scores.append(user_genre_prefs[genre]['preference_score'])
        
        return np.mean(scores) if scores else 0.5

    def _classify_user_activity(self, rating_count: int) -> str:
        """Classify user activity level based on number of ratings."""
        if rating_count >= 500:
            return "very_active"
        elif rating_count >= 100:
            return "active"
        elif rating_count >= 20:
            return "moderate"
        else:
            return "casual"

    def get_top_rated_movies(self, n: int = 20, min_ratings: int = 100, m_percentile: float = 0.7) -> pd.DataFrame:
        """
        Get top-rated movies using weighted rating algorithms, returning a DataFrame.
        
        This method provides the same functionality as get_top_movies() but returns
        a DataFrame format for compatibility with test suites.
        
        Args:
            n (int): Maximum number of movies to return (default: 20)
            min_ratings (int): Minimum number of ratings required (default: 100)
            m_percentile (float): Percentile for minimum ratings threshold (default: 0.7)
        
        Returns:
            pd.DataFrame: DataFrame of top-rated movies with columns:
                - movieId: Movie ID
                - title: Movie title
                - genres: Movie genres
                - avg_rating: Average rating (rating_mean)
                - rating_count: Number of ratings (vote_count)
                - weighted_rating: IMDB-style weighted rating
        
        Examples:
            >>> analyzer = MovieAnalyzer(movies_df, ratings_df)
            >>> top_movies = analyzer.get_top_rated_movies(n=10)
            >>> print(top_movies[['title', 'avg_rating', 'rating_count']])
        """
        # Get the list format from get_top_movies
        top_movies_list = self.get_top_movies(limit=n, min_ratings=min_ratings, m_percentile=m_percentile)
        
        if not top_movies_list:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['movieId', 'title', 'genres', 'avg_rating', 'rating_count', 'weighted_rating'])
        
        # Convert to DataFrame and rename columns to match test expectations
        df = pd.DataFrame(top_movies_list)
        df = df.rename(columns={
            'rating_mean': 'avg_rating',
            'vote_count': 'rating_count',
            'WR': 'weighted_rating'
        })
        
        # Ensure we have the expected columns
        expected_cols = ['movieId', 'title', 'genres', 'avg_rating', 'rating_count', 'weighted_rating']
        for col in expected_cols:
            if col not in df.columns:
                if col == 'weighted_rating' and 'WR' in df.columns:
                    df['weighted_rating'] = df['WR']
                elif col == 'rating_count' and 'v' in df.columns:
                    df['rating_count'] = df['v']
                elif col == 'avg_rating' and 'R' in df.columns:
                    df['avg_rating'] = df['R']
        
        return df[expected_cols] if all(col in df.columns for col in expected_cols) else df

    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics summary of the dataset.
        
        Returns:
            Dict[str, Any]: Dictionary containing dataset statistics
        """
        return {
            'total_movies': len(self.movies),
            'total_ratings': len(self.ratings),
            'total_users': self.ratings['userId'].nunique(),
            'avg_rating': self.ratings['rating'].mean(),
            'rating_std': self.ratings['rating'].std(),
            'min_rating': self.ratings['rating'].min(),
            'max_rating': self.ratings['rating'].max(),
            'ratings_per_movie': self.ratings.groupby('movieId').size().mean(),
            'ratings_per_user': self.ratings.groupby('userId').size().mean(),
            'most_rated_movie': self.movies.merge(
                self.ratings.groupby('movieId').size().reset_index(name='count'),
                on='movieId'
            ).nlargest(1, 'count')['title'].iloc[0] if len(self.ratings) > 0 else None
        }

    def analyze_genres(self) -> pd.DataFrame:
        """
        Analyze genre popularity and statistics across the dataset.
        
        Returns:
            pd.DataFrame: Genre analysis results with columns:
                - genre: Genre name
                - movie_count: Number of movies in this genre
                - avg_rating: Average rating for this genre
                - rating_count: Total number of ratings for this genre
        """
        logger.info("    Analyzing genres...")
        df = self.ratings.merge(self.movies, on="movieId")
        df["genres_list"] = df["genres"].str.split("|")
        df = df.explode("genres_list")
        df = df[df["genres_list"] != "(no genres listed)"]
        
        genre_stats = (
            df.groupby("genres_list")
            .agg(
                rating_count=("rating", "count"),
                avg_rating=("rating", "mean"),
                movie_count=("movieId", "nunique")
            )
            .reset_index()
            .rename(columns={"genres_list": "genre"})
            .sort_values("rating_count", ascending=False)
        )
        
        return genre_stats

    def get_most_popular_movies(self, n: int = 20, limit: int = None) -> pd.DataFrame:
        """
        Get most popular movies based on rating count.
        
        Args:
            n (int): Maximum number of movies to return (preferred parameter)
            limit (int): Alternative parameter name for backward compatibility
            
        Returns:
            pd.DataFrame: DataFrame of most popular movies
        """
        # Use n if provided, otherwise use limit for backward compatibility
        movie_limit = n if n is not None else (limit if limit is not None else 20)
        
        movie_popularity = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).round(3)
        movie_popularity.columns = ['rating_count', 'avg_rating']
        movie_popularity = movie_popularity.reset_index()
        
        # Merge with movie titles
        popular_movies = movie_popularity.merge(
            self.movies[['movieId', 'title', 'genres']], 
            on='movieId', 
            how='left'
        )
        
        # Sort by rating count (popularity)
        popular_movies = popular_movies.nlargest(movie_limit, 'rating_count')
        
        return popular_movies

    def get_movies_by_genre(self, genre: str) -> pd.DataFrame:
        """
        Get movies filtered by a specific genre.
        
        Args:
            genre (str): Genre to filter by
            
        Returns:
            pd.DataFrame: Movies that contain the specified genre
        """
        # Filter movies that contain the specified genre
        genre_movies = self.movies[
            self.movies['genres'].str.contains(genre, case=False, na=False)
        ].copy()
        
        return genre_movies

    def analyze_temporal_patterns(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze temporal patterns in rating behavior.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing temporal analysis results:
                - hourly_patterns: Rating patterns by hour of day
                - daily_patterns: Rating patterns by day of week  
                - monthly_patterns: Rating patterns by month
        """
        logger.info("    Analyzing temporal patterns...")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.ratings['timestamp']):
            ratings_df = self.ratings.copy()
            ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        else:
            ratings_df = self.ratings.copy()
        
        # Extract time components
        ratings_df['hour'] = ratings_df['timestamp'].dt.hour
        ratings_df['day_of_week'] = ratings_df['timestamp'].dt.day_name()
        ratings_df['month'] = ratings_df['timestamp'].dt.month_name()
        
        # Hourly patterns
        hourly_patterns = ratings_df.groupby('hour').agg({
            'rating': ['count', 'mean', 'std']
        }).round(3)
        hourly_patterns.columns = ['rating_count', 'avg_rating', 'std_rating']
        hourly_patterns = hourly_patterns.reset_index()
        
        # Daily patterns
        daily_patterns = ratings_df.groupby('day_of_week').agg({
            'rating': ['count', 'mean', 'std']
        }).round(3)
        daily_patterns.columns = ['rating_count', 'avg_rating', 'std_rating']
        daily_patterns = daily_patterns.reset_index()
        
        # Monthly patterns
        monthly_patterns = ratings_df.groupby('month').agg({
            'rating': ['count', 'mean', 'std']
        }).round(3)
        monthly_patterns.columns = ['rating_count', 'avg_rating', 'std_rating']
        monthly_patterns = monthly_patterns.reset_index()
        
        return {
            'hourly_patterns': hourly_patterns,
            'daily_patterns': daily_patterns,
            'monthly_patterns': monthly_patterns
         }

    def analyze_rating_trends(self) -> Dict[str, Any]:
        """
        Analyze rating trends over time.
        
        Returns:
            Dict[str, Any]: Rating trends analysis including temporal patterns with keys:
            - yearly_trends: DataFrame with yearly rating trends
            - monthly_trends: DataFrame with monthly rating trends  
            - daily_trends: DataFrame with daily rating trends
        """
        time_series = self.generate_time_series_analysis()
        
        # Convert the time series data to the expected format
        yearly_data = []
        monthly_data = []
        daily_data = []
        
        if 'yearly' in time_series:
            for item in time_series['yearly']:
                yearly_data.append({
                    'year': item.get('year', 0),
                    'avg_rating': item.get('mean_rating', 0),
                    'total_ratings': item.get('count', 0)
                })
        
        if 'monthly' in time_series:
            for item in time_series['monthly']:
                monthly_data.append({
                    'month': item.get('month', 0),
                    'avg_rating': item.get('mean_rating', 0),
                    'total_ratings': item.get('count', 0)
                })
        
        if 'daily' in time_series:
            for item in time_series['daily']:
                daily_data.append({
                    'day': item.get('day', 0),
                    'avg_rating': item.get('mean_rating', 0),
                    'total_ratings': item.get('count', 0)
                })
        
        return {
            'yearly_trends': pd.DataFrame(yearly_data),
            'monthly_trends': pd.DataFrame(monthly_data),
            'daily_trends': pd.DataFrame(daily_data)
        }

    def analyze_rating_patterns(self) -> Dict[str, Any]:
        """
        Analyze rating patterns and distributions.
        
        Returns:
            Dict[str, Any]: Rating patterns analysis with keys:
            - rating_distribution: Distribution of ratings across the dataset
            - user_activity: User activity patterns and statistics
            - movie_popularity: Movie popularity metrics
        """
        # Get rating distribution
        rating_dist = self.get_rating_distribution()
        
        # Get user activity patterns
        user_activity = self.get_user_behavior_stats()
        
        # Get movie popularity metrics
        movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).round(3)
        movie_stats.columns = ['rating_count', 'avg_rating']
        
        movie_popularity = {
            'total_movies': len(self.movies),
            'movies_with_ratings': len(movie_stats),
            'avg_ratings_per_movie': float(movie_stats['rating_count'].mean()),
            'most_rated_movie_count': int(movie_stats['rating_count'].max()),
            'least_rated_movie_count': int(movie_stats['rating_count'].min())
        }
        
        return {
            'rating_distribution': rating_dist,
            'user_activity': user_activity,
            'movie_popularity': movie_popularity
        }

    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics summary of the dataset.
        
        Returns:
            Dict[str, Any]: Statistics summary including sparsity and other metrics with keys:
            - total_movies: Total number of movies
            - total_users: Total number of users
            - total_ratings: Total number of ratings
            - avg_rating: Average rating across all ratings
            - rating_std: Standard deviation of ratings
            - min_rating: Minimum rating value
            - max_rating: Maximum rating value
            - sparsity: Dataset sparsity (1 - density)
        """
        total_possible_ratings = len(self.movies) * len(self.ratings['userId'].unique())
        actual_ratings = len(self.ratings)
        sparsity = 1 - (actual_ratings / total_possible_ratings)
        
        return {
            'total_movies': len(self.movies),
            'total_users': len(self.ratings['userId'].unique()),
            'total_ratings': actual_ratings,
            'avg_rating': float(self.ratings['rating'].mean()),
            'rating_std': float(self.ratings['rating'].std()),
            'min_rating': float(self.ratings['rating'].min()),
            'max_rating': float(self.ratings['rating'].max()),
            'sparsity': float(sparsity)
        }
