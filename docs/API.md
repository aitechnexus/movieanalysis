# 🔧 MovieLens Analysis Platform API Documentation

<div align="center">

![API Version](https://img.shields.io/badge/API-v1.0.0-blue.svg)
![OpenAPI](https://img.shields.io/badge/OpenAPI-3.0.3-green.svg)
![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)

**Comprehensive REST API for MovieLens data analysis and recommendations**

[🚀 Quick Start](#-quick-start) • [📋 Endpoints](#-endpoints) • [🔍 Examples](#-examples) • [🧪 Testing](#-testing)

</div>

---

## 🌟 Overview

The MovieLens Analysis Platform provides a comprehensive REST API for accessing movie data, analytics, and recommendations. This API enables programmatic access to all platform features including data retrieval, statistical analysis, and visualization generation.

### ✨ Key Features

- **🎬 Movie Data Access**: Complete movie catalog with ratings and metadata
- **📊 Advanced Analytics**: Statistical analysis and trend identification  
- **🤖 Recommendations**: Personalized movie recommendations
- **📈 Visualizations**: Dynamic chart and graph generation
- **⚡ High Performance**: Optimized queries with intelligent caching
- **🔒 Reliable**: Comprehensive error handling and validation

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [🔧 Base Information](#-base-information)
- [📋 Endpoints](#-endpoints)
  - [System Status](#system-status)
  - [Movies](#movies)
  - [Users & Recommendations](#users--recommendations)
  - [Analytics & Statistics](#analytics--statistics)
  - [Visualizations](#visualizations)
  - [System Management](#system-management)
- [🔍 Examples](#-examples)
- [❌ Error Handling](#-error-handling)
- [🧪 Testing](#-testing)
- [📚 SDKs & Libraries](#-sdks--libraries)

## 🚀 Quick Start

### Installation & Setup

```bash
# Start the API server
docker compose up --build

# Verify API is running
curl http://localhost:8001/api/status
```

### First API Call

```python
import requests

# Get system status
response = requests.get('http://localhost:8001/api/status')
print(response.json())

# Get top movies
response = requests.get('http://localhost:8001/api/movies/top?n=5')
top_movies = response.json()['data']['top_movies']
print(f"Top movie: {top_movies[0]['title']}")
```

### Authentication

Currently, no authentication is required for development. In production environments, implement proper authentication using API keys or OAuth 2.0.

## 🔧 Base Information

| Property | Value |
|----------|-------|
| **Base URL** | `http://localhost:8001/api` |
| **Content Type** | `application/json` |
| **Authentication** | None (development) |
| **Rate Limiting** | 100 requests/minute/IP |
| **API Version** | v1.0.0 |
| **OpenAPI Spec** | Available at `/api/docs` |

### Request Headers

```http
Content-Type: application/json
Accept: application/json
User-Agent: YourApp/1.0.0
```

## 📨 Response Format

All API responses follow a consistent JSON structure for predictable integration.

### ✅ Success Response

```json
{
  "success": true,
  "data": {
    // Response data here
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "execution_time": "0.123s",
    "api_version": "1.0.0",
    "request_id": "req_abc123"
  }
}
```

### ❌ Error Response

```json
{
  "success": false,
  "error": {
    "message": "Detailed error description",
    "code": "VALIDATION_ERROR",
    "type": "client_error",
    "details": {
      "field": "min_rating",
      "value": "invalid",
      "expected": "float between 0.0 and 5.0"
    }
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

### 📄 Pagination Response

```json
{
  "success": true,
  "data": {
    "items": [...],
    "pagination": {
      "total": 1500,
      "page": 1,
      "per_page": 50,
      "total_pages": 30,
      "has_next": true,
      "has_prev": false,
      "next_url": "/api/movies?page=2&limit=50",
      "prev_url": null
    }
  }
}
```

## 📋 Endpoints

### System Status

#### 🔍 GET /api/status

Returns comprehensive system health and status information.

**URL**: `GET /api/status`

**Response Example**:
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": "2h 15m 30s",
    "services": {
      "database": "connected",
      "cache": "active",
      "data_loader": "ready"
    },
    "statistics": {
      "total_movies": 62423,
      "total_ratings": 25000095,
      "total_users": 162541,
      "last_data_refresh": "2024-01-15T08:15:00Z"
    },
    "performance": {
      "memory_usage": "45%",
      "cpu_usage": "12%",
      "cache_hit_rate": "94.2%",
      "avg_response_time": "0.089s"
    }
  }
}
```

**Status Codes**:
- `200`: System is healthy and operational
- `503`: System is unavailable or experiencing issues

---

### Movies

#### 🎬 GET /api/movies

Retrieve a paginated list of movies with advanced filtering and sorting options.

**URL**: `GET /api/movies`

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 50 | Number of movies to return (max: 1000) |
| `offset` | int | 0 | Number of movies to skip |
| `page` | int | 1 | Page number (alternative to offset) |
| `genre` | string | - | Filter by genre (e.g., "Action", "Comedy") |
| `min_rating` | float | - | Minimum average rating (0.0-5.0) |
| `max_rating` | float | - | Maximum average rating (0.0-5.0) |
| `year` | int | - | Filter by release year |
| `decade` | string | - | Filter by decade (e.g., "1990s", "2000s") |
| `min_votes` | int | - | Minimum number of ratings |
| `search` | string | - | Search in movie titles |
| `sort_by` | string | "title" | Sort field: "title", "rating", "year", "votes", "popularity" |
| `sort_order` | string | "asc" | Sort order: "asc", "desc" |

**Example Requests**:

```bash
# Get top-rated action movies from the 1990s
GET /api/movies?genre=Action&decade=1990s&min_rating=4.0&sort_by=rating&sort_order=desc&limit=10

# Search for movies with "star" in the title
GET /api/movies?search=star&sort_by=popularity&sort_order=desc

# Get recent movies with many ratings
GET /api/movies?year=2020&min_votes=1000&sort_by=votes&sort_order=desc
```

**Response Example**:
```json
{
  "success": true,
  "data": {
    "movies": [
      {
        "movieId": 318,
        "title": "The Shawshank Redemption (1994)",
        "genres": ["Crime", "Drama"],
        "year": 1994,
        "avg_rating": 4.45,
        "rating_count": 311,
        "popularity_score": 0.95,
        "imdb_weighted_rating": 4.42,
        "wilson_score": 4.38
      }
    ],
    "pagination": {
      "total": 1500,
      "page": 1,
      "per_page": 10,
      "total_pages": 150,
      "has_next": true,
      "has_prev": false,
      "next_url": "/api/movies?page=2&limit=10&genre=Action",
      "prev_url": null
    },
    "filters_applied": {
      "genre": "Action",
      "decade": "1990s",
      "min_rating": 4.0
    },
    "sort_applied": {
      "field": "rating",
      "order": "desc"
    }
  }
}
```

**Status Codes**:
- `200`: Success - Movies retrieved successfully
- `400`: Bad Request - Invalid parameters
- `404`: Not Found - No movies match the criteria
- `500`: Internal Server Error

---

#### 🏆 GET /api/movies/top

Get top-rated movies based on average rating and minimum vote threshold.

**URL**: `GET /api/movies/top`

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | int | 10 | Number of movies to return (max: 100) |
| `min_ratings` | int | 50 | Minimum number of ratings required |
| `genre` | string | - | Filter by specific genre |
| `algorithm` | string | "weighted" | Rating algorithm: "simple", "weighted", "wilson" |

**Example Requests**:

```bash
# Get top 5 drama movies with at least 100 ratings
GET /api/movies/top?n=5&min_ratings=100&genre=Drama

# Get top movies using Wilson Score algorithm
GET /api/movies/top?n=10&algorithm=wilson&min_ratings=200
```

**Response Example**:

```json
{
  "success": true,
  "data": {
    "top_movies": [
      {
        "movieId": 318,
        "title": "The Shawshank Redemption (1994)",
        "genres": ["Crime", "Drama"],
        "year": 1994,
        "avg_rating": 4.45,
        "rating_count": 311,
        "rank": 1,
        "weighted_score": 4.42,
        "confidence_interval": [4.38, 4.52]
      },
      {
        "movieId": 858,
        "title": "The Godfather (1972)",
        "genres": ["Crime", "Drama"],
        "year": 1972,
        "avg_rating": 4.29,
        "rating_count": 192,
        "rank": 2,
        "weighted_score": 4.25,
        "confidence_interval": [4.18, 4.40]
      }
    ],
    "metadata": {
      "algorithm_used": "weighted",
      "min_ratings_threshold": 100,
      "genre_filter": "Drama",
      "total_qualifying_movies": 245
    }
  }
}
```

**Status Codes**:
- `200`: Success - Top movies retrieved
- `400`: Bad Request - Invalid parameters
- `404`: Not Found - No movies match criteria
- `500`: Internal Server Error

---

#### 🔥 GET /api/movies/popular

Get most popular movies based on rating count, recency, and engagement metrics.

**URL**: `GET /api/movies/popular`

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | int | 10 | Number of movies to return (max: 100) |
| `time_period` | string | "all" | Time period: "all", "year", "month", "week" |
| `genre` | string | - | Filter by specific genre |
| `min_year` | int | - | Minimum release year |
| `max_year` | int | - | Maximum release year |

**Example Requests**:

```bash
# Get popular movies from the last year
GET /api/movies/popular?time_period=year&n=20

# Get popular action movies from 2000s
GET /api/movies/popular?genre=Action&min_year=2000&max_year=2009
```

**Response Example**:

```json
{
  "success": true,
  "data": {
    "popular_movies": [
      {
        "movieId": 356,
        "title": "Forrest Gump (1994)",
        "genres": ["Comedy", "Drama", "Romance"],
        "year": 1994,
        "rating_count": 329,
        "avg_rating": 4.16,
        "popularity_rank": 1,
        "popularity_score": 0.92,
        "trending_factor": 1.15,
        "recent_activity": {
          "ratings_last_30_days": 45,
          "growth_rate": 0.08
        }
      }
    ],
    "metadata": {
      "time_period": "all",
      "total_movies_considered": 9742,
      "popularity_algorithm": "engagement_weighted"
    }
  }
}
```

**Status Codes**:
- `200`: Success - Popular movies retrieved
- `400`: Bad Request - Invalid time period or parameters
- `500`: Internal Server Error

---

#### 🎬 GET /api/movies/{movie_id}

Get comprehensive information about a specific movie including ratings, statistics, and recommendations.

**URL**: `GET /api/movies/{movie_id}`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `movie_id` | int | Yes | The unique movie identifier |

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_similar` | boolean | false | Include similar movie recommendations |
| `include_stats` | boolean | true | Include detailed statistics |
| `include_reviews` | boolean | false | Include sample reviews (if available) |

**Example Requests**:

```bash
# Get basic movie information
GET /api/movies/1

# Get movie with similar recommendations
GET /api/movies/1?include_similar=true&include_stats=true
```

```json
{
  "success": true,
  "data": {
    "movie": {
      "movieId": 1,
      "title": "Toy Story (1995)",
      "genres": ["Adventure", "Animation", "Children", "Comedy", "Fantasy"],
      "year": 1995,
      "avg_rating": 3.92,
      "rating_count": 215,
      "rating_distribution": {
        "1": 2,
        "2": 7,
        "3": 27,
        "4": 118,
        "5": 61
      },
      "statistics": {
        "percentile_rank": 78.5,
        "genre_rank": {
          "Animation": 12,
          "Children": 5
        },
        "rating_trend": "stable",
        "popularity_score": 0.73
      },
      "genre_analysis": {
        "primary_genre": "Animation",
        "genre_popularity": 0.85,
        "cross_genre_appeal": 0.67
      }
    },
    "similar_movies": [
      {
        "movieId": 3114,
        "title": "Toy Story 2 (1999)",
        "similarity_score": 0.92,
        "genres": ["Animation", "Children", "Comedy"]
      }
    ],
    "metadata": {
      "last_updated": "2024-01-15T10:30:00Z",
      "data_completeness": 0.95
    }
  }
}
```

**Status Codes**:
- `200`: Success - Movie details retrieved
- `404`: Not Found - Movie ID does not exist
- `500`: Internal Server Error

---

#### 🔍 GET /api/movies/{movie_id}/similar

Find movies similar to the specified movie using various similarity algorithms.

**URL**: `GET /api/movies/{movie_id}/similar`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `movie_id` | int | Yes | The movie ID to find similarities for |

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | int | 5 | Number of similar movies to return (max: 50) |
| `method` | string | "hybrid" | Similarity method: "genre", "rating", "collaborative", "hybrid" |
| `min_similarity` | float | 0.1 | Minimum similarity score threshold |
| `exclude_same_year` | boolean | false | Exclude movies from the same year |

**Example Requests**:

```bash
# Get similar movies using hybrid algorithm
GET /api/movies/1/similar?n=10&method=hybrid

# Get genre-based similarities with high threshold
GET /api/movies/1/similar?method=genre&min_similarity=0.7&n=5
```

**Response Example**:

```json
{
  "success": true,
  "data": {
    "similar_movies": [
      {
        "movieId": 3114,
        "title": "Toy Story 2 (1999)",
        "genres": ["Animation", "Children", "Comedy"],
        "year": 1999,
        "similarity_score": 0.92,
        "similarity_factors": {
          "genre_match": 0.95,
          "rating_pattern": 0.88,
          "user_overlap": 0.73
        },
        "avg_rating": 3.95,
        "rating_count": 198
      }
    ],
    "base_movie": {
      "movieId": 1,
      "title": "Toy Story (1995)",
      "genres": ["Adventure", "Animation", "Children", "Comedy", "Fantasy"]
    },
    "algorithm_info": {
      "method_used": "hybrid",
      "weights": {
        "genre_similarity": 0.4,
        "rating_similarity": 0.3,
        "collaborative_filtering": 0.3
      }
    }
  }
}
```

**Status Codes**:
- `200`: Success - Similar movies found
- `404`: Not Found - Base movie ID does not exist
- `400`: Bad Request - Invalid similarity method or parameters
- `500`: Internal Server Error

---

## 👥 User Endpoints

#### 🎯 GET /api/users/{user_id}/preferences

Get comprehensive user preferences, viewing patterns, and personalized insights.

**URL**: `GET /api/users/{user_id}/preferences`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | int | Yes | The unique user identifier |

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_history` | boolean | false | Include rating history |
| `time_range` | string | "all" | Time range: "all", "year", "6months", "month" |
| `detailed_genres` | boolean | true | Include detailed genre breakdown |

**Example Requests**:

```bash
# Get basic user preferences
GET /api/users/123/preferences

# Get detailed preferences with history
GET /api/users/123/preferences?include_history=true&detailed_genres=true
```

```json
{
  "success": true,
  "data": {
    "user_preferences": {
      "userId": 123,
      "profile_summary": {
        "total_ratings": 45,
        "avg_rating_given": 3.8,
        "rating_variance": 1.2,
        "activity_level": "moderate"
      },
      "genre_preferences": [
        {
          "genre": "Drama",
          "count": 15,
          "avg_rating": 4.2,
          "preference_score": 0.89,
          "trend": "increasing"
        },
        {
          "genre": "Action", 
          "count": 12,
          "avg_rating": 3.9,
          "preference_score": 0.76,
          "trend": "stable"
        },
        {
          "genre": "Comedy",
          "count": 8,
          "avg_rating": 3.5,
          "preference_score": 0.62,
          "trend": "decreasing"
        }
      ],
      "rating_patterns": {
        "distribution": {
          "1": 1,
          "2": 3,
          "3": 8,
          "4": 20,
          "5": 13
        },
        "rating_tendency": "generous",
        "consistency_score": 0.73
      },
      "temporal_analysis": {
        "first_rating": "2023-01-15T14:30:00Z",
        "last_rating": "2024-01-10T09:15:00Z",
        "most_active_period": "March 2023",
        "rating_frequency": "2.1 per week",
        "seasonal_patterns": {
          "winter": 0.85,
          "spring": 1.15,
          "summer": 0.95,
          "fall": 1.05
        }
      },
      "behavioral_insights": {
        "discovery_preference": "popular_recent",
        "genre_diversity": 0.67,
        "rating_predictability": 0.78,
        "binge_tendency": "moderate"
      }
    }
  }
}
```

**Status Codes**:
- `200`: Success - User preferences retrieved
- `404`: Not Found - User ID does not exist
- `403`: Forbidden - Access denied to user data
- `500`: Internal Server Error

---

#### 🎬 GET /api/users/{user_id}/recommendations

Get personalized movie recommendations using advanced collaborative filtering and content-based algorithms.

**URL**: `GET /api/users/{user_id}/recommendations`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | int | Yes | The unique user identifier |

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | int | 10 | Number of recommendations (max: 100) |
| `method` | string | "hybrid" | Algorithm: "collaborative", "content", "hybrid", "popularity" |
| `genre_filter` | string | - | Filter by specific genre |
| `min_year` | int | - | Minimum release year |
| `max_year` | int | - | Maximum release year |
| `exclude_rated` | boolean | true | Exclude already rated movies |
| `diversity_factor` | float | 0.3 | Recommendation diversity (0.0-1.0) |

**Example Requests**:

```bash
# Get hybrid recommendations
GET /api/users/123/recommendations?n=20&method=hybrid

# Get diverse action movie recommendations
GET /api/users/123/recommendations?genre_filter=Action&diversity_factor=0.8&n=15
```

**Response Example**:
```json
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "movieId": 2571,
        "title": "The Matrix (1999)",
        "genres": ["Action", "Sci-Fi", "Thriller"],
        "year": 1999,
        "predicted_rating": 4.3,
        "confidence": 0.87,
        "recommendation_score": 0.94,
        "reasoning": {
          "primary_factor": "collaborative_filtering",
          "explanation": "Users with similar preferences rated this highly",
          "supporting_factors": [
            "Genre match with user preferences",
            "High rating from similar users",
            "Popular in user's demographic"
          ]
        },
        "metadata": {
          "avg_rating": 4.2,
          "rating_count": 278,
          "popularity_rank": 15
        }
      },
      {
        "movieId": 1196,
        "title": "Star Wars: Episode V - The Empire Strikes Back (1980)",
        "genres": ["Action", "Adventure", "Sci-Fi"],
        "year": 1980,
        "predicted_rating": 4.1,
        "confidence": 0.82,
        "recommendation_score": 0.89,
        "reasoning": {
          "primary_factor": "content_based",
          "explanation": "Strong genre alignment with your preferences",
          "supporting_factors": [
            "High-rated Sci-Fi movie",
            "Classic status",
            "Similar to highly-rated movies you've enjoyed"
          ]
        },
        "metadata": {
          "avg_rating": 4.0,
          "rating_count": 251,
          "popularity_rank": 8
        }
      }
    ],
    "algorithm_details": {
      "method_used": "hybrid",
      "weights": {
        "collaborative_filtering": 0.6,
        "content_based": 0.3,
        "popularity_boost": 0.1
      },
      "user_profile_strength": 0.75,
      "total_candidates_considered": 1500,
      "diversity_score": 0.67
    },
    "personalization_insights": {
      "top_recommended_genres": ["Sci-Fi", "Action", "Drama"],
      "recommendation_novelty": 0.73,
      "serendipity_factor": 0.45
    }
  }
}
```

**Status Codes**:
- `200`: Success - Recommendations generated
- `404`: Not Found - User ID does not exist
- `400`: Bad Request - Invalid parameters
- `403`: Forbidden - Insufficient user data for recommendations
- `500`: Internal Server Error

---

## 📊 Analytics & Statistics

#### 📈 GET /api/statistics

Get comprehensive platform statistics including movies, ratings, users, and trends.

**URL**: `GET /api/statistics`

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_trends` | boolean | false | Include temporal trend analysis |
| `include_demographics` | boolean | false | Include user demographic insights |
| `time_range` | string | "all" | Time range: "all", "year", "month" |

**Example Requests**:

```bash
# Get basic statistics
GET /api/statistics

# Get comprehensive statistics with trends
GET /api/statistics?include_trends=true&include_demographics=true
```

**Response Example**:
```json
{
  "success": true,
  "data": {
    "platform_overview": {
      "movies": {
        "total_count": 9742,
        "avg_rating": 3.52,
        "most_common_genre": "Drama",
        "year_range": [1902, 2018],
        "recent_additions": 156,
        "quality_score": 0.78
      },
      "ratings": {
        "total_count": 100836,
        "unique_users": 610,
        "avg_ratings_per_user": 165.3,
        "median_ratings_per_user": 96,
        "rating_density": 0.017,
        "rating_distribution": {
          "0.5": 1370,
          "1.0": 2811,
          "1.5": 1791,
          "2.0": 7551,
          "2.5": 5550,
          "3.0": 20047,
          "3.5": 13136,
          "4.0": 26818,
          "4.5": 8551,
          "5.0": 13211
        }
      },
      "user_engagement": {
        "active_users_last_30_days": 245,
        "avg_session_duration": "24.5 minutes",
        "retention_rate": 0.73,
        "power_users": 89
      }
    },
    "content_insights": {
      "genre_distribution": {
        "Drama": 4361,
        "Comedy": 3756,
        "Thriller": 1894,
        "Action": 1828,
        "Romance": 1596
      },
      "decade_analysis": {
        "most_productive": "1990s",
        "highest_rated": "1950s",
        "most_diverse": "2000s"
      },
      "quality_metrics": {
        "movies_above_4_stars": 1247,
        "critically_acclaimed": 892,
        "hidden_gems": 234
      }
    },
    "temporal_trends": {
      "rating_evolution": {
        "trend": "slightly_increasing",
        "annual_growth": 0.02,
        "seasonal_patterns": {
          "peak_months": ["November", "December"],
          "low_months": ["June", "July"]
        }
      },
      "genre_popularity_shifts": {
        "rising": ["Sci-Fi", "Animation"],
        "stable": ["Drama", "Comedy"],
        "declining": ["Western", "Musical"]
      }
    }
  }
}
```

**Status Codes**:
- `200`: Success - Statistics retrieved
- `500`: Internal Server Error

---

#### 🎭 GET /api/genres

Get comprehensive genre analysis, statistics, and trends.

**URL**: `GET /api/genres`

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_trends` | boolean | false | Include temporal trend analysis |
| `include_combinations` | boolean | true | Include genre combination analysis |
| `min_movie_count` | int | 10 | Minimum movies required for genre inclusion |
| `sort_by` | string | "popularity" | Sort by: "popularity", "rating", "count", "name" |

**Example Requests**:

```bash
# Get basic genre statistics
GET /api/genres

# Get comprehensive genre analysis with trends
GET /api/genres?include_trends=true&include_combinations=true&sort_by=rating
```

**Response Example**:

```json
{
  "success": true,
  "data": {
    "genres": [
      {
        "genre": "Drama",
        "statistics": {
          "movie_count": 4361,
          "avg_rating": 3.68,
          "total_ratings": 45023,
          "popularity_rank": 1,
          "rating_variance": 0.85,
          "user_engagement": 0.89
        },
        "trends": {
          "rating_trend": "stable",
          "popularity_trend": "increasing",
          "growth_rate": 0.05,
          "peak_years": [1995, 2001, 2008]
        },
        "demographics": {
          "avg_user_age_preference": 32.5,
          "gender_preference": {
            "male": 0.52,
            "female": 0.48
          }
        }
      }
    ],
    "genre_combinations": [
      {
        "combination": ["Action", "Adventure"],
        "movie_count": 303,
        "avg_rating": 3.45,
        "popularity_score": 0.78,
        "synergy_factor": 1.23
      },
      {
        "combination": ["Comedy", "Romance"],
        "movie_count": 287,
        "avg_rating": 3.52,
        "popularity_score": 0.71,
        "synergy_factor": 1.15
      }
    ],
    "insights": {
      "most_versatile_genre": "Drama",
      "best_combination": ["Action", "Adventure"],
      "emerging_trends": ["Sci-Fi + Thriller", "Animation + Adventure"]
    }
  }
}
```

**Status Codes**:
- `200`: Success - Genre data retrieved
- `400`: Bad Request - Invalid sort parameter
- `500`: Internal Server Error

---

#### 📊 GET /api/trends

Get comprehensive rating and popularity trends over time with advanced analytics.

**URL**: `GET /api/trends`

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | string | "monthly" | Time period: "daily", "weekly", "monthly", "yearly", "decade" |
| `year` | int | - | Specific year for detailed analysis |
| `genre` | string | - | Filter trends by specific genre |
| `metric` | string | "rating" | Trend metric: "rating", "popularity", "count", "diversity" |
| `include_forecasts` | boolean | false | Include trend forecasting |

**Example Requests**:

```bash
# Get monthly rating trends
GET /api/trends?period=monthly&metric=rating

# Get yearly trends for Action movies with forecasts
GET /api/trends?period=yearly&genre=Action&include_forecasts=true
```

**Response Example**:
```json
{
  "success": true,
  "data": {
    "temporal_analysis": {
      "rating_trends": [
        {
          "period": "2023-01",
          "avg_rating": 3.7,
          "rating_count": 1250,
          "unique_movies": 450,
          "trend_indicator": "stable",
          "volatility": 0.12
        },
        {
          "period": "2023-02",
          "avg_rating": 3.72,
          "rating_count": 1180,
          "unique_movies": 425,
          "trend_indicator": "increasing",
          "volatility": 0.09
        }
      ],
      "genre_trends": [
        {
          "genre": "Action",
          "trend_direction": "increasing",
          "growth_rate": 0.15,
          "peak_period": "2023-06",
          "momentum": 0.78,
          "forecast": {
            "next_3_months": "continued_growth",
            "confidence": 0.82
          }
        },
        {
          "genre": "Drama",
          "trend_direction": "stable",
          "growth_rate": 0.02,
          "peak_period": "2023-03",
          "momentum": 0.45,
          "forecast": {
            "next_3_months": "stable",
            "confidence": 0.91
          }
        }
      ]
    },
    "insights": {
      "overall_trend": "slightly_positive",
      "seasonal_patterns": {
        "detected": true,
        "peak_months": ["November", "December"],
        "low_months": ["June", "August"],
        "amplitude": 0.23
      },
      "anomalies": [
        {
          "period": "2023-07",
          "type": "rating_spike",
          "magnitude": 0.31,
          "possible_cause": "blockbuster_releases"
        }
      ],
      "predictions": {
        "next_quarter_rating": 3.68,
        "confidence_interval": [3.61, 3.75],
        "trending_genres": ["Sci-Fi", "Animation"]
      }
    }
  }
}
```

**Status Codes**:
- `200`: Success - Trends data retrieved
- `400`: Bad Request - Invalid period or parameters
- `500`: Internal Server Error

---

## 🎨 Visualization Endpoints

#### 📊 GET /api/visualizations

Get comprehensive list of available visualizations and their configurations.

**URL**: `GET /api/visualizations`

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `category` | string | "all" | Filter by category: "all", "ratings", "genres", "trends", "users" |
| `include_examples` | boolean | false | Include example configurations |

**Response Example**:

```json
{
  "success": true,
  "data": {
    "available_visualizations": [
      {
        "type": "rating_distribution",
        "category": "ratings",
        "description": "Distribution of movie ratings across the platform",
        "parameters": {
          "required": ["format"],
          "optional": ["bins", "style", "color_scheme", "title"]
        },
        "supported_formats": ["png", "svg", "pdf", "json"],
        "example_config": {
          "bins": 20,
          "format": "png",
          "style": "seaborn"
        }
      },
      {
        "type": "genre_popularity",
        "category": "genres",
        "description": "Popularity and rating analysis of different genres",
        "parameters": {
          "required": ["chart_type"],
          "optional": ["top_n", "format", "time_range"]
        },
        "supported_formats": ["png", "svg", "json"],
        "chart_types": ["bar", "pie", "treemap", "bubble"]
      }
    ]
  }
}
```

---

#### 🎯 POST /api/visualizations

Generate custom visualizations with advanced configuration options.

**URL**: `POST /api/visualizations`

**Request Body Schema**:

```json
{
  "type": "string (required)",
  "parameters": {
    "format": "string (required): png|svg|pdf|json",
    "style": "string (optional): default|seaborn|ggplot|dark",
    "dimensions": {
      "width": "int (optional): 400-2000",
      "height": "int (optional): 300-1500"
    },
    "title": "string (optional)",
    "color_scheme": "string (optional): viridis|plasma|cool|warm"
  },
  "filters": {
    "genre": "string (optional)",
    "year_range": "[int, int] (optional)",
    "rating_range": "[float, float] (optional)",
    "user_id": "int (optional)"
  },
  "options": {
    "include_metadata": "boolean (optional): default true",
    "cache_duration": "int (optional): seconds, default 3600"
  }
}
```

**Example Request**:

```json
{
  "type": "rating_distribution",
  "parameters": {
    "format": "png",
    "style": "seaborn",
    "dimensions": {
      "width": 800,
      "height": 600
    },
    "title": "Action Movie Rating Distribution",
    "color_scheme": "viridis"
  },
  "filters": {
    "genre": "Action",
    "year_range": [2000, 2023],
    "rating_range": [1.0, 5.0]
  },
  "options": {
    "include_metadata": true,
    "cache_duration": 7200
  }
}
```

**Response Example**:
```json
{
  "success": true,
  "data": {
    "visualization": {
      "id": "viz_123456",
      "type": "rating_distribution",
      "url": "/api/visualizations/viz_123456/download",
      "thumbnail_url": "/api/visualizations/viz_123456/thumbnail",
      "metadata": {
        "format": "png",
        "size": "800x600",
        "created_at": "2024-01-15T10:30:00Z",
        "data_points": 1500
      }
    }
  }
}
```

#### GET /api/visualizations/{viz_id}/download
Download a generated visualization.

**Path Parameters:**
- `viz_id` (string): The visualization ID

**Response:** Binary image data with appropriate content-type header.

---

### System Management

#### POST /api/refresh
Refresh the data cache and reload datasets.

**Request Body (optional):**
```json
{
  "force_download": false,
  "clear_cache": true,
  "components": ["movies", "ratings", "analysis"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "refresh_status": {
      "started_at": "2024-01-15T10:30:00Z",
      "completed_at": "2024-01-15T10:32:15Z",
      "duration": "2m 15s",
      "components_refreshed": ["movies", "ratings", "analysis"],
      "cache_cleared": true,
      "new_data_loaded": true
    }
  }
}
```

#### DELETE /api/cache
Clear the application cache.

**Parameters:**
- `cache_type` (string, optional): Type of cache to clear ("all", "data", "visualizations", "analysis")

**Response:**
```json
{
  "success": true,
  "data": {
    "cache_cleared": {
      "types": ["data", "analysis"],
      "size_freed": "150MB",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  }
}
```

---

## Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_PARAMETER` | Invalid or missing parameter | 400 |
| `MOVIE_NOT_FOUND` | Movie ID not found | 404 |
| `USER_NOT_FOUND` | User ID not found | 404 |
| `INVALID_GENRE` | Genre not recognized | 400 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `DATA_NOT_LOADED` | Dataset not available | 503 |
| `PROCESSING_ERROR` | Internal processing error | 500 |
| `VISUALIZATION_ERROR` | Visualization generation failed | 500 |

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default Limit**: 100 requests per minute per IP address
- **Burst Limit**: 20 requests per 10 seconds
- **Headers**: Rate limit information is included in response headers:
  - `X-RateLimit-Limit`: Request limit per window
  - `X-RateLimit-Remaining`: Remaining requests in current window
  - `X-RateLimit-Reset`: Time when the rate limit resets

## CORS Support

The API supports Cross-Origin Resource Sharing (CORS) with the following configuration:

- **Allowed Origins**: `*` (development), specific domains (production)
- **Allowed Methods**: `GET`, `POST`, `PUT`, `DELETE`, `OPTIONS`
- **Allowed Headers**: `Content-Type`, `Authorization`, `X-Requested-With`

## Pagination

List endpoints support pagination with the following parameters:

- `limit`: Number of items per page (default: 50, max: 1000)
- `offset`: Number of items to skip
- `page`: Page number (alternative to offset)

Pagination information is included in the response:

```json
{
  "pagination": {
    "total": 1500,
    "page": 1,
    "per_page": 50,
    "total_pages": 30,
    "has_next": true,
    "has_prev": false,
    "next_url": "/api/movies?page=2",
    "prev_url": null
  }
}
```

## Filtering and Sorting

Many endpoints support filtering and sorting:

### Common Filters
- `genre`: Filter by movie genre
- `year`: Filter by release year
- `min_rating`, `max_rating`: Rating range filters
- `min_votes`: Minimum number of ratings

### Sorting Options
- `sort_by`: Field to sort by
- `sort_order`: `asc` or `desc`

## Examples

### Python Client Example

```python
import requests

class MovieLensAPI:
    def __init__(self, base_url="http://localhost:5000/api"):
        self.base_url = base_url
    
    def get_top_movies(self, n=10, genre=None):
        params = {"n": n}
        if genre:
            params["genre"] = genre
        
        response = requests.get(f"{self.base_url}/movies/top", params=params)
        return response.json()
    
    def get_recommendations(self, user_id, n=10):
        response = requests.get(f"{self.base_url}/users/{user_id}/recommendations", 
                              params={"n": n})
        return response.json()

# Usage
api = MovieLensAPI()
top_movies = api.get_top_movies(n=5, genre="Action")
recommendations = api.get_recommendations(user_id=123, n=10)
```

### JavaScript Client Example

```javascript
class MovieLensAPI {
    constructor(baseUrl = 'http://localhost:5000/api') {
        this.baseUrl = baseUrl;
    }
    
    async getTopMovies(n = 10, genre = null) {
        const params = new URLSearchParams({ n });
        if (genre) params.append('genre', genre);
        
        const response = await fetch(`${this.baseUrl}/movies/top?${params}`);
        return response.json();
    }
    
    async getMovieDetails(movieId) {
        const response = await fetch(`${this.baseUrl}/movies/${movieId}`);
        return response.json();
    }
}

// Usage
const api = new MovieLensAPI();
const topMovies = await api.getTopMovies(5, 'Drama');
const movieDetails = await api.getMovieDetails(1);
```

## Testing the API

### Using curl

```bash
# Get system status
curl -X GET "http://localhost:5000/api/status"

# Get top movies
curl -X GET "http://localhost:5000/api/movies/top?n=5&min_ratings=100"

# Get user recommendations
curl -X GET "http://localhost:5000/api/users/123/recommendations?n=10"

# Generate visualization
curl -X POST "http://localhost:5000/api/visualizations" \
     -H "Content-Type: application/json" \
     -d '{"type": "rating_distribution", "parameters": {"bins": 20}}'
```

### Using Postman

Import the API collection using the provided Postman collection file or create requests manually using the endpoint documentation above.

## Support and Feedback

For API support, bug reports, or feature requests:

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides available in `/docs`
- **Community**: Join discussions on GitHub Discussions

---

*This API documentation is automatically generated and kept up-to-date with the latest platform features.*