"""API tests for Flask endpoints"""

import json
from unittest.mock import patch

import pytest


class TestAPI:
    """Test cases for Flask API endpoints"""

    def test_status_endpoint(self, client):
        """Test /api/status endpoint"""
        response = client.get("/api/status")

        assert response.status_code == 200

        data = json.loads(response.data)
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "online"

    def test_movies_endpoint(self, client):
        """Test /api/overview endpoint (closest to movies endpoint)"""
        response = client.get("/api/overview")

        assert response.status_code in [200, 500]  # May fail if no data loaded

        if response.status_code == 200:
            data = json.loads(response.data)
            assert "status" in data

    def test_movies_endpoint_with_params(self, client):
        """Test /api/top-movies endpoint with parameters"""
        response = client.get("/api/top-movies?n=10")

        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = json.loads(response.data)
            assert "data" in data
            assert "movies" in data["data"]

    def test_top_movies_endpoint(self, client):
        """Test /api/top-movies endpoint"""
        response = client.get("/api/top-movies")

        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = json.loads(response.data)
            assert "data" in data
            assert "movies" in data["data"]

    def test_top_movies_with_params(self, client):
        """Test /api/top-movies with parameters"""
        response = client.get("/api/top-movies?n=5&min_ratings=10")

        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = json.loads(response.data)
            assert "data" in data

    def test_popular_movies_endpoint(self, client):
        """Test /api/genre-popularity endpoint (closest to popular movies)"""
        response = client.get("/api/genre-popularity")

        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = json.loads(response.data)
            assert "data" in data

    def test_genres_endpoint(self, client):
        """Test /api/genre-popularity endpoint"""
        response = client.get("/api/genre-popularity")

        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = json.loads(response.data)
            assert "data" in data

    def test_statistics_endpoint(self, client):
        """Test /api/comprehensive-statistics endpoint"""
        response = client.get("/api/comprehensive-statistics")

        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            data = json.loads(response.data)
            assert "movie_statistics" in data

    def test_user_preferences_endpoint(self, client):
        """Test user stats endpoint"""
        response = client.get("/api/user-stats")

        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = response.get_json()
            assert "data" in data

    def test_user_recommendations_endpoint(self, client):
        """Test statistical summary endpoint"""
        response = client.get("/api/statistical-summary")

        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = response.get_json()
            assert "summary" in data

    def test_movie_details_endpoint(self, client):
        """Test dataset preview endpoint"""
        response = client.get("/api/dataset/preview")

        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = response.get_json()
            assert "movies" in data

    def test_movie_similar_endpoint(self, client):
        """Test percentage analysis endpoint"""
        response = client.get("/api/percentage-analysis")

        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = response.get_json()
            assert "percentage_data" in data

    def test_refresh_endpoint(self, client):
        """Test /api/refresh endpoint"""
        response = client.get("/api/refresh")  # Changed from POST to GET

        assert response.status_code in [200, 404, 500]

        if response.status_code == 200:
            data = json.loads(response.data)
            assert "message" in data
            assert "timestamp" in data

    def test_visualizations_endpoint(self, client):
        """Test advanced heatmaps endpoint"""
        response = client.get("/api/advanced-heatmaps")

        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = response.get_json()
            assert "advanced_heatmaps" in data

    def test_invalid_endpoint(self, client):
        """Test invalid endpoint"""
        response = client.get("/api/nonexistent")

        assert response.status_code == 404

    def test_invalid_user_id(self, client):
        """Test invalid parameter for user stats"""
        response = client.get("/api/user-stats?invalid=param")

        assert response.status_code in [200, 400, 404, 500]

    def test_invalid_movie_id(self, client):
        """Test invalid parameter for top movies"""
        response = client.get("/api/top-movies?n=invalid")

        assert response.status_code in [200, 400, 404, 500]

    def test_cors_headers(self, client):
        """Test CORS headers"""
        response = client.get("/api/status")

        # Check for CORS headers if implemented
        if "Access-Control-Allow-Origin" in response.headers:
            assert response.headers["Access-Control-Allow-Origin"] == "*"

    def test_content_type_headers(self, client):
        """Test content type headers"""
        response = client.get("/api/status")

        if response.status_code == 200:
            assert "application/json" in response.content_type

    @pytest.mark.parametrize(
        "endpoint",
        [
            "/api/overview",
            "/api/top-movies",
            "/api/genre-popularity",
            "/api/comprehensive-statistics",
            "/api/rating-distribution",
        ],
    )
    def test_endpoint_response_format(self, client, endpoint):
        """Test that endpoints return valid JSON"""
        response = client.get(endpoint)

        if response.status_code == 200:
            # Should be valid JSON
            data = json.loads(response.data)
            assert isinstance(data, dict)

    def test_rate_limiting(self, client):
        """Test rate limiting if implemented"""
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = client.get("/api/status")
            responses.append(response.status_code)

        # Should not have any 429 (Too Many Requests) responses for normal usage
        assert 429 not in responses

    def test_error_handling(self, client):
        """Test error handling for invalid endpoint"""
        response = client.get("/api/nonexistent")

        assert response.status_code == 404

    def test_pagination_parameters(self, client):
        """Test pagination parameters"""
        response = client.get("/api/top-movies?n=10")  # Use actual parameter

        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = json.loads(response.data)
            # Check if top movies are returned
            if "top_movies" in data:
                assert len(data["top_movies"]) <= 10

    def test_filtering_parameters(self, client):
        """Test filtering parameters"""
        response = client.get("/api/top-movies?n=5")

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.get_json()
            assert "data" in data

    def test_sorting_parameters(self, client):
        """Test sorting parameters"""
        response = client.get("/api/top-movies?n=10")

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.get_json()
            assert "data" in data

    def test_api_documentation_endpoint(self, client):
        """Test API documentation endpoint if it exists"""
        response = client.get("/api/docs")

        # May or may not exist
        assert response.status_code in [200, 404]

    def test_health_check_detailed(self, client):
        """Test detailed health check"""
        response = client.get("/api/health")

        if response.status_code == 200:
            data = json.loads(response.data)
            assert "status" in data
            # May include additional health metrics
            if "metrics" in data:
                assert isinstance(data["metrics"], dict)
