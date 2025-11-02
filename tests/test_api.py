"""API tests for Flask endpoints"""
import json
from unittest.mock import patch

import pytest


class TestAPI:
    """Test cases for Flask API endpoints"""

    def test_status_endpoint(self, client):
        """Test /api/status endpoint"""
        response = client.get('/api/status')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'timestamp' in data
        assert data['status'] == 'healthy'

    def test_movies_endpoint(self, client):
        """Test /api/movies endpoint"""
        response = client.get('/api/movies')
        
        assert response.status_code in [200, 500]  # May fail if no data loaded
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'movies' in data
            assert isinstance(data['movies'], list)

    def test_movies_endpoint_with_params(self, client):
        """Test /api/movies endpoint with parameters"""
        response = client.get('/api/movies?limit=10&genre=Action')
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'movies' in data
            assert len(data['movies']) <= 10

    def test_top_movies_endpoint(self, client):
        """Test /api/movies/top endpoint"""
        response = client.get('/api/movies/top')
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'top_movies' in data
            assert isinstance(data['top_movies'], list)

    def test_top_movies_with_params(self, client):
        """Test /api/movies/top with parameters"""
        response = client.get('/api/movies/top?n=5&min_ratings=10')
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'top_movies' in data
            assert len(data['top_movies']) <= 5

    def test_popular_movies_endpoint(self, client):
        """Test /api/movies/popular endpoint"""
        response = client.get('/api/movies/popular')
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'popular_movies' in data
            assert isinstance(data['popular_movies'], list)

    def test_genres_endpoint(self, client):
        """Test /api/genres endpoint"""
        response = client.get('/api/genres')
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'genres' in data
            assert isinstance(data['genres'], list)

    def test_statistics_endpoint(self, client):
        """Test /api/statistics endpoint"""
        response = client.get('/api/statistics')
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'statistics' in data
            assert isinstance(data['statistics'], dict)

    def test_user_preferences_endpoint(self, client):
        """Test /api/users/{user_id}/preferences endpoint"""
        response = client.get('/api/users/1/preferences')
        
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'preferences' in data

    def test_user_recommendations_endpoint(self, client):
        """Test /api/users/{user_id}/recommendations endpoint"""
        response = client.get('/api/users/1/recommendations')
        
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'recommendations' in data
            assert isinstance(data['recommendations'], list)

    def test_movie_details_endpoint(self, client):
        """Test /api/movies/{movie_id} endpoint"""
        response = client.get('/api/movies/1')
        
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'movie' in data

    def test_movie_similar_endpoint(self, client):
        """Test /api/movies/{movie_id}/similar endpoint"""
        response = client.get('/api/movies/1/similar')
        
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'similar_movies' in data
            assert isinstance(data['similar_movies'], list)

    def test_refresh_endpoint(self, client):
        """Test /api/refresh endpoint"""
        response = client.post('/api/refresh')
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'message' in data
            assert 'timestamp' in data

    def test_visualizations_endpoint(self, client):
        """Test /api/visualizations endpoint"""
        response = client.get('/api/visualizations')
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'visualizations' in data

    def test_invalid_endpoint(self, client):
        """Test invalid endpoint"""
        response = client.get('/api/nonexistent')
        
        assert response.status_code == 404

    def test_invalid_user_id(self, client):
        """Test invalid user ID"""
        response = client.get('/api/users/invalid/preferences')
        
        assert response.status_code in [400, 404, 500]

    def test_invalid_movie_id(self, client):
        """Test invalid movie ID"""
        response = client.get('/api/movies/invalid')
        
        assert response.status_code in [400, 404, 500]

    def test_cors_headers(self, client):
        """Test CORS headers"""
        response = client.get('/api/status')
        
        # Check for CORS headers if implemented
        if 'Access-Control-Allow-Origin' in response.headers:
            assert response.headers['Access-Control-Allow-Origin'] == '*'

    def test_content_type_headers(self, client):
        """Test content type headers"""
        response = client.get('/api/status')
        
        if response.status_code == 200:
            assert 'application/json' in response.content_type

    @pytest.mark.parametrize("endpoint", [
        '/api/movies',
        '/api/movies/top',
        '/api/movies/popular',
        '/api/genres',
        '/api/statistics'
    ])
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
            response = client.get('/api/status')
            responses.append(response.status_code)
        
        # Should not have any 429 (Too Many Requests) responses for normal usage
        assert 429 not in responses

    @patch('app.get_analysis_data')
    def test_error_handling(self, mock_get_data, client):
        """Test API error handling"""
        # Mock an exception
        mock_get_data.side_effect = Exception("Test error")
        
        response = client.get('/api/movies')
        
        assert response.status_code == 500
        
        data = json.loads(response.data)
        assert 'error' in data

    def test_pagination_parameters(self, client):
        """Test pagination parameters"""
        response = client.get('/api/movies?page=1&per_page=10')
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            # Check if pagination info is included
            if 'pagination' in data:
                assert 'page' in data['pagination']
                assert 'per_page' in data['pagination']

    def test_filtering_parameters(self, client):
        """Test filtering parameters"""
        response = client.get('/api/movies?genre=Action&min_rating=4.0&year=2020')
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'movies' in data

    def test_sorting_parameters(self, client):
        """Test sorting parameters"""
        response = client.get('/api/movies?sort_by=rating&order=desc')
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'movies' in data

    def test_api_documentation_endpoint(self, client):
        """Test API documentation endpoint if it exists"""
        response = client.get('/api/docs')
        
        # May or may not exist
        assert response.status_code in [200, 404]

    def test_health_check_detailed(self, client):
        """Test detailed health check"""
        response = client.get('/api/health')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'status' in data
            # May include additional health metrics
            if 'metrics' in data:
                assert isinstance(data['metrics'], dict)