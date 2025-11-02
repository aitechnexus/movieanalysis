# Contributing to MovieLens Analysis Platform

Thank you for your interest in contributing to the MovieLens Analysis Platform! This document provides guidelines and information for contributors to help maintain code quality and ensure smooth collaboration.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Workflow](#development-workflow)
- [Performance Guidelines](#performance-guidelines)
- [Security Considerations](#security-considerations)

---

## Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, or identity.

### Expected Behavior
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior
- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing private information without permission
- Any conduct that would be inappropriate in a professional setting

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Docker and Docker Compose
- Git
- Basic understanding of data analysis and web development

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/your-org/movielens-analysis.git
cd movielens-analysis

# Set up development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run tests to verify setup
python -m pytest tests/ -v

# Start development environment
docker-compose up --build
```

---

## Development Setup

### Environment Configuration

1. **Create Environment File**
```bash
cp .env.example .env
# Edit .env with your configuration
```

2. **Install Development Dependencies**
```bash
pip install -r requirements-dev.txt
```

3. **Set Up Pre-commit Hooks**
```bash
pre-commit install
```

### IDE Configuration

#### VS Code Settings
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

#### PyCharm Settings
- Set Python interpreter to project virtual environment
- Enable pytest as test runner
- Configure Black as code formatter
- Enable pylint for code analysis

---

## Contributing Guidelines

### Types of Contributions

#### 🐛 Bug Fixes
- Fix existing functionality issues
- Improve error handling
- Resolve performance problems

#### ✨ New Features
- Add new analysis capabilities
- Implement additional visualizations
- Extend API functionality

#### 📚 Documentation
- Improve existing documentation
- Add code examples
- Create tutorials and guides

#### 🧪 Testing
- Add missing test coverage
- Improve test quality
- Add integration tests

#### 🔧 Maintenance
- Code refactoring
- Dependency updates
- Performance optimizations

### Contribution Process

1. **Check Existing Issues**: Look for related issues or discussions
2. **Create Issue**: If none exists, create a detailed issue description
3. **Fork Repository**: Create your own fork for development
4. **Create Branch**: Use descriptive branch names
5. **Develop**: Follow coding standards and write tests
6. **Test**: Ensure all tests pass and coverage requirements are met
7. **Document**: Update relevant documentation
8. **Submit PR**: Create a detailed pull request

---

## Coding Standards

### Python Style Guide

#### Code Formatting
- **Formatter**: Black (line length: 88 characters)
- **Import Sorting**: isort
- **Linting**: pylint, flake8

```python
# Good example
def analyze_movie_ratings(
    movies_df: pd.DataFrame, 
    ratings_df: pd.DataFrame,
    min_ratings: int = 50
) -> Dict[str, Any]:
    """
    Analyze movie ratings and return statistics.
    
    Args:
        movies_df: DataFrame containing movie information
        ratings_df: DataFrame containing user ratings
        min_ratings: Minimum number of ratings required
        
    Returns:
        Dictionary containing analysis results
        
    Raises:
        ValueError: If dataframes are empty or invalid
    """
    if movies_df.empty or ratings_df.empty:
        raise ValueError("Input dataframes cannot be empty")
    
    # Analysis logic here
    return {"status": "success", "results": analysis_results}
```

#### Naming Conventions
- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private Methods**: `_leading_underscore`

```python
# Good examples
class MovieAnalyzer:
    MAX_CACHE_SIZE = 1000
    
    def __init__(self):
        self._cache = {}
    
    def get_top_movies(self, n: int = 10) -> List[Dict]:
        return self._fetch_from_cache_or_compute(n)
    
    def _fetch_from_cache_or_compute(self, n: int) -> List[Dict]:
        # Private method implementation
        pass
```

#### Type Hints
Always use type hints for function parameters and return values:

```python
from typing import List, Dict, Optional, Union
import pandas as pd

def process_ratings(
    ratings: pd.DataFrame,
    user_id: Optional[int] = None
) -> Dict[str, Union[int, float, List[str]]]:
    """Process user ratings with optional filtering."""
    pass
```

### Documentation Standards

#### Docstring Format (Google Style)
```python
def calculate_similarity_matrix(
    ratings_matrix: pd.DataFrame,
    method: str = "cosine"
) -> np.ndarray:
    """
    Calculate similarity matrix between items.
    
    This function computes pairwise similarities between items based on
    user rating patterns using the specified similarity metric.
    
    Args:
        ratings_matrix: User-item rating matrix with users as rows
        method: Similarity calculation method ('cosine', 'pearson', 'jaccard')
        
    Returns:
        Square similarity matrix with shape (n_items, n_items)
        
    Raises:
        ValueError: If method is not supported
        RuntimeError: If matrix computation fails
        
    Example:
        >>> ratings = pd.DataFrame([[4, 0, 5], [0, 3, 4]])
        >>> similarity = calculate_similarity_matrix(ratings, "cosine")
        >>> similarity.shape
        (3, 3)
    """
```

#### Code Comments
```python
# Good: Explain why, not what
def normalize_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    # Subtract user mean to account for rating bias differences
    user_means = ratings.groupby('userId')['rating'].mean()
    normalized = ratings.copy()
    normalized['rating'] = (
        ratings['rating'] - ratings['userId'].map(user_means)
    )
    return normalized

# Bad: Obvious comment
def normalize_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    # Calculate mean ratings for each user
    user_means = ratings.groupby('userId')['rating'].mean()
```

---

## Testing Requirements

### Test Coverage Standards
- **Minimum Coverage**: 80% overall
- **Critical Modules**: 90% coverage required
- **New Features**: 100% coverage for new code

### Test Categories

#### Unit Tests
```python
# tests/test_analyzer.py
import pytest
from src.analyzer import MovieAnalyzer

class TestMovieAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return MovieAnalyzer()
    
    @pytest.fixture
    def sample_data(self):
        return {
            'movies': pd.DataFrame({
                'movieId': [1, 2, 3],
                'title': ['Movie A', 'Movie B', 'Movie C'],
                'genres': ['Action', 'Comedy', 'Drama']
            }),
            'ratings': pd.DataFrame({
                'userId': [1, 1, 2, 2],
                'movieId': [1, 2, 1, 3],
                'rating': [4.0, 3.5, 5.0, 2.0]
            })
        }
    
    def test_get_top_movies_returns_correct_format(self, analyzer, sample_data):
        """Test that get_top_movies returns properly formatted results."""
        result = analyzer.get_top_movies(sample_data['movies'], sample_data['ratings'])
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all('movieId' in movie for movie in result)
        assert all('avg_rating' in movie for movie in result)
    
    def test_get_top_movies_with_invalid_input(self, analyzer):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="Input dataframes cannot be empty"):
            analyzer.get_top_movies(pd.DataFrame(), pd.DataFrame())
```

#### Integration Tests
```python
# tests/test_integration.py
import pytest
import requests
from src.app import create_app

class TestAPIIntegration:
    @pytest.fixture
    def client(self):
        app = create_app(testing=True)
        return app.test_client()
    
    def test_full_analysis_workflow(self, client):
        """Test complete analysis workflow through API."""
        # Test data loading
        response = client.post('/api/refresh')
        assert response.status_code == 200
        
        # Test analysis
        response = client.get('/api/movies/top?n=5')
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['data']) == 5
```

#### Performance Tests
```python
# tests/test_performance.py
import time
import pytest
from src.analyzer import MovieAnalyzer

class TestPerformance:
    def test_large_dataset_performance(self, large_dataset):
        """Ensure analysis completes within acceptable time."""
        analyzer = MovieAnalyzer()
        
        start_time = time.time()
        result = analyzer.get_top_movies(large_dataset['movies'], large_dataset['ratings'])
        execution_time = time.time() - start_time
        
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert len(result) > 0
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/test_unit/ -v
python -m pytest tests/test_integration/ -v
python -m pytest tests/test_performance/ -v

# Run tests in parallel
python -m pytest tests/ -n auto
```

---

## Documentation Standards

### API Documentation
- Use OpenAPI/Swagger specifications
- Include request/response examples
- Document all parameters and error codes
- Provide usage examples in multiple languages

### Code Documentation
- Comprehensive docstrings for all public methods
- Inline comments for complex logic
- Type hints for all function signatures
- Examples in docstrings where helpful

### User Documentation
- Clear setup and installation instructions
- Tutorial-style guides for common use cases
- Troubleshooting sections
- Performance optimization tips

---

## Pull Request Process

### Before Submitting

1. **Code Quality Checklist**
   - [ ] Code follows style guidelines
   - [ ] All tests pass
   - [ ] Coverage requirements met
   - [ ] Documentation updated
   - [ ] No linting errors

2. **Testing Checklist**
   - [ ] Unit tests for new functionality
   - [ ] Integration tests if applicable
   - [ ] Performance tests for critical paths
   - [ ] Manual testing completed

### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added and passing
- [ ] No breaking changes (or clearly documented)

## Screenshots (if applicable)
Add screenshots for UI changes.

## Additional Notes
Any additional information for reviewers.
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and quality checks
2. **Peer Review**: At least one team member reviews the code
3. **Maintainer Review**: Core maintainer provides final approval
4. **Merge**: Squash and merge after all approvals

---

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
A clear and concise description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. macOS, Ubuntu 20.04]
- Python Version: [e.g. 3.9.7]
- Docker Version: [e.g. 20.10.8]
- Browser: [e.g. Chrome 95.0]

**Additional Context**
Any other context about the problem.
```

### Feature Requests

```markdown
**Feature Description**
A clear and concise description of the feature.

**Problem Statement**
What problem does this feature solve?

**Proposed Solution**
Describe your proposed solution.

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Any other context, mockups, or examples.
```

---

## Development Workflow

### Branch Naming Convention
- `feature/description-of-feature`
- `bugfix/description-of-bug`
- `hotfix/critical-issue`
- `docs/documentation-update`
- `refactor/code-improvement`

### Commit Message Format
```
type(scope): brief description

Detailed explanation of changes if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(api): add user recommendation endpoint

Add new endpoint for personalized movie recommendations
based on user rating history and collaborative filtering.

Fixes #45
```

### Release Process

1. **Version Bumping**: Follow semantic versioning (MAJOR.MINOR.PATCH)
2. **Changelog**: Update CHANGELOG.md with new features and fixes
3. **Testing**: Run full test suite including performance tests
4. **Documentation**: Update version-specific documentation
5. **Tagging**: Create git tag with version number
6. **Deployment**: Deploy to staging, then production

---

## Performance Guidelines

### Code Performance

#### Database Queries
```python
# Good: Efficient query with proper indexing
def get_user_ratings_optimized(user_id: int) -> pd.DataFrame:
    query = """
    SELECT movieId, rating, timestamp 
    FROM ratings 
    WHERE userId = %s 
    ORDER BY timestamp DESC
    """
    return pd.read_sql(query, connection, params=[user_id])

# Bad: Loading entire dataset
def get_user_ratings_inefficient(user_id: int) -> pd.DataFrame:
    all_ratings = pd.read_sql("SELECT * FROM ratings", connection)
    return all_ratings[all_ratings['userId'] == user_id]
```

#### Memory Management
```python
# Good: Process data in chunks
def process_large_dataset(file_path: str) -> Dict[str, Any]:
    results = []
    chunk_size = 10000
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
        
    return combine_results(results)

# Bad: Load entire dataset into memory
def process_large_dataset_bad(file_path: str) -> Dict[str, Any]:
    data = pd.read_csv(file_path)  # Could cause memory issues
    return process_data(data)
```

#### Caching Strategy
```python
from functools import lru_cache
import redis

class AnalysisCache:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    @lru_cache(maxsize=128)
    def get_movie_stats(self, movie_id: int) -> Dict[str, Any]:
        """Cache frequently accessed movie statistics."""
        cache_key = f"movie_stats:{movie_id}"
        cached_result = self.redis_client.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        
        stats = self._calculate_movie_stats(movie_id)
        self.redis_client.setex(cache_key, 3600, json.dumps(stats))
        return stats
```

### API Performance

#### Response Time Targets
- **Simple queries**: < 200ms
- **Complex analysis**: < 2s
- **Data loading**: < 5s
- **Report generation**: < 10s

#### Optimization Techniques
1. **Pagination**: Limit response sizes
2. **Caching**: Cache expensive computations
3. **Async Processing**: Use background tasks for heavy operations
4. **Database Indexing**: Optimize query performance
5. **Connection Pooling**: Reuse database connections

---

## Security Considerations

### Input Validation
```python
from marshmallow import Schema, fields, validate

class MovieQuerySchema(Schema):
    genre = fields.Str(validate=validate.OneOf(['Action', 'Comedy', 'Drama', ...]))
    year = fields.Int(validate=validate.Range(min=1900, max=2030))
    limit = fields.Int(validate=validate.Range(min=1, max=1000))
    
def get_movies_by_genre(request_data: dict) -> List[Dict]:
    schema = MovieQuerySchema()
    try:
        validated_data = schema.load(request_data)
    except ValidationError as err:
        raise ValueError(f"Invalid input: {err.messages}")
    
    return fetch_movies(**validated_data)
```

### SQL Injection Prevention
```python
# Good: Parameterized queries
def get_user_ratings(user_id: int, movie_genre: str) -> pd.DataFrame:
    query = """
    SELECT r.rating, m.title 
    FROM ratings r 
    JOIN movies m ON r.movieId = m.movieId 
    WHERE r.userId = %s AND m.genres LIKE %s
    """
    return pd.read_sql(query, connection, params=[user_id, f"%{movie_genre}%"])

# Bad: String concatenation (vulnerable to SQL injection)
def get_user_ratings_bad(user_id: int, movie_genre: str) -> pd.DataFrame:
    query = f"""
    SELECT r.rating, m.title 
    FROM ratings r 
    JOIN movies m ON r.movieId = m.movieId 
    WHERE r.userId = {user_id} AND m.genres LIKE '%{movie_genre}%'
    """
    return pd.read_sql(query, connection)
```

### Data Privacy
- Anonymize user data in logs
- Implement data retention policies
- Secure sensitive configuration
- Use environment variables for secrets

---

## Getting Help

### Resources
- **Documentation**: Check `/docs` directory
- **API Reference**: `/docs/API.md`
- **Testing Guide**: `/TESTING.md`
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas

### Contact
- **Maintainers**: @maintainer1, @maintainer2
- **Community**: Join our Slack/Discord channel
- **Email**: project-team@example.com

---

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

**Thank you for contributing to the MovieLens Analysis Platform!** 🎬📊

Your contributions help make this project better for everyone. We appreciate your time and effort in improving the platform.