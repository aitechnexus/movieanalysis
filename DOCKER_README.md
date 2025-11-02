# Docker Deployment Guide

This guide explains how to run the Movie Data Analysis Platform using Docker and Docker Compose.

## Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)

## Quick Start

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd movielens-analysis
   ```

2. **Build and start the services:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Frontend: http://localhost:8000
   - Backend API: http://localhost:8001

## Services

### Backend Service
- **Port:** 8001
- **Technology:** Flask API with Python
- **Health Check:** Monitors `/api/status` endpoint
- **Features:** 
  - Data processing and analysis
  - Statistical computations
  - Visualization generation
  - RESTful API endpoints

### Frontend Service
- **Port:** 8000
- **Technology:** Static HTML/CSS/JS served by Nginx
- **Health Check:** Monitors nginx availability
- **Dependencies:** Waits for backend service to be healthy
- **Features:**
  - Interactive dashboard
  - Data visualization
  - Dataset management
  - Statistical analysis interface

## Service Dependencies

The frontend service uses `depends_on` with health check conditions to ensure:
1. Backend service starts first
2. Backend health check passes before frontend starts
3. Nginx can properly resolve the backend upstream
4. No "host not found" errors during startup

## Docker Commands

### Build and Run
```bash
# Build and start all services
docker-compose up --build

# Run in detached mode (background)
docker-compose up -d --build

# Build specific service
docker-compose build backend
docker-compose build frontend
```

### Management
```bash
# Stop all services
docker-compose down

# View logs
docker-compose logs
docker-compose logs backend
docker-compose logs frontend

# Restart services
docker-compose restart

# View running containers
docker-compose ps
```

### Development
```bash
# Rebuild and restart after code changes
docker-compose down
docker-compose up --build

# Access container shell
docker-compose exec backend bash
docker-compose exec frontend sh
```

## Data Persistence

- **Data Directory:** `./data` (mounted to `/app/data` in backend container)
- **Outputs Directory:** `./outputs` (mounted to `/app/outputs` in backend container)

Your data and generated outputs will persist between container restarts.

## Environment Configuration

The application automatically detects whether it's running in Docker and adjusts API URLs accordingly:

- **Local Development:** API calls go to `http://localhost:8001`
- **Docker Environment:** API calls use relative URLs proxied through Nginx

## Health Checks

Both services include health checks:

- **Backend:** Checks `/api/status` endpoint every 30 seconds
- **Frontend:** Checks HTTP response every 30 seconds

## Troubleshooting

### Port Conflicts
If ports 8000 or 8001 are already in use, modify the `docker-compose.yml` file:

```yaml
services:
  backend:
    ports:
      - "8002:8001"  # Change external port
  frontend:
    ports:
      - "8080:80"    # Change external port
```

### Container Logs
View detailed logs to diagnose issues:

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Reset Everything
To completely reset the Docker environment:

```bash
docker-compose down -v
docker system prune -f
docker-compose up --build
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "host not found in upstream 'backend'" Error
**Problem:** Nginx cannot resolve the backend service hostname.
**Solution:** This has been fixed with:
- Health check dependencies in docker-compose.yml
- Upstream block in nginx.conf
- Proper service startup order

#### 2. Services Not Starting
**Problem:** Services fail to start or health checks fail.
**Solution:**
```bash
# Check service logs
docker-compose logs backend
docker-compose logs frontend

# Restart services
docker-compose restart
```

#### 3. Port Conflicts
**Problem:** Ports 8000 or 8001 are already in use.
**Solution:**
```bash
# Check what's using the ports
lsof -i :8000
lsof -i :8001

# Stop conflicting services or change ports in docker-compose.yml
```

#### 4. Build Failures
**Problem:** Docker build fails due to dependency issues.
**Solution:**
```bash
# Clean build with no cache
docker-compose build --no-cache

# Update base images
docker-compose pull
```

## Production Considerations

For production deployment:

1. **Environment Variables:** Set `FLASK_ENV=production` and `FLASK_DEBUG=0`
2. **Security:** Use proper secrets management
3. **Scaling:** Consider using Docker Swarm or Kubernetes
4. **Monitoring:** Add logging and monitoring solutions
5. **SSL/TLS:** Configure HTTPS with proper certificates

## API Endpoints

Once running, the following API endpoints are available:

- `GET /api/status` - API health check
- `GET /api/analysis` - Get analysis data
- `GET /api/statistical-summary` - Statistical summary
- `GET /api/comprehensive-statistics` - Comprehensive stats
- `GET /api/advanced-heatmaps` - Advanced heatmap data
- `GET /api/percentage-analysis` - Percentage analysis
- `POST /api/upload` - Upload dataset
- `POST /api/generate-report` - Generate analysis report

## Support

For issues related to Docker deployment, check:

1. Docker and Docker Compose versions
2. Available system resources (RAM, disk space)
3. Port availability
4. Container logs for error messages