#!/usr/bin/env python3
"""
Gunicorn configuration for MovieLens Analysis API
Production-ready WSGI server configuration
"""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8001"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "movielens-api"

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (disabled for development, enable for production with certificates)
# keyfile = None
# certfile = None

# Application
wsgi_app = "app:app"

# Preload application for better performance
preload_app = True

# Enable threading for better concurrency
threads = 2

# Graceful timeout for worker shutdown
graceful_timeout = 30

# Environment variables
raw_env = [
    "PYTHONPATH=/app",
    "FLASK_ENV=production"
]

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("MovieLens Analysis API server is ready. PID: %s", os.getpid())

def worker_int(worker):
    """Called just after a worker has been killed by a signal."""
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    worker.log.info("Worker initialized (pid: %s)", worker.pid)

def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    worker.log.info("Worker received SIGABRT signal")