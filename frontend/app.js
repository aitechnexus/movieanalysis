// MovieLens Analysis Dashboard - JavaScript Application
class MovieLensDashboard {
    constructor() {
        // Use relative URL for API calls to work with nginx proxy
        this.apiBaseUrl = '/api';
        this.charts = {};
        this.data = {};
        this.currentGenreView = 'bar';
        this.currentTimeSeriesMetric = 'count';
        
        this.init();
    }

    async init() {
        this.showLoading(true);
        await this.loadAllData();
        this.initializeCharts();
        this.setupEventListeners();
        this.showLoading(false);
        this.updateApiStatus(true);
    }

    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        overlay.classList.toggle('hidden', !show);
    }

    updateApiStatus(connected) {
        const indicator = document.getElementById('apiStatus');
        const text = document.getElementById('apiStatusText');
        
        indicator.classList.toggle('connected', connected);
        text.textContent = connected ? 'Connected' : 'Disconnected';
    }

    async loadAllData() {
        try {
            // Load all data in parallel
            const [overview, ratingDist, topMovies, genreData, timeSeries, userStats] = await Promise.all([
                this.fetchData('/overview'),
                this.fetchData('/rating-distribution'),
                this.fetchData('/top-movies'),
                this.fetchData('/genre-popularity'),
                this.fetchData('/time-series'),
                this.fetchData('/user-stats')
            ]);

            this.data = {
                overview,
                ratingDist,
                topMovies,
                genreData,
                timeSeries,
                userStats
            };

            this.updateOverviewCards();
            this.updateLastUpdated();

        } catch (error) {
            console.error('Failed to load data:', error);
            this.updateApiStatus(false);
            this.showErrorMessage('Failed to load analysis data. Please check the API connection.');
        }
    }

    async fetchData(endpoint) {
        const response = await fetch(`${this.apiBaseUrl}${endpoint}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        return result.data;
    }

    updateOverviewCards() {
        const overview = this.data.overview;
        if (!overview) return;

        document.getElementById('totalMovies').textContent = overview.n_movies?.toLocaleString() || '-';
        document.getElementById('totalRatings').textContent = overview.n_ratings?.toLocaleString() || '-';
        document.getElementById('totalUsers').textContent = overview.n_users?.toLocaleString() || '-';
        
        if (overview.date_range) {
            document.getElementById('dateRange').textContent = 
                `${overview.date_range[0]} - ${overview.date_range[1]}`;
        }
    }

    updateLastUpdated() {
        const overview = this.data.overview;
        if (overview?.analysis_date) {
            document.getElementById('lastUpdated').textContent = overview.analysis_date;
        }
    }

    initializeCharts() {
        // Set Chart.js defaults for dark theme
        Chart.defaults.color = '#a0aec0';
        Chart.defaults.borderColor = '#3a4553';
        Chart.defaults.backgroundColor = 'rgba(102, 126, 234, 0.1)';
        
        this.createRatingDistributionChart();
        this.createTopMoviesChart();
        this.createGenreChart();
        this.createTimeSeriesChart();
        this.createUserActivityChart();
    }

    createRatingDistributionChart() {
        const ctx = document.getElementById('ratingChart').getContext('2d');
        const data = this.data.ratingDist;
        
        if (!data?.chart) return;

        this.charts.rating = new Chart(ctx, {
            type: 'bar',
            data: data.chart,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: '#252a3a',
                        titleColor: '#ffffff',
                        bodyColor: '#a0aec0',
                        borderColor: '#3a4553',
                        borderWidth: 1
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: '#3a4553'
                        },
                        ticks: {
                            callback: function(value) {
                                return value.toLocaleString();
                            }
                        }
                    },
                    x: {
                        grid: {
                            color: '#3a4553'
                        }
                    }
                }
            }
        });

        // Update statistics
        this.updateRatingStats(data.statistics);
    }

    updateRatingStats(stats) {
        if (!stats) return;
        
        const statsHtml = `
            <h4>Rating Statistics</h4>
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-value">${stats.mean?.toFixed(2) || '-'}</span>
                    <span class="stat-label">Mean</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${stats.median?.toFixed(2) || '-'}</span>
                    <span class="stat-label">Median</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${stats.std?.toFixed(2) || '-'}</span>
                    <span class="stat-label">Std Dev</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${stats.min?.toFixed(1) || '-'} - ${stats.max?.toFixed(1) || '-'}</span>
                    <span class="stat-label">Range</span>
                </div>
            </div>
        `;
        
        document.getElementById('ratingStats').innerHTML = statsHtml;
    }

    createTopMoviesChart() {
        const ctx = document.getElementById('topMoviesChart').getContext('2d');
        const data = this.data.topMovies;
        
        if (!data?.chart) return;

        this.charts.topMovies = new Chart(ctx, {
            type: 'bar',
            data: data.chart,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: '#252a3a',
                        titleColor: '#ffffff',
                        bodyColor: '#a0aec0',
                        borderColor: '#3a4553',
                        borderWidth: 1,
                        callbacks: {
                            afterBody: function(context) {
                                const movieIndex = context[0].dataIndex;
                                const movie = data.movies[movieIndex];
                                if (movie) {
                                    return [
                                        `Votes: ${movie.vote_count.toLocaleString()}`,
                                        `Avg Rating: ${movie.rating_mean.toFixed(2)}`,
                                        `Genres: ${movie.genres}`
                                    ];
                                }
                                return [];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        grid: {
                            color: '#3a4553'
                        }
                    },
                    y: {
                        grid: {
                            color: '#3a4553'
                        }
                    }
                }
            }
        });
    }

    createGenreChart() {
        const ctx = document.getElementById('genreChart').getContext('2d');
        this.showGenreChart('bar');
    }

    showGenreChart(type) {
        const ctx = document.getElementById('genreChart').getContext('2d');
        const data = this.data.genreData;
        
        if (!data) return;

        // Destroy existing chart
        if (this.charts.genre) {
            this.charts.genre.destroy();
        }

        // Update button states
        document.getElementById('genreBarBtn').classList.toggle('active', type === 'bar');
        document.getElementById('genreScatterBtn').classList.toggle('active', type === 'scatter');

        if (type === 'bar') {
            this.charts.genre = new Chart(ctx, {
                type: 'bar',
                data: data.bar_chart,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: '#252a3a',
                            titleColor: '#ffffff',
                            bodyColor: '#a0aec0',
                            borderColor: '#3a4553',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            beginAtZero: true,
                            grid: {
                                color: '#3a4553'
                            },
                            ticks: {
                                callback: function(value) {
                                    return value.toLocaleString();
                                }
                            }
                        },
                        y: {
                            grid: {
                                color: '#3a4553'
                            }
                        }
                    }
                }
            });
        } else {
            this.charts.genre = new Chart(ctx, {
                type: 'scatter',
                data: data.scatter_chart,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: '#252a3a',
                            titleColor: '#ffffff',
                            bodyColor: '#a0aec0',
                            borderColor: '#3a4553',
                            borderWidth: 1,
                            callbacks: {
                                title: function(context) {
                                    return context[0].raw.label;
                                },
                                label: function(context) {
                                    return [
                                        `Ratings: ${context.raw.x.toLocaleString()}`,
                                        `Avg Rating: ${context.raw.y.toFixed(2)}`
                                    ];
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'logarithmic',
                            title: {
                                display: true,
                                text: 'Number of Ratings (log scale)'
                            },
                            grid: {
                                color: '#3a4553'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Average Rating'
                            },
                            grid: {
                                color: '#3a4553'
                            }
                        }
                    }
                }
            });
        }

        this.currentGenreView = type;
    }

    createTimeSeriesChart() {
        const ctx = document.getElementById('timeSeriesChart').getContext('2d');
        const data = this.data.timeSeries;
        
        if (!data?.chart) return;

        this.charts.timeSeries = new Chart(ctx, {
            type: 'line',
            data: data.chart,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    tooltip: {
                        backgroundColor: '#252a3a',
                        titleColor: '#ffffff',
                        bodyColor: '#a0aec0',
                        borderColor: '#3a4553',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            parser: 'yyyy-MM',
                            tooltipFormat: 'MMM yyyy',
                            displayFormats: {
                                month: 'MMM yy'
                            }
                        },
                        grid: {
                            color: '#3a4553'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: {
                            color: '#3a4553'
                        },
                        ticks: {
                            callback: function(value) {
                                return value.toLocaleString();
                            }
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    }

    createUserActivityChart() {
        const ctx = document.getElementById('userActivityChart').getContext('2d');
        const data = this.data.userStats;
        
        if (!data?.chart) return;

        this.charts.userActivity = new Chart(ctx, {
            type: 'doughnut',
            data: data.chart,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        backgroundColor: '#252a3a',
                        titleColor: '#ffffff',
                        bodyColor: '#a0aec0',
                        borderColor: '#3a4553',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.parsed / total) * 100).toFixed(1);
                                return `${context.label}: ${context.parsed.toLocaleString()} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });

        // Update user statistics
        this.updateUserStats(data.statistics);
    }

    updateUserStats(stats) {
        if (!stats) return;
        
        const statsHtml = `
            <h4>User Statistics</h4>
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-value">${stats.total_users?.toLocaleString() || '-'}</span>
                    <span class="stat-label">Total Users</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${stats.avg_ratings_per_user?.toFixed(1) || '-'}</span>
                    <span class="stat-label">Avg Ratings</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${stats.median_ratings_per_user?.toFixed(1) || '-'}</span>
                    <span class="stat-label">Median Ratings</span>
                </div>
            </div>
        `;
        
        document.getElementById('userStats').innerHTML = statsHtml;
    }

    setupEventListeners() {
        // Add any additional event listeners here
        window.addEventListener('resize', () => {
            Object.values(this.charts).forEach(chart => {
                if (chart) chart.resize();
            });
        });
    }

    showErrorMessage(message) {
        // Create a simple error notification
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #f56565;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            z-index: 1001;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        `;
        errorDiv.textContent = message;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
}

// Global functions for HTML event handlers
function refreshData() {
    if (window.dashboard) {
        window.dashboard.showLoading(true);
        window.dashboard.loadAllData().then(() => {
            window.dashboard.showLoading(false);
            // Reinitialize charts with new data
            Object.values(window.dashboard.charts).forEach(chart => {
                if (chart) chart.destroy();
            });
            window.dashboard.charts = {};
            window.dashboard.initializeCharts();
        });
    }
}

function showGenreChart(type) {
    if (window.dashboard) {
        window.dashboard.showGenreChart(type);
    }
}

function toggleChartType(chartId) {
    // Placeholder for chart type toggling
    console.log('Toggle chart type for:', chartId);
}

function updateTopMoviesLimit() {
    // Placeholder for updating top movies limit
    console.log('Update top movies limit');
}

function toggleTimeSeriesMetric(metric) {
    // Placeholder for time series metric toggling
    console.log('Toggle time series metric:', metric);
}

function showDatasetPage() {
    window.location.href = 'dataset.html';
}

function showDashboard() {
    // Already on dashboard page
    console.log('Already on dashboard page');
}

function showTrends() {
    window.location.href = 'trends.html';
}

function showDocumentation() {
    window.location.href = 'documentation.html';
}

function showReports() {
    window.location.href = 'reports.html';
}

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new MovieLensDashboard();
});