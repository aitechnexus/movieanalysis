// Dataset Manager JavaScript Application
class DatasetManager {
    constructor() {
        console.log('DatasetManager: Constructor called');
        // Use relative URL for API calls to work with nginx proxy
        this.apiBaseUrl = '/api';
        this.uploadedFiles = [];
        this.currentDataset = null;
        
        console.log('DatasetManager: Constructor completed, init() will be called separately');
    }

    async init() {
        console.log('DatasetManager: Starting initialization...');
        this.setupEventListeners();
        console.log('DatasetManager: Event listeners set up');
        await this.checkApiConnection();
        console.log('DatasetManager: API connection checked');
        await this.refreshDatasetStatus();
        console.log('DatasetManager: Dataset status refreshed');
        this.showLoading(false); // Hide loading overlay after initialization
        console.log('DatasetManager: Initialization complete, loading overlay hidden');
    }

    async checkApiConnection() {
        try {
            // Add a small delay to ensure DOM is fully loaded
            await new Promise(resolve => setTimeout(resolve, 100));
            
            const response = await fetch(`${this.apiBaseUrl}/status`);
            if (response.ok) {
                this.updateApiStatus(true);
            } else {
                this.updateApiStatus(false);
            }
        } catch (error) {
            console.error('API connection failed:', error);
            this.updateApiStatus(false);
        }
    }

    updateApiStatus(connected) {
        console.log('updateApiStatus called with connected:', connected);
        
        // Wait for elements to be available with retry mechanism
        const waitForElement = (id, maxRetries = 10) => {
            return new Promise((resolve, reject) => {
                let retries = 0;
                const checkElement = () => {
                    const element = document.getElementById(id);
                    if (element) {
                        console.log(`Element ${id} found after ${retries} retries`);
                        resolve(element);
                    } else if (retries < maxRetries) {
                        retries++;
                        console.log(`Element ${id} not found, retry ${retries}/${maxRetries}`);
                        setTimeout(checkElement, 100);
                    } else {
                        console.error(`Element ${id} not found after ${maxRetries} retries`);
                        reject(new Error(`Element ${id} not found`));
                    }
                };
                checkElement();
            });
        };

        // Use the retry mechanism for both elements
        Promise.all([
            waitForElement('apiStatus'),
            waitForElement('apiStatusText')
        ]).then(([indicator, text]) => {
            if (connected) {
                indicator.classList.remove('offline');
                indicator.classList.add('connected');
                text.textContent = 'API Connected';
            } else {
                indicator.classList.remove('connected');
                indicator.classList.add('offline');
                text.textContent = 'API Disconnected';
            }
        }).catch(error => {
            console.error('Failed to update API status:', error);
        });
    }

    setupEventListeners() {
        // File upload drag and drop
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        // URL input enter key
        document.getElementById('datasetUrl').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.importFromUrl();
            }
        });
    }

    handleDragOver(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        this.addFiles(files);
    }

    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.addFiles(files);
    }

    addFiles(files) {
        const validFiles = files.filter(file => {
            const validExtensions = ['.csv', '.tsv', '.txt'];
            return validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        });

        if (validFiles.length !== files.length) {
            this.showNotification('Some files were skipped. Only CSV, TSV, and TXT files are supported.', 'warning');
        }

        validFiles.forEach(file => {
            if (!this.uploadedFiles.find(f => f.name === file.name)) {
                this.uploadedFiles.push(file);
            }
        });

        this.updateFileList();
        this.updateUploadButton();
    }

    updateFileList() {
        const fileList = document.getElementById('fileList');
        
        if (this.uploadedFiles.length === 0) {
            fileList.innerHTML = '';
            return;
        }

        fileList.innerHTML = this.uploadedFiles.map((file, index) => `
            <div class="file-item">
                <div class="file-info">
                    <i class="fas fa-file-csv"></i>
                    <div>
                        <div class="file-name">${file.name}</div>
                        <div class="file-size">${this.formatFileSize(file.size)}</div>
                    </div>
                </div>
                <button class="file-remove" onclick="datasetManager.removeFile(${index})">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `).join('');
    }

    removeFile(index) {
        this.uploadedFiles.splice(index, 1);
        this.updateFileList();
        this.updateUploadButton();
    }

    updateUploadButton() {
        const uploadBtn = document.getElementById('uploadBtn');
        uploadBtn.disabled = this.uploadedFiles.length === 0;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async refreshDatasetStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/dataset/status`);
            if (response.ok) {
                const data = await response.json();
                this.updateDatasetStatus(data);
            } else {
                this.updateDatasetStatus({
                    current_dataset: 'None loaded',
                    status: 'Not loaded',
                    last_updated: 'Never'
                });
            }
        } catch (error) {
            console.error('Failed to fetch dataset status:', error);
            this.updateDatasetStatus({
                current_dataset: 'Error loading status',
                status: 'Error',
                last_updated: 'Error'
            });
        }
    }

    updateDatasetStatus(status) {
        // Update status display
        const isLoaded = status.loaded || false;
        document.getElementById('currentDataset').textContent = isLoaded ? 'MovieLens Dataset' : 'None loaded';
        document.getElementById('datasetStatus').textContent = isLoaded ? 'Loaded' : 'Not loaded';
        document.getElementById('lastUpdated').textContent = isLoaded ? new Date().toLocaleString() : 'Never';
        
        // Enable/disable navigation buttons based on dataset status
        this.updateNavigationButtons(isLoaded);
    }

    updateNavigationButtons(isLoaded) {
        // Get all navigation buttons that should be enabled/disabled
        const navButtons = document.querySelectorAll('.nav-btn');
        
        navButtons.forEach(button => {
            const buttonText = button.textContent.trim();
            
            // Enable/disable buttons based on dataset status
            // Keep Documentation always enabled, enable others only when data is loaded
            if (buttonText === 'Documentation' || buttonText === 'Dataset') {
                // These buttons are always enabled
                button.disabled = false;
                button.removeAttribute('title');
            } else if (buttonText === 'Dashboard' || buttonText === 'Trends' || buttonText === 'Reports') {
                // These buttons require loaded data
                button.disabled = !isLoaded;
                if (!isLoaded) {
                    button.setAttribute('title', 'Load data first');
                } else {
                    button.removeAttribute('title');
                }
            }
        });
    }

    async loadSampleDataset() {
        this.showLoading(true, 'Loading sample dataset...');
        this.showProgress(true);
        
        try {
            // Simulate progress updates
            this.updateProgress(10, 'Downloading MovieLens 25M dataset...');
            
            const response = await fetch(`${this.apiBaseUrl}/dataset/load-sample`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const result = await response.json();
                
                // Simulate progress
                this.updateProgress(50, 'Extracting files...');
                await this.delay(1000);
                
                this.updateProgress(80, 'Processing data...');
                await this.delay(1500);
                
                this.updateProgress(100, 'Dataset loaded successfully!');
                await this.delay(500);
                
                this.showNotification('Sample dataset loaded successfully!', 'success');
                await this.refreshDatasetStatus();
                this.showDatasetPreview(result.preview);
                
            } else {
                const error = await response.json();
                throw new Error(error.message || 'Failed to load sample dataset');
            }
        } catch (error) {
            console.error('Error loading sample dataset:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
            this.showProgress(false);
        }
    }

    async uploadDataset() {
        if (this.uploadedFiles.length === 0) {
            this.showNotification('Please select files to upload', 'warning');
            return;
        }

        this.showLoading(true, 'Uploading dataset...');
        this.showProgress(true);

        try {
            const formData = new FormData();
            this.uploadedFiles.forEach(file => {
                formData.append('files', file);
            });

            this.updateProgress(20, 'Uploading files...');

            const response = await fetch(`${this.apiBaseUrl}/dataset/upload`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                
                this.updateProgress(60, 'Processing uploaded files...');
                await this.delay(1000);
                
                this.updateProgress(90, 'Validating dataset...');
                await this.delay(500);
                
                this.updateProgress(100, 'Upload completed successfully!');
                await this.delay(500);
                
                this.showNotification('Dataset uploaded successfully!', 'success');
                await this.refreshDatasetStatus();
                this.showDatasetPreview(result.preview);
                
                // Clear uploaded files
                this.uploadedFiles = [];
                this.updateFileList();
                this.updateUploadButton();
                
            } else {
                const error = await response.json();
                throw new Error(error.message || 'Failed to upload dataset');
            }
        } catch (error) {
            console.error('Error uploading dataset:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
            this.showProgress(false);
        }
    }

    async importFromUrl() {
        const url = document.getElementById('datasetUrl').value.trim();
        
        if (!url) {
            this.showNotification('Please enter a valid URL', 'warning');
            return;
        }

        this.showLoading(true, 'Importing from URL...');
        this.showProgress(true);

        try {
            this.updateProgress(10, 'Validating URL...');
            
            const response = await fetch(`${this.apiBaseUrl}/dataset/load-url`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url })
            });

            if (response.ok) {
                const result = await response.json();
                
                this.updateProgress(40, 'Downloading dataset...');
                await this.delay(2000);
                
                this.updateProgress(70, 'Extracting and processing...');
                await this.delay(1500);
                
                this.updateProgress(100, 'Import completed successfully!');
                await this.delay(500);
                
                this.showNotification('Dataset imported successfully!', 'success');
                await this.refreshDatasetStatus();
                this.showDatasetPreview(result.preview);
                
                // Clear URL input
                document.getElementById('datasetUrl').value = '';
                
            } else {
                const error = await response.json();
                throw new Error(error.message || 'Failed to import dataset');
            }
        } catch (error) {
            console.error('Error importing dataset:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
            this.showProgress(false);
        }
    }

    showDatasetPreview(preview) {
        const previewSection = document.getElementById('datasetPreviewSection');
        const previewStats = document.getElementById('previewStats');
        const previewTables = document.getElementById('previewTables');

        if (!preview) {
            previewSection.style.display = 'none';
            return;
        }

        // Update statistics
        previewStats.innerHTML = `
            <div class="preview-stat">
                <div class="preview-stat-value">${preview.n_movies?.toLocaleString() || '-'}</div>
                <div class="preview-stat-label">Movies</div>
            </div>
            <div class="preview-stat">
                <div class="preview-stat-value">${preview.n_ratings?.toLocaleString() || '-'}</div>
                <div class="preview-stat-label">Ratings</div>
            </div>
            <div class="preview-stat">
                <div class="preview-stat-value">${preview.n_users?.toLocaleString() || '-'}</div>
                <div class="preview-stat-label">Users</div>
            </div>
            <div class="preview-stat">
                <div class="preview-stat-value">${preview.date_range ? `${preview.date_range[0]} - ${preview.date_range[1]}` : '-'}</div>
                <div class="preview-stat-label">Date Range</div>
            </div>
        `;

        // Show preview section
        previewSection.style.display = 'block';
    }

    async analyzeDataset() {
        this.showLoading(true, 'Running analysis...');
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/dataset/analyze`, {
                method: 'POST'
            });

            if (response.ok) {
                this.showNotification('Analysis completed! Redirecting to dashboard...', 'success');
                setTimeout(() => {
                    this.showDashboard();
                }, 1500);
            } else {
                const error = await response.json();
                throw new Error(error.message || 'Failed to analyze dataset');
            }
        } catch (error) {
            console.error('Error analyzing dataset:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async clearDataset() {
        if (!confirm('Are you sure you want to clear the current dataset? This action cannot be undone.')) {
            return;
        }

        this.showLoading(true, 'Clearing dataset...');

        try {
            const response = await fetch(`${this.apiBaseUrl}/dataset/clear`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.showNotification('Dataset cleared successfully!', 'success');
                await this.refreshDatasetStatus();
                document.getElementById('datasetPreviewSection').style.display = 'none';
            } else {
                const error = await response.json();
                throw new Error(error.message || 'Failed to clear dataset');
            }
        } catch (error) {
            console.error('Error clearing dataset:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    showLoading(show, text = 'Loading...') {
        console.log(`DatasetManager: showLoading called with show=${show}, text="${text}"`);
        
        // Wait for elements to be available with retry mechanism
        const waitForElement = (id, maxRetries = 10) => {
            return new Promise((resolve, reject) => {
                let retries = 0;
                const checkElement = () => {
                    const element = document.getElementById(id);
                    if (element) {
                        console.log(`Element ${id} found after ${retries} retries`);
                        resolve(element);
                    } else if (retries < maxRetries) {
                        retries++;
                        console.log(`Element ${id} not found, retry ${retries}/${maxRetries}`);
                        setTimeout(checkElement, 100);
                    } else {
                        console.error(`Element ${id} not found after ${maxRetries} retries`);
                        reject(new Error(`Element ${id} not found`));
                    }
                };
                checkElement();
            });
        };

        // Use the retry mechanism for both elements
        Promise.all([
            waitForElement('loadingOverlay'),
            waitForElement('loadingText')
        ]).then(([overlay, loadingText]) => {
            overlay.classList.toggle('hidden', !show);
            loadingText.textContent = text;
            console.log(`DatasetManager: Loading overlay ${show ? 'shown' : 'hidden'}`);
        }).catch(error => {
            console.error('Failed to show/hide loading overlay:', error);
        });
    }

    showProgress(show) {
        const progressSection = document.getElementById('progressSection');
        progressSection.style.display = show ? 'block' : 'none';
        
        if (!show) {
            this.updateProgress(0, '');
        }
    }

    updateProgress(percentage, text, details = '') {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const progressDetails = document.getElementById('progressDetails');
        
        progressFill.style.width = `${percentage}%`;
        progressText.textContent = text;
        progressDetails.textContent = details;
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            color: white;
            z-index: 1001;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 400px;
        `;
        
        const colors = {
            success: '#48bb78',
            error: '#f56565',
            warning: '#ed8936',
            info: '#4299e1'
        };
        
        notification.style.background = colors[type] || colors.info;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    showDashboard() {
        window.location.href = 'index.html';
    }
}

// Global functions for HTML event handlers
function refreshDatasetStatus() {
    if (window.datasetManager) {
        window.datasetManager.refreshDatasetStatus();
    }
}

function loadSampleDataset() {
    if (window.datasetManager) {
        window.datasetManager.loadSampleDataset();
    }
}

function uploadDataset() {
    if (window.datasetManager) {
        window.datasetManager.uploadDataset();
    }
}

function importFromUrl() {
    if (window.datasetManager) {
        window.datasetManager.importFromUrl();
    }
}

function analyzeDataset() {
    if (window.datasetManager) {
        window.datasetManager.analyzeDataset();
    }
}

function clearDataset() {
    if (window.datasetManager) {
        window.datasetManager.clearDataset();
    }
}

function showDatasetPage() {
    // Already on dataset page
    console.log('Already on dataset page');
}

function showDashboard() {
    window.location.href = 'index.html';
}

function showDatasetPage() {
    // Already on dataset page
    console.log('Already on dataset page');
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

// Initialize the dataset manager when the page loads
document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOMContentLoaded event fired, creating DatasetManager...');
    
    // Show loading overlay immediately
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.remove('hidden');
        console.log('Loading overlay shown');
    }
    
    window.datasetManager = new DatasetManager();
    console.log('DatasetManager created, now calling init()...');
    await window.datasetManager.init();
    console.log('DatasetManager initialization completed');
});