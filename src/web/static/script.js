/**
 * Pokemon RL Bot Dashboard JavaScript
 * Handles real-time communication, game controls, and UI updates
 */

class PokemonDashboard {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.isStreaming = false;
        this.isTraining = false;
        
        // Charts
        this.rewardChart = null;
        this.performanceChart = null;
        
        // Metrics tracking
        this.rewardHistory = [];
        this.performanceHistory = [];
        this.maxHistoryLength = 100;
        
        // FPS tracking
        this.lastFrameTime = 0;
        this.fpsCounter = 0;
        this.fpsUpdateInterval = null;
        
        // Keyboard controls
        this.keyboardEnabled = true;
        this.actionMapping = {
            'ArrowUp': 0,     // Up
            'ArrowLeft': 1,   // Left
            'ArrowRight': 2,  // Right
            'ArrowDown': 3,   // Down
            'KeyZ': 4,        // A button
            'KeyX': 5,        // B button
            'Enter': 6,       // Start
            'Space': 7,       // Select
            'Escape': 8       // No action
        };
        
        this.init();
    }
    
    init() {
        this.initializeSocket();
        this.setupEventListeners();
        this.initializeCharts();
        this.hideLoadingOverlay();
        this.startFPSCounter();
        
        console.log('Pokemon Dashboard initialized');
    }
    
    initializeSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('✅ Connected to server via WebSocket');
            this.isConnected = true;
            this.updateConnectionStatus(true);
            this.showNotification('Connected to server', 'success');
            
            // Test: Request a screenshot immediately
            console.log('🔄 Requesting test screenshot via WebSocket');
            this.socket.emit('request_screenshot');
        });
        
        this.socket.on('disconnect', () => {
            console.log('❌ Disconnected from server via WebSocket');
            this.isConnected = false;
            this.updateConnectionStatus(false);
            this.showNotification('Disconnected from server', 'error');
        });
        
        this.socket.on('frame', (data) => {
            console.log('Received frame data:', data ? 'yes' : 'no', data && data.data ? `${data.data.length} chars` : 'no data');
            this.updateGameFrame(data);
        });
        
        this.socket.on('metrics', (data) => {
            this.updateMetrics(data);
        });
        
        this.socket.on('training_status', (data) => {
            this.updateTrainingStatus(data);
        });
        
        this.socket.on('action_result', (data) => {
            this.updateActionResult(data);
        });
        
        this.socket.on('episode_complete', (data) => {
            this.handleEpisodeComplete(data);
        });
        
        this.socket.on('notification', (data) => {
            this.showNotification(data.message, data.level);
        });
        
        this.socket.on('error', (data) => {
            this.showNotification(data.message, 'error');
        });
        
        this.socket.on('streaming_status', (data) => {
            this.isStreaming = data.active;
            this.updateStreamingControls();
        });
        
        this.socket.on('connection_status', (data) => {
            console.log('Connection status:', data);
        });
    }
    
    setupEventListeners() {
        // Streaming controls
        document.getElementById('start-streaming-btn').addEventListener('click', () => {
            this.startStreaming();
        });
        
        document.getElementById('stop-streaming-btn').addEventListener('click', () => {
            this.stopStreaming();
        });
        
        // Training controls
        document.getElementById('start-training-btn').addEventListener('click', () => {
            this.startTraining();
        });
        
        document.getElementById('stop-training-btn').addEventListener('click', () => {
            this.stopTraining();
        });
        
        document.getElementById('reset-env-btn').addEventListener('click', () => {
            this.resetEnvironment();
        });
        
        // Manual control buttons
        document.querySelectorAll('.dpad-btn, .action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = parseInt(e.currentTarget.dataset.action);
                this.sendAction(action);
                this.highlightButton(e.currentTarget);
            });
        });
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (this.keyboardEnabled && !e.repeat) {
                this.handleKeyPress(e);
            }
        });
        
        // Prevent arrow keys from scrolling
        document.addEventListener('keydown', (e) => {
            if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) {
                e.preventDefault();
            }
        });
        
        // Game frame click to request frame
        document.getElementById('game-frame').addEventListener('click', () => {
            if (this.isStreaming) {
                this.socket.emit('request_frame');
            }
        });
        
        // Window focus/blur events
        window.addEventListener('focus', () => {
            this.keyboardEnabled = true;
        });
        
        window.addEventListener('blur', () => {
            this.keyboardEnabled = false;
        });
    }
    
    initializeCharts() {
        // Reward Chart
        const rewardCtx = document.getElementById('reward-chart').getContext('2d');
        this.rewardChart = new Chart(rewardCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Episode Reward',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#ffffff' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#b0b0b0' },
                        grid: { color: '#404040' }
                    },
                    y: {
                        ticks: { color: '#b0b0b0' },
                        grid: { color: '#404040' }
                    }
                }
            }
        });
        
        // Performance Chart
        const perfCtx = document.getElementById('performance-chart').getContext('2d');
        this.performanceChart = new Chart(perfCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Loss',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Learning Rate',
                        data: [],
                        borderColor: '#27ae60',
                        backgroundColor: 'rgba(39, 174, 96, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#ffffff' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#b0b0b0' },
                        grid: { color: '#404040' }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        ticks: { color: '#b0b0b0' },
                        grid: { color: '#404040' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        ticks: { color: '#b0b0b0' },
                        grid: { drawOnChartArea: false }
                    }
                }
            }
        });
    }
    
    startStreaming() {
        this.socket.emit('start_streaming');
        this.showLoadingOverlay('Starting stream...');
        
        setTimeout(() => {
            this.hideLoadingOverlay();
            this.isStreaming = true;
            this.updateStreamingControls();
            this.hideGameOverlay();
            this.startFrameRequests();
        }, 1000);
    }
    
    stopStreaming() {
        this.socket.emit('stop_streaming');
        this.isStreaming = false;
        this.updateStreamingControls();
        this.showGameOverlay();
        this.stopFrameRequests();
    }
    
    startFrameRequests() {
        if (this.frameRequestInterval) {
            clearInterval(this.frameRequestInterval);
        }
        
        this.frameRequestInterval = setInterval(() => {
            if (this.isStreaming && this.isConnected) {
                this.socket.emit('request_frame');
            }
        }, 33); // ~30 FPS
    }
    
    stopFrameRequests() {
        if (this.frameRequestInterval) {
            clearInterval(this.frameRequestInterval);
            this.frameRequestInterval = null;
        }
    }
    
    startTraining() {
        fetch('/api/start_training', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    this.showNotification(data.error, 'error');
                } else {
                    this.isTraining = true;
                    this.updateTrainingControls();
                    this.showNotification('Training started', 'success');
                    this.socket.emit('start_metrics');
                }
            })
            .catch(error => {
                this.showNotification('Failed to start training', 'error');
                console.error('Error:', error);
            });
    }
    
    stopTraining() {
        fetch('/api/stop_training', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                this.isTraining = false;
                this.updateTrainingControls();
                this.showNotification('Training stopped', 'warning');
                this.socket.emit('stop_metrics');
            })
            .catch(error => {
                this.showNotification('Failed to stop training', 'error');
                console.error('Error:', error);
            });
    }
    
    resetEnvironment() {
        fetch('/api/reset_environment', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    this.showNotification(data.error, 'error');
                } else {
                    this.showNotification('Environment reset', 'success');
                    this.resetEpisodeStats();
                }
            })
            .catch(error => {
                this.showNotification('Failed to reset environment', 'error');
                console.error('Error:', error);
            });
    }
    
    sendAction(action) {
        if (this.isConnected) {
            fetch('/api/manual_action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: action })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    this.showNotification(data.error, 'error');
                } else {
                    this.updateActionResult(data);
                }
            })
            .catch(error => {
                console.error('Error sending action:', error);
            });
            
            // Update last action display
            const actionNames = ['Up', 'Left', 'Right', 'Down', 'A', 'B', 'Start', 'Select', 'No-op'];
            document.getElementById('last-action').textContent = actionNames[action] || 'Unknown';
        }
    }
    
    handleKeyPress(event) {
        const action = this.actionMapping[event.code];
        if (action !== undefined) {
            event.preventDefault();
            this.sendAction(action);
            this.highlightKeyboardAction(event.code);
        }
    }
    
    highlightButton(button) {
        button.style.transform = 'scale(0.95)';
        button.style.backgroundColor = '#27ae60';
        
        setTimeout(() => {
            button.style.transform = '';
            button.style.backgroundColor = '';
        }, 150);
    }
    
    highlightKeyboardAction(keyCode) {
        // Find corresponding button and highlight it
        const action = this.actionMapping[keyCode];
        const button = document.querySelector(`[data-action="${action}"]`);
        if (button) {
            this.highlightButton(button);
        }
    }
    
    updateGameFrame(data) {
        console.log('updateGameFrame called with:', data ? 'data exists' : 'no data');
        const gameFrame = document.getElementById('game-frame');
        // Support both PNG and JPEG formats
        if (data && data.data) {
            gameFrame.src = `data:image/png;base64,${data.data}`;
            console.log('Updated game frame src with', data.data.length, 'characters');
        } else {
            console.log('No frame data to display');
        }
        
        // Update resolution display
        if (data.size) {
            document.getElementById('resolution').textContent = `${data.size[0]}x${data.size[1]}`;
        }
        
        // Update FPS
        const now = performance.now();
        if (this.lastFrameTime > 0) {
            const fps = 1000 / (now - this.lastFrameTime);
            this.fpsCounter = Math.round(fps * 10) / 10;
        }
        this.lastFrameTime = now;
    }
    
    updateMetrics(data) {
        // Update episode reward chart
        if (data.episode_rewards && data.episode_rewards.length > 0) {
            this.rewardHistory = data.episode_rewards.slice(-this.maxHistoryLength);
            this.rewardChart.data.labels = this.rewardHistory.map((_, i) => i + 1);
            this.rewardChart.data.datasets[0].data = this.rewardHistory;
            this.rewardChart.update('none');
        }
        
        // Update performance chart
        if (data.loss_history) {
            this.performanceHistory = data.loss_history.slice(-this.maxHistoryLength);
            this.performanceChart.data.labels = this.performanceHistory.map((_, i) => i + 1);
            this.performanceChart.data.datasets[0].data = this.performanceHistory;
            this.performanceChart.update('none');
        }
        
        // Update metric displays
        if (data.average_reward !== undefined) {
            document.getElementById('avg-reward').textContent = data.average_reward.toFixed(2);
        }
        if (data.best_reward !== undefined) {
            document.getElementById('best-reward').textContent = data.best_reward.toFixed(2);
        }
        if (data.episodes_completed !== undefined) {
            document.getElementById('episodes-completed').textContent = data.episodes_completed;
        }
        if (data.training_time !== undefined) {
            document.getElementById('training-time').textContent = this.formatTime(data.training_time);
        }
    }
    
    updateTrainingStatus(data) {
        document.getElementById('current-episode').textContent = data.episode || 0;
        document.getElementById('total-reward').textContent = (data.total_reward || 0).toFixed(2);
        
        if (data.done) {
            this.handleEpisodeComplete(data);
        }
    }
    
    updateActionResult(data) {
        document.getElementById('total-reward').textContent = (data.total_reward || 0).toFixed(2);
        
        // Increment step count
        const stepCount = document.getElementById('step-count');
        stepCount.textContent = parseInt(stepCount.textContent) + 1;
    }
    
    handleEpisodeComplete(data) {
        this.showNotification(`Episode ${data.episode} completed! Final reward: ${data.final_reward}`, 'success');
        this.resetEpisodeStats();
        
        // Update episode counter
        document.getElementById('current-episode').textContent = data.episode || 0;
    }
    
    resetEpisodeStats() {
        document.getElementById('total-reward').textContent = '0';
        document.getElementById('step-count').textContent = '0';
        document.getElementById('last-action').textContent = 'None';
    }
    
    updateConnectionStatus(connected) {
        const statusIcon = document.getElementById('connection-status');
        const statusText = document.getElementById('connection-text');
        
        if (connected) {
            statusIcon.className = 'fas fa-circle connected';
            statusText.textContent = 'Connected';
        } else {
            statusIcon.className = 'fas fa-circle disconnected';
            statusText.textContent = 'Disconnected';
        }
    }
    
    updateStreamingControls() {
        const startBtn = document.getElementById('start-streaming-btn');
        const stopBtn = document.getElementById('stop-streaming-btn');
        
        startBtn.disabled = this.isStreaming;
        stopBtn.disabled = !this.isStreaming;
    }
    
    updateTrainingControls() {
        const startBtn = document.getElementById('start-training-btn');
        const stopBtn = document.getElementById('stop-training-btn');
        const statusIcon = document.getElementById('training-status');
        const statusText = document.getElementById('training-text');
        
        startBtn.disabled = this.isTraining;
        stopBtn.disabled = !this.isTraining;
        
        if (this.isTraining) {
            statusIcon.className = 'fas fa-play-circle active';
            statusText.textContent = 'Training';
        } else {
            statusIcon.className = 'fas fa-play-circle inactive';
            statusText.textContent = 'Stopped';
        }
    }
    
    showGameOverlay() {
        document.getElementById('game-overlay').style.display = 'flex';
    }
    
    hideGameOverlay() {
        document.getElementById('game-overlay').style.display = 'none';
    }
    
    showLoadingOverlay(message = 'Loading...') {
        const overlay = document.getElementById('loading-overlay');
        overlay.querySelector('p').textContent = message;
        overlay.classList.remove('hidden');
    }
    
    hideLoadingOverlay() {
        document.getElementById('loading-overlay').classList.add('hidden');
    }
    
    showNotification(message, level = 'info') {
        const notificationsContainer = document.getElementById('notifications');
        const notification = document.createElement('div');
        notification.className = `notification ${level}`;
        
        notification.innerHTML = `
            ${message}
            <button class="close-btn">&times;</button>
        `;
        
        // Add close functionality
        notification.querySelector('.close-btn').addEventListener('click', () => {
            notification.remove();
        });
        
        notificationsContainer.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
    
    startFPSCounter() {
        this.fpsUpdateInterval = setInterval(() => {
            document.getElementById('fps-counter').textContent = this.fpsCounter.toFixed(1);
        }, 1000);
    }
    
    formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    // API Methods
    async fetchStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            this.isTraining = data.training_active;
            this.updateTrainingControls();
            this.updateTrainingStatus(data);
            
            return data;
        } catch (error) {
            console.error('Error fetching status:', error);
            return null;
        }
    }
    
    async fetchMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const data = await response.json();
            this.updateMetrics(data);
            return data;
        } catch (error) {
            console.error('Error fetching metrics:', error);
            return null;
        }
    }
    
    // Lifecycle methods
    destroy() {
        this.stopFrameRequests();
        
        if (this.fpsUpdateInterval) {
            clearInterval(this.fpsUpdateInterval);
        }
        
        if (this.socket) {
            this.socket.disconnect();
        }
        
        if (this.rewardChart) {
            this.rewardChart.destroy();
        }
        
        if (this.performanceChart) {
            this.performanceChart.destroy();
        }
        
        console.log('Dashboard destroyed');
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new PokemonDashboard();
    
    // Initial status fetch
    window.dashboard.fetchStatus();
    
    // Periodic status updates
    setInterval(() => {
        window.dashboard.fetchStatus();
    }, 10000); // Every 10 seconds
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.destroy();
    }
});

// Debug helpers (available in console)
window.debugHelpers = {
    sendAction: (action) => window.dashboard.sendAction(action),
    startStreaming: () => window.dashboard.startStreaming(),
    stopStreaming: () => window.dashboard.stopStreaming(),
    startTraining: () => window.dashboard.startTraining(),
    stopTraining: () => window.dashboard.stopTraining(),
    resetEnvironment: () => window.dashboard.resetEnvironment(),
    showNotification: (msg, level) => window.dashboard.showNotification(msg, level)
};
