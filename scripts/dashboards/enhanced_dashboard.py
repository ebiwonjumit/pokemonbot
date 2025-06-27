#!/usr/bin/env python3
"""
Enhanced Pokemon RL Bot Dashboard with Screen Capture
Web interface with live emulator feed
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import time
import threading
import subprocess
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pokemon-rl-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
game_process = None
bot_status = {
    'running': False,
    'episode': 0,
    'reward': 0.0,
    'steps': 0,
    'game_connected': False
}

# Screen capture thread
capture_thread = None
capture_running = False

def capture_screen():
    """Capture the current screen and return as base64 encoded image"""
    try:
        os.environ['DISPLAY'] = ':99'
        
        # Capture screen with scrot
        subprocess.run(['scrot', '/tmp/current_screen.png'], 
                      capture_output=True, check=True, timeout=5)
        
        # Read and encode the image
        with open('/tmp/current_screen.png', 'rb') as f:
            image_data = f.read()
            
        # Convert to base64 for web display
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return f'data:image/png;base64,{base64_image}'
        
    except Exception as e:
        print(f'Screen capture error: {e}')
        return None

def screen_capture_loop():
    """Continuously capture screen and emit to clients"""
    global capture_running
    capture_running = True
    
    while capture_running:
        try:
            image_data = capture_screen()
            if image_data:
                socketio.emit('screen_update', {'image': image_data})
            time.sleep(0.5)  # Capture at ~2 FPS
        except Exception as e:
            print(f'Capture loop error: {e}')
            time.sleep(1)

@app.route('/')
def index():
    """Main dashboard page."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pokemon RL Bot Dashboard</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f0f0; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            .header { text-align: center; color: #333; margin-bottom: 30px; }
            .status { display: flex; gap: 20px; margin-bottom: 30px; }
            .status-card { flex: 1; padding: 20px; background: #f8f9fa; border-radius: 8px; text-align: center; }
            .status-card.active { background: #d4edda; border: 2px solid #28a745; }
            .status-card.inactive { background: #f8d7da; border: 2px solid #dc3545; }
            .controls { text-align: center; margin: 30px 0; }
            .btn { padding: 12px 24px; margin: 10px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            .btn-primary { background: #007bff; color: white; }
            .btn-success { background: #28a745; color: white; }
            .btn-danger { background: #dc3545; color: white; }
            .btn:hover { opacity: 0.8; }
            .log { background: #f8f9fa; padding: 20px; border-radius: 8px; height: 300px; overflow-y: scroll; font-family: monospace; }
            .game-container { text-align: center; margin: 20px 0; }
            #game-frame { max-width: 800px; max-height: 600px; background: #000; border: 2px solid #333; margin: 20px auto; display: block; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎮 Pokemon RL Bot Dashboard</h1>
                <p>Real-time monitoring and control for your Pokemon AI</p>
            </div>
            
            <div class="status">
                <div id="game-status" class="status-card inactive">
                    <h3>🎯 Game Status</h3>
                    <p id="game-text">Disconnected</p>
                </div>
                <div id="bot-status" class="status-card inactive">
                    <h3>🤖 Bot Status</h3>
                    <p id="bot-text">Stopped</p>
                </div>
                <div id="training-status" class="status-card inactive">
                    <h3>📈 Training</h3>
                    <p id="training-text">Not Started</p>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn btn-success" onclick="startGame()">🎮 Start Game</button>
                <button class="btn btn-primary" onclick="startBot()">🤖 Start Bot</button>
                <button class="btn btn-danger" onclick="stopAll()">⏹️ Stop All</button>
                <button class="btn btn-primary" onclick="takeScreenshot()">📷 Screenshot</button>
                <button class="btn btn-primary" onclick="toggleCapture()">📺 Toggle Feed</button>
            </div>
            
            <div class="game-container">
                <h3>🎮 Live Emulator Feed</h3>
                <img id="game-frame" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjYwMCI+PHJlY3Qgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgZmlsbD0iIzAwMCIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmaWxsPSIjZmZmIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+V2FpdGluZyBmb3IgZW11bGF0b3IuLi48L3RleHQ+PC9zdmc+" alt="Game Feed" />
            </div>
            
            <div class="log" id="log-output">
                <div>🎮 Pokemon RL Bot Dashboard Started</div>
                <div>📡 Connected to server</div>
                <div>⏳ Ready for commands...</div>
            </div>
        </div>
        
        <script>
            const socket = io();
            let captureEnabled = false;
            
            function addLog(message) {
                const log = document.getElementById('log-output');
                const time = new Date().toLocaleTimeString();
                log.innerHTML += '<div>[' + time + '] ' + message + '</div>';
                log.scrollTop = log.scrollHeight;
            }
            
            function updateStatus(gameRunning, botRunning, trainingActive) {
                const gameStatus = document.getElementById('game-status');
                const botStatus = document.getElementById('bot-status');
                const trainingStatus = document.getElementById('training-status');
                
                gameStatus.className = 'status-card ' + (gameRunning ? 'active' : 'inactive');
                document.getElementById('game-text').textContent = gameRunning ? 'Connected' : 'Disconnected';
                
                botStatus.className = 'status-card ' + (botRunning ? 'active' : 'inactive');
                document.getElementById('bot-text').textContent = botRunning ? 'Running' : 'Stopped';
                
                trainingStatus.className = 'status-card ' + (trainingActive ? 'active' : 'inactive');
                document.getElementById('training-text').textContent = trainingActive ? 'Training' : 'Not Started';
            }
            
            function startGame() {
                addLog('🎮 Starting Pokemon Leaf Green...');
                socket.emit('start_game');
            }
            
            function startBot() {
                addLog('🤖 Starting bot control...');
                socket.emit('start_bot');
            }
            
            function stopAll() {
                addLog('⏹️ Stopping all processes...');
                socket.emit('stop_all');
            }
            
            function takeScreenshot() {
                addLog('📷 Taking screenshot...');
                socket.emit('take_screenshot');
            }
            
            function toggleCapture() {
                captureEnabled = !captureEnabled;
                if (captureEnabled) {
                    addLog('📺 Screen capture enabled');
                    socket.emit('start_capture');
                } else {
                    addLog('📺 Screen capture disabled');
                    socket.emit('stop_capture');
                }
            }
            
            // Socket events
            socket.on('status_update', function(data) {
                updateStatus(data.game_running, data.bot_running, data.training_active);
            });
            
            socket.on('log_message', function(data) {
                addLog(data.message);
            });
            
            socket.on('screen_update', function(data) {
                if (captureEnabled && data.image) {
                    document.getElementById('game-frame').src = data.image;
                }
            });
            
            socket.on('connect', function() {
                addLog('✅ Connected to Pokemon RL Bot server');
            });
            
            socket.on('disconnect', function() {
                addLog('❌ Disconnected from server');
                updateStatus(false, false, false);
            });
        </script>
    </body>
    </html>
    '''

@socketio.on('start_game')
def handle_start_game():
    """Start the Pokemon game."""
    global game_process
    
    try:
        if game_process is None or game_process.poll() is not None:
            os.environ['DISPLAY'] = ':99'
            rom_path = "roms/pokemon_leaf_green.gba"
            game_process = subprocess.Popen([
                "vbam", 
                "--no-opengl", 
                "--frameskip=0", 
                "--no-show-speed",
                rom_path
            ])
            bot_status['game_connected'] = True
            
            emit('log_message', {'message': '✅ Pokemon Leaf Green started'})
            emit('status_update', {
                'game_running': True,
                'bot_running': bot_status['running'],
                'training_active': False
            })
        else:
            emit('log_message', {'message': '⚠️ Game is already running'})
    except Exception as e:
        emit('log_message', {'message': f'❌ Failed to start game: {e}'})

@socketio.on('start_capture')
def handle_start_capture():
    """Start screen capture."""
    global capture_thread, capture_running
    
    if not capture_running:
        capture_thread = threading.Thread(target=screen_capture_loop, daemon=True)
        capture_thread.start()
        emit('log_message', {'message': '📺 Screen capture started'})
    else:
        emit('log_message', {'message': '📺 Screen capture already running'})

@socketio.on('stop_capture')
def handle_stop_capture():
    """Stop screen capture."""
    global capture_running
    
    capture_running = False
    emit('log_message', {'message': '📺 Screen capture stopped'})

@socketio.on('start_bot')
def handle_start_bot():
    """Start the bot (simulation for now)."""
    bot_status['running'] = True
    emit('log_message', {'message': '🤖 Bot started (demo mode)'})
    emit('status_update', {
        'game_running': bot_status['game_connected'],
        'bot_running': True,
        'training_active': True
    })
    
    # Simulate bot activity
    def simulate_bot():
        episode = 0
        while bot_status['running']:
            episode += 1
            bot_status['episode'] = episode
            bot_status['reward'] += 1.5
            bot_status['steps'] += 10
            
            socketio.emit('log_message', {
                'message': f'🎯 Episode {episode}: Reward {bot_status["reward"]:.1f}, Steps: {bot_status["steps"]}'
            })
            time.sleep(3)
    
    threading.Thread(target=simulate_bot, daemon=True).start()

@socketio.on('stop_all')
def handle_stop_all():
    """Stop all processes."""
    global game_process, capture_running
    
    bot_status['running'] = False
    bot_status['game_connected'] = False
    capture_running = False
    
    if game_process and game_process.poll() is None:
        game_process.terminate()
        try:
            game_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            game_process.kill()
    
    emit('log_message', {'message': '⏹️ All processes stopped'})
    emit('status_update', {
        'game_running': False,
        'bot_running': False,
        'training_active': False
    })

@socketio.on('take_screenshot')
def handle_screenshot():
    """Take a screenshot."""
    try:
        image_data = capture_screen()
        if image_data:
            # Save screenshot
            import base64
            image_binary = base64.b64decode(image_data.split(',')[1])
            with open('data/screenshots/dashboard_screenshot.png', 'wb') as f:
                f.write(image_binary)
            emit('log_message', {'message': '📷 Screenshot saved to data/screenshots/'})
        else:
            emit('log_message', {'message': '❌ Screenshot failed'})
    except Exception as e:
        emit('log_message', {'message': f'❌ Screenshot failed: {e}'})

if __name__ == '__main__':
    print("🎮 Starting Pokemon RL Bot Dashboard...")
    print("📱 Open http://localhost:7500 in your browser")
    print("⏹️ Press Ctrl+C to stop")
    
    try:
        socketio.run(app, host='0.0.0.0', port=7500, debug=False)
    except KeyboardInterrupt:
        print("\n⏹️ Dashboard stopped by user")
        if game_process and game_process.poll() is None:
            game_process.terminate()
