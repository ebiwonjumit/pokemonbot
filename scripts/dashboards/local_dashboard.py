#!/usr/bin/env python3
"""
Pokemon RL Bot - Local Dashboard (macOS/Linux with GUI)
Optimized for local development with native screen capture and GUI support
"""

import os
import sys
import time
import base64
import subprocess
import signal
import threading
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import psutil

# Add src to path for imports (go up two levels from scripts/dashboards to root)
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

# Simple logging setup for local development
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import our utilities, fallback to simple alternatives if not available
try:
    from utils.logger import setup_logger
    from utils.metrics import MetricsCollector
except ImportError:
    logger.warning("Could not import utils modules, using simple alternatives")
    setup_logger = lambda name: logger
    
    class MockMetrics:
        def collect(self):
            return {}
    
    MetricsCollector = MockMetrics

class LocalDashboard:
    def __init__(self):
        # Get project root path
        self.project_root = Path(__file__).parent.parent.parent
        
        self.app = Flask(__name__, 
                        template_folder=str(self.project_root / 'src/web/templates'),
                        static_folder=str(self.project_root / 'src/web/static'))
        self.app.config['SECRET_KEY'] = 'pokemon-rl-bot-local'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        try:
            self.logger = setup_logger("local_dashboard")
        except:
            self.logger = logger
            
        try:
            self.metrics = MetricsCollector()
        except:
            self.metrics = MockMetrics()
        
        # Local paths
        self.rom_path = self.project_root / "roms" / "pokemon_leaf_green.gba"
        self.logs_dir = self.project_root / "logs"
        
        # Process tracking
        self.emulator_process = None
        self.bot_process = None
        self.is_running = False
        
        # Screenshot thread
        self.screenshot_thread = None
        self.screenshot_running = False
        
        self.setup_routes()
        self.setup_socketio()
        
    def setup_routes(self):
        """Set up Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
            
        @self.app.route('/api/status')
        def status():
            return jsonify({
                'emulator_running': self.is_emulator_running(),
                'bot_running': self.is_bot_running(),
                'rom_exists': self.rom_path.exists(),
                'system_info': self.get_system_info(),
                'metrics': self.get_metrics()
            })
            
        @self.app.route('/api/start_emulator', methods=['POST'])
        def start_emulator():
            success, message = self.start_emulator()
            return jsonify({'success': success, 'message': message})
            
        @self.app.route('/api/stop_emulator', methods=['POST'])
        def stop_emulator():
            success, message = self.stop_emulator()
            return jsonify({'success': success, 'message': message})
            
        @self.app.route('/api/start_bot', methods=['POST'])
        def start_bot():
            success, message = self.start_bot()
            return jsonify({'success': success, 'message': message})
            
        @self.app.route('/api/stop_bot', methods=['POST'])
        def stop_bot():
            success, message = self.stop_bot()
            return jsonify({'success': success, 'message': message})
            
        @self.app.route('/api/logs')
        def get_logs():
            return jsonify(self.get_recent_logs())
            
        @self.app.route('/api/test_screenshot')
        def test_screenshot():
            screenshot = self.capture_screenshot()
            if screenshot:
                return jsonify({'success': True, 'message': 'Screenshot captured', 'size': len(screenshot)})
            else:
                return jsonify({'success': False, 'message': 'Screenshot failed'})
                
        @self.app.route('/api/start_streaming', methods=['POST'])
        def start_streaming():
            if not self.screenshot_running:
                self.start_screenshot_capture()
                return jsonify({'success': True, 'message': 'Screenshot streaming started'})
            else:
                return jsonify({'success': False, 'message': 'Streaming already active'})
            
    def setup_socketio(self):
        """Set up WebSocket handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info("Client connected to local dashboard")
            emit('status_update', self.get_status_data())
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info("Client disconnected from local dashboard")
            
        @self.socketio.on('request_screenshot')
        def handle_screenshot_request():
            screenshot = self.capture_screenshot()
            if screenshot:
                emit('frame', {'data': screenshot})
                
    def is_emulator_running(self):
        """Check if VBA-M emulator is running"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                name = proc.info['name'].lower()
                cmdline = ' '.join(proc.info['cmdline'] or []).lower()
                if ('vbam' in name or 'visualboy' in name or 
                    'vbam' in cmdline or 'pokemon' in cmdline):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
        
    def is_bot_running(self):
        """Check if the RL bot is running"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'train.py' in cmdline or 'pokemon' in cmdline.lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
        
    def start_emulator(self):
        """Start the VBA-M emulator"""
        try:
            if not self.rom_path.exists():
                return False, f"ROM file not found: {self.rom_path}"
                
            if self.is_emulator_running():
                return False, "Emulator is already running"
                
            # Start VBA-M with appropriate flags for local GUI
            cmd = [
                'vbam',
                '--no-show-speed',
                '--frameskip=0',
                str(self.rom_path)
            ]
            
            self.emulator_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            if self.emulator_process.poll() is None:
                self.logger.info("Emulator started successfully")
                self.start_screenshot_capture()
                return True, "Emulator started successfully"
            else:
                return False, "Emulator failed to start"
                
        except Exception as e:
            self.logger.error(f"Error starting emulator: {e}")
            return False, f"Error starting emulator: {e}"
            
    def stop_emulator(self):
        """Stop the VBA-M emulator"""
        try:
            self.stop_screenshot_capture()
            
            # Kill all VBA-M processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if 'vbam' in proc.info['name'].lower():
                        proc.terminate()
                        proc.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    continue
                    
            if self.emulator_process:
                try:
                    self.emulator_process.terminate()
                    self.emulator_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.emulator_process.kill()
                self.emulator_process = None
                
            self.logger.info("Emulator stopped")
            return True, "Emulator stopped successfully"
            
        except Exception as e:
            self.logger.error(f"Error stopping emulator: {e}")
            return False, f"Error stopping emulator: {e}"
            
    def start_bot(self):
        """Start the RL training bot"""
        try:
            if not self.is_emulator_running():
                return False, "Start emulator first"
                
            if self.is_bot_running():
                return False, "Bot is already running"
                
            # Start the training script
            cmd = [
                sys.executable,
                'scripts/train.py',
                '--config', 'config.json'
            ]
            
            self.bot_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            self.logger.info("Bot started successfully")
            return True, "Bot started successfully"
            
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            return False, f"Error starting bot: {e}"
            
    def stop_bot(self):
        """Stop the RL training bot"""
        try:
            # Kill training processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'train.py' in cmdline:
                        proc.terminate()
                        proc.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    continue
                    
            if self.bot_process:
                try:
                    self.bot_process.terminate()
                    self.bot_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.bot_process.kill()
                self.bot_process = None
                
            self.logger.info("Bot stopped")
            return True, "Bot stopped successfully"
            
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
            return False, f"Error stopping bot: {e}"
            
    def capture_screenshot(self):
        """Capture screenshot of the emulator window (macOS/Linux)"""
        try:
            if sys.platform == 'darwin':  # macOS
                # Use screencapture with temporary file (stdout doesn't work reliably)
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                
                result = subprocess.run([
                    'screencapture', '-x', '-t', 'png', temp_path
                ], capture_output=True, timeout=5)
                
                if result.returncode == 0 and os.path.exists(temp_path):
                    try:
                        with open(temp_path, 'rb') as f:
                            screenshot_data = base64.b64encode(f.read()).decode('utf-8')
                        os.unlink(temp_path)  # Clean up temp file
                        return screenshot_data
                    except Exception as e:
                        self.logger.error(f"Error reading screenshot file: {e}")
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                else:
                    self.logger.error(f"screencapture failed: {result.stderr.decode()}")
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    
            else:  # Linux with GUI
                # Use scrot or import
                for cmd in [['scrot', '-z', '-'], ['import', '-window', 'root', 'png:-']]:
                    try:
                        result = subprocess.run(cmd, capture_output=True, timeout=5)
                        if result.returncode == 0:
                            return base64.b64encode(result.stdout).decode('utf-8')
                    except FileNotFoundError:
                        continue
                        
        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            
        return None
        
    def start_screenshot_capture(self):
        """Start the screenshot capture thread"""
        if self.screenshot_running:
            return
            
        self.screenshot_running = True
        self.screenshot_thread = threading.Thread(target=self._screenshot_loop)
        self.screenshot_thread.daemon = True
        self.screenshot_thread.start()
        
    def stop_screenshot_capture(self):
        """Stop the screenshot capture thread"""
        self.screenshot_running = False
        if self.screenshot_thread:
            self.screenshot_thread.join(timeout=2)
            
    def _screenshot_loop(self):
        """Screenshot capture loop"""
        while self.screenshot_running:
            try:
                screenshot = self.capture_screenshot()
                if screenshot:
                    self.socketio.emit('frame', {'data': screenshot})
                time.sleep(0.5)  # 2 FPS for local testing
            except Exception as e:
                self.logger.error(f"Screenshot loop error: {e}")
                time.sleep(1)
                
    def get_system_info(self):
        """Get local system information"""
        return {
            'platform': sys.platform,
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
        
    def get_metrics(self):
        """Get training metrics"""
        try:
            metrics_file = self.logs_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading metrics: {e}")
            
        return {
            'episodes': 0,
            'total_reward': 0,
            'average_reward': 0,
            'training_time': 0
        }
        
    def get_recent_logs(self):
        """Get recent log entries"""
        logs = []
        try:
            log_files = ['training.log', 'agent.log', 'environment.log']
            for log_file in log_files:
                log_path = self.logs_dir / log_file
                if log_path.exists():
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        logs.extend([
                            {'file': log_file, 'message': line.strip()}
                            for line in lines[-50:]  # Last 50 lines per file
                            if line.strip()
                        ])
        except Exception as e:
            self.logger.error(f"Error reading logs: {e}")
            
        return logs[-100:]  # Return last 100 log entries total
        
    def get_status_data(self):
        """Get complete status data"""
        return {
            'emulator_running': self.is_emulator_running(),
            'bot_running': self.is_bot_running(),
            'rom_exists': self.rom_path.exists(),
            'system_info': self.get_system_info(),
            'metrics': self.get_metrics()
        }
        
    def run(self, host='127.0.0.1', port=7500, debug=False):
        """Run the local dashboard"""
        self.logger.info(f"Starting Pokemon RL Bot Local Dashboard on {host}:{port}")
        
        # Handle shutdown gracefully
        def signal_handler(sig, frame):
            self.logger.info("Shutting down dashboard...")
            self.stop_screenshot_capture()
            self.stop_emulator()
            self.stop_bot()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            self.socketio.run(self.app, 
                            host=host, 
                            port=port, 
                            debug=debug,
                            allow_unsafe_werkzeug=True)
        except KeyboardInterrupt:
            self.logger.info("Dashboard stopped by user")
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
        finally:
            self.stop_screenshot_capture()


def main():
    """Main entry point"""
    dashboard = LocalDashboard()
    
    print("🎮 Pokemon RL Bot - Local Dashboard")
    print("==================================")
    print("📱 Dashboard: http://127.0.0.1:7500")
    print("🎯 Optimized for local development")
    print("⭐️ Features: Native GUI, Screen Capture, Process Management")
    print()
    
    dashboard.run(debug=True)


if __name__ == '__main__':
    main()
