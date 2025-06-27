#!/usr/bin/env python3
"""
Pokemon RL Bot - Cloud Dashboard (Linux Headless)
Optimized for cloud deployment with virtual display and headless operation
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

# Simple logging setup for cloud environment
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

class CloudDashboard:
    def __init__(self):
        # Get project root path
        self.project_root = Path(__file__).parent.parent.parent
        
        self.app = Flask(__name__, 
                        template_folder=str(self.project_root / 'src/web/templates'),
                        static_folder=str(self.project_root / 'src/web/static'))
        self.app.config['SECRET_KEY'] = 'pokemon-rl-bot-cloud'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        try:
            self.logger = setup_logger("cloud_dashboard")
        except:
            self.logger = logger
            
        try:
            self.metrics = MetricsCollector()
        except:
            self.metrics = MockMetrics()
        
        # Cloud paths
        self.rom_path = self.project_root / "roms" / "pokemon_leaf_green.gba"
        self.logs_dir = self.project_root / "logs"
        
        # Process tracking
        self.emulator_process = None
        self.bot_process = None
        
        # Screenshot thread
        self.screenshot_thread = None
        self.screenshot_running = False
        
        # Virtual display settings
        self.display = os.environ.get('DISPLAY', ':99')
        
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
                'display_available': self.check_display(),
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
            
        @self.app.route('/api/test_display')
        def test_display():
            success, message = self.test_virtual_display()
            return jsonify({'success': success, 'message': message})
            
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
            self.logger.info("Client connected to cloud dashboard")
            emit('status_update', self.get_status_data())
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info("Client disconnected from cloud dashboard")
            
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
            
    def check_display(self):
        """Check if virtual display is available"""
        try:
            env = os.environ.copy()
            env['DISPLAY'] = self.display
            result = subprocess.run(['xdpyinfo'], 
                                  capture_output=True, 
                                  env=env, 
                                  timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
            
    def test_virtual_display(self):
        """Test virtual display functionality"""
        try:
            if not self.check_display():
                return False, f"Virtual display {self.display} not available"
                
            # Test xwininfo
            env = os.environ.copy()
            env['DISPLAY'] = self.display
            result = subprocess.run(['xwininfo', '-root'], 
                                  capture_output=True, 
                                  env=env, 
                                  timeout=10)
            
            if result.returncode == 0:
                return True, f"Virtual display {self.display} is working"
            else:
                return False, f"Virtual display test failed: {result.stderr.decode()}"
                
        except Exception as e:
            return False, f"Display test error: {e}"
            
    def start_emulator(self):
        """Start the VBA-M emulator in headless mode"""
        try:
            if not self.rom_path.exists():
                return False, f"ROM file not found: {self.rom_path}"
                
            if not self.check_display():
                return False, f"Virtual display {self.display} not available"
                
            if self.is_emulator_running():
                return False, "Emulator is already running"
                
            # Start VBA-M with headless flags
            env = os.environ.copy()
            env['DISPLAY'] = self.display
            
            cmd = [
                'vbam',
                '--no-opengl',
                '--frameskip=0',
                '--no-show-speed',
                str(self.rom_path)
            ]
            
            self.emulator_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Give it a moment to start
            time.sleep(3)
            
            if self.is_emulator_running():
                logger.info("Cloud emulator started successfully")
                self.start_screenshot_capture()
                return True, "Emulator started successfully"
            else:
                return False, "Emulator failed to start"
                
        except Exception as e:
            logger.error(f"Error starting emulator: {e}")
            return False, f"Error starting emulator: {e}"
            
    def stop_emulator(self):
        """Stop the VBA-M emulator"""
        try:
            self.stop_screenshot_capture()
            
            # Kill VBA-M processes
            subprocess.run(['pkill', '-f', 'vbam'], capture_output=True)
            
            if self.emulator_process:
                try:
                    self.emulator_process.terminate()
                    self.emulator_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.emulator_process.kill()
                self.emulator_process = None
                
            logger.info("Cloud emulator stopped")
            return True, "Emulator stopped successfully"
            
        except Exception as e:
            logger.error(f"Error stopping emulator: {e}")
            return False, f"Error stopping emulator: {e}"
            
    def start_bot(self):
        """Start the RL training bot"""
        try:
            if not self.is_emulator_running():
                return False, "Start emulator first"
                
            if self.is_bot_running():
                return False, "Bot is already running"
                
            # Start the training script
            env = os.environ.copy()
            env['DISPLAY'] = self.display
            
            cmd = [
                sys.executable,
                'scripts/train.py',
                '--config', 'config.json'
            ]
            
            self.bot_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root,
                preexec_fn=os.setsid
            )
            
            logger.info("Cloud bot started successfully")
            return True, "Bot started successfully"
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            return False, f"Error starting bot: {e}"
            
    def stop_bot(self):
        """Stop the RL training bot"""
        try:
            # Kill training processes
            subprocess.run(['pkill', '-f', 'train.py'], capture_output=True)
            
            if self.bot_process:
                try:
                    self.bot_process.terminate()
                    self.bot_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.bot_process.kill()
                self.bot_process = None
                
            logger.info("Cloud bot stopped")
            return True, "Bot stopped successfully"
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            return False, f"Error stopping bot: {e}"
            
    def capture_screenshot(self):
        """Capture screenshot using scrot (Linux headless)"""
        try:
            if not self.check_display():
                return None
                
            env = os.environ.copy()
            env['DISPLAY'] = self.display
            
            # Use scrot with virtual display
            result = subprocess.run([
                'scrot', '-z', '-'
            ], capture_output=True, env=env, timeout=10)
            
            if result.returncode == 0 and result.stdout:
                return base64.b64encode(result.stdout).decode('utf-8')
            else:
                logger.error(f"Screenshot failed: {result.stderr.decode()}")
                
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            
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
        """Screenshot capture loop for cloud"""
        while self.screenshot_running:
            try:
                screenshot = self.capture_screenshot()
                if screenshot:
                    self.socketio.emit('frame', {'data': screenshot})
                time.sleep(1.0)  # 1 FPS for cloud to reduce bandwidth
            except Exception as e:
                self.logger.error(f"Screenshot loop error: {e}")
                time.sleep(2)
                
    def get_system_info(self):
        """Get cloud system information"""
        info = {
            'platform': 'linux-cloud',
            'display': self.display,
            'display_available': self.check_display()
        }
        
        # Try to get system info
        try:
            # CPU info
            with open('/proc/cpuinfo', 'r') as f:
                cpu_lines = [line for line in f if 'model name' in line]
                if cpu_lines:
                    info['cpu'] = cpu_lines[0].split(':')[1].strip()
                    
            # Memory info
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        info['memory_total'] = line.split()[1] + ' kB'
                    elif line.startswith('MemAvailable:'):
                        info['memory_available'] = line.split()[1] + ' kB'
                        
            # Disk info
            result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    info['disk_usage'] = lines[1].split()[4]  # Usage percentage
                    
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            
        return info
        
    def get_metrics(self):
        """Get training metrics"""
        try:
            metrics_file = self.logs_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            
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
            log_files = ['training.log', 'agent.log', 'environment.log', 'dashboard.log']
            for log_file in log_files:
                log_path = self.logs_dir / log_file
                if log_path.exists():
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        logs.extend([
                            {'file': log_file, 'message': line.strip()}
                            for line in lines[-30:]  # Last 30 lines per file
                            if line.strip()
                        ])
        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            
        return logs[-100:]  # Return last 100 log entries total
        
    def get_status_data(self):
        """Get complete status data"""
        return {
            'emulator_running': self.is_emulator_running(),
            'bot_running': self.is_bot_running(),
            'rom_exists': self.rom_path.exists(),
            'display_available': self.check_display(),
            'system_info': self.get_system_info(),
            'metrics': self.get_metrics()
        }
        
    def run(self, host='0.0.0.0', port=7500, debug=False):
        """Run the cloud dashboard"""
        logger.info(f"Starting Pokemon RL Bot Cloud Dashboard on {host}:{port}")
        logger.info(f"Virtual display: {self.display}")
        
        # Handle shutdown gracefully
        def signal_handler(sig, frame):
            logger.info("Shutting down cloud dashboard...")
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
            logger.info("Dashboard stopped by user")
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
        finally:
            self.stop_screenshot_capture()


def main():
    """Main entry point for cloud dashboard"""
    dashboard = CloudDashboard()
    
    print("🎮 Pokemon RL Bot - Cloud Dashboard")
    print("===================================")
    print(f"📱 Dashboard: http://0.0.0.0:7500")
    print(f"🖥️  Virtual Display: {dashboard.display}")
    print("☁️  Optimized for cloud deployment")
    print("⭐️ Features: Headless Operation, Virtual Display, Remote Access")
    print()
    
    # Test display before starting
    success, message = dashboard.test_virtual_display()
    if success:
        print(f"✅ {message}")
    else:
        print(f"⚠️  {message}")
    print()
    
    dashboard.run(debug=False)


if __name__ == '__main__':
    main()
