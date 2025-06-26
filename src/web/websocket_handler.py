"""
WebSocket Handler for Pokemon RL Bot

This module handles real-time communication between the web dashboard
and the Pokemon RL training system using Socket.IO.
"""

import asyncio
import threading
import time
from typing import Dict, Any, Callable, Optional

from flask_socketio import SocketIO
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class WebSocketHandler:
    """Handles WebSocket communications for the Pokemon RL bot."""
    
    def __init__(self, socketio: SocketIO):
        """Initialize the WebSocket handler.
        
        Args:
            socketio: Flask-SocketIO instance
        """
        self.socketio = socketio
        self.connected_clients = set()
        self.frame_rate = 30  # FPS for game streaming
        self.metrics_rate = 5  # Metrics update rate in seconds
        
        # Streaming control
        self.streaming_active = False
        self.metrics_active = False
        
        # Callback functions
        self.frame_callback: Optional[Callable] = None
        self.metrics_callback: Optional[Callable] = None
        
        # Background threads
        self.frame_thread: Optional[threading.Thread] = None
        self.metrics_thread: Optional[threading.Thread] = None
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup Socket.IO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            client_id = self.socketio.request.sid
            self.connected_clients.add(client_id)
            logger.info(f"Client connected: {client_id} (Total: {len(self.connected_clients)})")
            
            # Send initial status
            self._emit_to_client(client_id, 'connection_status', {
                'connected': True,
                'client_id': client_id,
                'server_time': time.time()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            client_id = self.socketio.request.sid
            self.connected_clients.discard(client_id)
            logger.info(f"Client disconnected: {client_id} (Total: {len(self.connected_clients)})")
        
        @self.socketio.on('start_streaming')
        def handle_start_streaming():
            """Start game frame streaming."""
            client_id = self.socketio.request.sid
            logger.info(f"Client {client_id} requested streaming start")
            
            if not self.streaming_active:
                self.start_frame_streaming()
            
            self._emit_to_client(client_id, 'streaming_status', {
                'active': self.streaming_active
            })
        
        @self.socketio.on('stop_streaming')
        def handle_stop_streaming():
            """Stop game frame streaming."""
            client_id = self.socketio.request.sid
            logger.info(f"Client {client_id} requested streaming stop")
            
            # Only stop if no other clients need streaming
            if len(self.connected_clients) <= 1:
                self.stop_frame_streaming()
            
            self._emit_to_client(client_id, 'streaming_status', {
                'active': self.streaming_active
            })
        
        @self.socketio.on('start_metrics')
        def handle_start_metrics():
            """Start metrics broadcasting."""
            client_id = self.socketio.request.sid
            logger.info(f"Client {client_id} requested metrics start")
            
            if not self.metrics_active:
                self.start_metrics_broadcasting()
            
            self._emit_to_client(client_id, 'metrics_status', {
                'active': self.metrics_active
            })
        
        @self.socketio.on('stop_metrics')
        def handle_stop_metrics():
            """Stop metrics broadcasting."""
            client_id = self.socketio.request.sid
            logger.info(f"Client {client_id} requested metrics stop")
            
            # Only stop if no other clients need metrics
            if len(self.connected_clients) <= 1:
                self.stop_metrics_broadcasting()
            
            self._emit_to_client(client_id, 'metrics_status', {
                'active': self.metrics_active
            })
        
        @self.socketio.on('set_frame_rate')
        def handle_set_frame_rate(data):
            """Set streaming frame rate."""
            client_id = self.socketio.request.sid
            new_rate = data.get('rate', 30)
            
            if 1 <= new_rate <= 60:
                self.frame_rate = new_rate
                logger.info(f"Frame rate set to {new_rate} FPS by client {client_id}")
                
                self._emit_to_client(client_id, 'frame_rate_updated', {
                    'rate': self.frame_rate
                })
            else:
                self._emit_to_client(client_id, 'error', {
                    'message': 'Frame rate must be between 1 and 60 FPS'
                })
        
        @self.socketio.on('ping')
        def handle_ping():
            """Handle ping from client."""
            client_id = self.socketio.request.sid
            self._emit_to_client(client_id, 'pong', {
                'timestamp': time.time()
            })
    
    def _emit_to_client(self, client_id: str, event: str, data: Dict[str, Any]):
        """Emit event to specific client.
        
        Args:
            client_id: Client socket ID
            event: Event name
            data: Event data
        """
        try:
            self.socketio.emit(event, data, room=client_id)
        except Exception as e:
            logger.error(f"Error emitting to client {client_id}: {e}")
    
    def _emit_to_all(self, event: str, data: Dict[str, Any]):
        """Emit event to all connected clients.
        
        Args:
            event: Event name
            data: Event data
        """
        try:
            self.socketio.emit(event, data)
        except Exception as e:
            logger.error(f"Error emitting to all clients: {e}")
    
    def set_frame_callback(self, callback: Callable[[], np.ndarray]):
        """Set callback function to get game frames.
        
        Args:
            callback: Function that returns current game frame
        """
        self.frame_callback = callback
    
    def set_metrics_callback(self, callback: Callable[[], Dict[str, Any]]):
        """Set callback function to get training metrics.
        
        Args:
            callback: Function that returns current metrics
        """
        self.metrics_callback = callback
    
    def start_frame_streaming(self):
        """Start streaming game frames to clients."""
        if self.streaming_active:
            return
        
        self.streaming_active = True
        logger.info("Starting frame streaming")
        
        self.frame_thread = threading.Thread(
            target=self._frame_streaming_loop,
            daemon=True
        )
        self.frame_thread.start()
    
    def stop_frame_streaming(self):
        """Stop streaming game frames."""
        self.streaming_active = False
        logger.info("Stopping frame streaming")
        
        if self.frame_thread:
            self.frame_thread.join(timeout=1.0)
    
    def start_metrics_broadcasting(self):
        """Start broadcasting training metrics to clients."""
        if self.metrics_active:
            return
        
        self.metrics_active = True
        logger.info("Starting metrics broadcasting")
        
        self.metrics_thread = threading.Thread(
            target=self._metrics_broadcasting_loop,
            daemon=True
        )
        self.metrics_thread.start()
    
    def stop_metrics_broadcasting(self):
        """Stop broadcasting training metrics."""
        self.metrics_active = False
        logger.info("Stopping metrics broadcasting")
        
        if self.metrics_thread:
            self.metrics_thread.join(timeout=1.0)
    
    def _frame_streaming_loop(self):
        """Main loop for streaming game frames."""
        frame_interval = 1.0 / self.frame_rate
        
        while self.streaming_active and self.connected_clients:
            try:
                if self.frame_callback:
                    frame = self.frame_callback()
                    if frame is not None:
                        self.broadcast_frame(frame)
                
                time.sleep(frame_interval)
                
            except Exception as e:
                logger.error(f"Error in frame streaming loop: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def _metrics_broadcasting_loop(self):
        """Main loop for broadcasting training metrics."""
        while self.metrics_active and self.connected_clients:
            try:
                if self.metrics_callback:
                    metrics = self.metrics_callback()
                    if metrics:
                        self.broadcast_metrics(metrics)
                
                time.sleep(self.metrics_rate)
                
            except Exception as e:
                logger.error(f"Error in metrics broadcasting loop: {e}")
                time.sleep(1.0)  # Longer pause on error
    
    def broadcast_frame(self, frame: np.ndarray):
        """Broadcast game frame to all connected clients.
        
        Args:
            frame: Game frame as numpy array
        """
        import base64
        import cv2
        
        try:
            # Resize frame for web transmission if needed
            height, width = frame.shape[:2]
            if width > 640:  # Limit width for bandwidth
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            self._emit_to_all('frame', {
                'data': frame_b64,
                'timestamp': time.time(),
                'size': [frame.shape[1], frame.shape[0]]  # width, height
            })
            
        except Exception as e:
            logger.error(f"Error broadcasting frame: {e}")
    
    def broadcast_metrics(self, metrics: Dict[str, Any]):
        """Broadcast training metrics to all connected clients.
        
        Args:
            metrics: Training metrics dictionary
        """
        try:
            metrics['timestamp'] = time.time()
            self._emit_to_all('metrics', metrics)
            
        except Exception as e:
            logger.error(f"Error broadcasting metrics: {e}")
    
    def broadcast_event(self, event: str, data: Dict[str, Any]):
        """Broadcast custom event to all connected clients.
        
        Args:
            event: Event name
            data: Event data
        """
        try:
            data['timestamp'] = time.time()
            self._emit_to_all(event, data)
            
        except Exception as e:
            logger.error(f"Error broadcasting event {event}: {e}")
    
    def send_notification(self, message: str, level: str = 'info'):
        """Send notification to all connected clients.
        
        Args:
            message: Notification message
            level: Notification level ('info', 'warning', 'error', 'success')
        """
        self.broadcast_event('notification', {
            'message': message,
            'level': level
        })
    
    def get_client_count(self) -> int:
        """Get number of connected clients.
        
        Returns:
            Number of connected clients
        """
        return len(self.connected_clients)
    
    def is_streaming(self) -> bool:
        """Check if frame streaming is active.
        
        Returns:
            True if streaming is active
        """
        return self.streaming_active
    
    def is_broadcasting_metrics(self) -> bool:
        """Check if metrics broadcasting is active.
        
        Returns:
            True if metrics broadcasting is active
        """
        return self.metrics_active
    
    def shutdown(self):
        """Shutdown the WebSocket handler."""
        logger.info("Shutting down WebSocket handler")
        
        self.stop_frame_streaming()
        self.stop_metrics_broadcasting()
        
        # Notify clients of shutdown
        self._emit_to_all('server_shutdown', {
            'message': 'Server is shutting down'
        })
        
        self.connected_clients.clear()


class StreamingManager:
    """Manages streaming of game frames and metrics."""
    
    def __init__(self, websocket_handler: WebSocketHandler):
        """Initialize the streaming manager.
        
        Args:
            websocket_handler: WebSocket handler instance
        """
        self.websocket_handler = websocket_handler
        self.current_frame: Optional[np.ndarray] = None
        self.current_metrics: Dict[str, Any] = {}
        
        # Setup callbacks
        self.websocket_handler.set_frame_callback(self.get_current_frame)
        self.websocket_handler.set_metrics_callback(self.get_current_metrics)
    
    def update_frame(self, frame: np.ndarray):
        """Update current game frame.
        
        Args:
            frame: New game frame
        """
        self.current_frame = frame.copy() if frame is not None else None
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update current training metrics.
        
        Args:
            metrics: New training metrics
        """
        self.current_metrics = metrics.copy()
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current game frame.
        
        Returns:
            Current game frame or None
        """
        return self.current_frame
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current training metrics.
        
        Returns:
            Current training metrics
        """
        return self.current_metrics
