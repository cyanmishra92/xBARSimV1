#!/usr/bin/env python3
"""
Basic web-based visualization server for ReRAM Crossbar Simulator
Provides real-time monitoring and educational visualization capabilities
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import asdict

try:
    from flask import Flask, render_template_string, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    SocketIO = None

import numpy as np

# Import simulator components using absolute imports for WSL compatibility
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.execution_engine import ExecutionEngine
from src.core.hierarchy import ReRAMChip
from src.core.dnn_manager import DNNManager


class WebVisualizationServer:
    """
    Basic web-based visualization server for real-time monitoring
    """
    
    def __init__(self, port: int = 8080):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask dependencies not available. Install with: pip install flask flask-socketio")
        
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'reram_simulator_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.execution_engine = None
        self.monitoring = False
        self.stats_thread = None
        
        self.setup_routes()
        self.setup_socketio_events()
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>ReRAM Crossbar Simulator - Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: white; }
        .header { text-align: center; margin-bottom: 30px; }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: #2d2d2d; padding: 20px; border-radius: 8px; border: 1px solid #444; }
        .status { text-align: center; margin: 20px 0; }
        .connected { color: #4CAF50; }
        .disconnected { color: #f44336; }
        .crossbar-grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 5px; margin: 20px 0; }
        .crossbar-cell { width: 30px; height: 30px; border: 1px solid #666; border-radius: 4px; 
                        display: flex; align-items: center; justify-content: center; font-size: 10px; }
        .crossbar-idle { background: #333; }
        .crossbar-low { background: #4CAF50; }
        .crossbar-medium { background: #FF9800; }
        .crossbar-high { background: #f44336; }
        .metric { margin: 10px 0; }
        .memory-bar { background: #333; height: 20px; border-radius: 10px; overflow: hidden; }
        .memory-fill { background: linear-gradient(90deg, #4CAF50, #8BC34A); height: 100%; transition: width 0.3s; }
        button { background: #4CAF50; color: white; border: none; padding: 10px 20px; 
                border-radius: 4px; cursor: pointer; margin: 5px; }
        button:disabled { background: #666; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 ReRAM Crossbar Simulator - Real-Time Dashboard</h1>
        <div id="status" class="status disconnected">Disconnected</div>
        <button id="startBtn" onclick="startMonitoring()">Start Monitoring</button>
        <button id="stopBtn" onclick="stopMonitoring()" disabled>Stop Monitoring</button>
    </div>
    
    <div class="container">
        <div class="panel">
            <h3>🔥 Crossbar Activity</h3>
            <div id="crossbar-grid" class="crossbar-grid">
                <!-- 32 crossbars (4x8 grid) -->
                <script>
                    for(let i = 0; i < 32; i++) {
                        document.write('<div class="crossbar-cell crossbar-idle" id="crossbar-' + i + '">' + i + '</div>');
                    }
                </script>
            </div>
        </div>
        
        <div class="panel">
            <h3>📊 Performance Metrics</h3>
            <div class="metric">Operations: <span id="total-operations">0</span></div>
            <div class="metric">Execution Cycles: <span id="execution-cycles">0</span></div>
            <div class="metric">Energy: <span id="energy-consumption">0 mJ</span></div>
            <div class="metric">Throughput: <span id="throughput">0 ops/s</span></div>
        </div>
        
        <div class="panel">
            <h3>💾 Memory Utilization</h3>
            <div class="metric">
                Global Buffer: <span id="global-util">0%</span>
                <div class="memory-bar"><div id="global-fill" class="memory-fill" style="width: 0%"></div></div>
            </div>
            <div class="metric">
                Shared Buffers: <span id="shared-util">0%</span>
                <div class="memory-bar"><div id="shared-fill" class="memory-fill" style="width: 0%"></div></div>
            </div>
            <div class="metric">
                Local Buffers: <span id="local-util">0%</span>
                <div class="memory-bar"><div id="local-fill" class="memory-fill" style="width: 0%"></div></div>
            </div>
        </div>
        
        <div class="panel">
            <h3>⚙️ System Status</h3>
            <div class="metric">Current Layer: <span id="current-layer">Waiting...</span></div>
            <div class="metric">Progress: <span id="progress">0%</span></div>
            <div class="metric">ADC Utilization: <span id="adc-util">0%</span></div>
            <div class="metric">DAC Utilization: <span id="dac-util">0%</span></div>
        </div>
    </div>

    <script>
        const socket = io();
        
        socket.on('connect', function() {
            document.getElementById('status').innerText = 'Connected';
            document.getElementById('status').className = 'status connected';
            document.getElementById('startBtn').disabled = false;
        });
        
        socket.on('disconnect', function() {
            document.getElementById('status').innerText = 'Disconnected';
            document.getElementById('status').className = 'status disconnected';
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = true;
        });
        
        socket.on('stats_update', function(data) {
            console.log('Stats update:', data);
            
            // Update performance metrics
            if(data.chip) {
                document.getElementById('total-operations').innerText = 
                    (data.chip.performance?.total_operations || 0).toLocaleString();
                document.getElementById('execution-cycles').innerText = 
                    (data.chip.execution_cycles || 0).toLocaleString();
                
                const energy = (data.chip.energy?.total_energy || 0) * 1000;
                document.getElementById('energy-consumption').innerText = energy.toFixed(2) + ' mJ';
            }
            
            // Update crossbar activity
            if(data.crossbars) {
                data.crossbars.forEach((crossbar, index) => {
                    const cell = document.getElementById('crossbar-' + index);
                    if(cell) {
                        cell.className = 'crossbar-cell ';
                        const util = crossbar.utilization || 0;
                        if(util > 0.7) cell.className += 'crossbar-high';
                        else if(util > 0.3) cell.className += 'crossbar-medium';
                        else if(util > 0.1) cell.className += 'crossbar-low';
                        else cell.className += 'crossbar-idle';
                    }
                });
            }
            
            // Update memory utilization
            if(data.memory) {
                const globalUtil = getMemoryUtil(data.memory, 'global_buffer');
                const sharedUtil = getMemoryUtil(data.memory, 'shared_buffers');
                const localUtil = getMemoryUtil(data.memory, 'local_buffers');
                
                document.getElementById('global-util').innerText = (globalUtil * 100).toFixed(1) + '%';
                document.getElementById('global-fill').style.width = (globalUtil * 100) + '%';
                
                document.getElementById('shared-util').innerText = (sharedUtil * 100).toFixed(1) + '%';
                document.getElementById('shared-fill').style.width = (sharedUtil * 100) + '%';
                
                document.getElementById('local-util').innerText = (localUtil * 100).toFixed(1) + '%';
                document.getElementById('local-fill').style.width = (localUtil * 100) + '%';
            }
            
            // Update execution progress
            if(data.execution) {
                document.getElementById('current-layer').innerText = 
                    'Layer ' + ((data.execution.current_layer || 0) + 1) + ' of ' + (data.execution.total_layers || 1);
                document.getElementById('progress').innerText = 
                    ((data.execution.progress || 0) * 100).toFixed(1) + '%';
            }
        });
        
        socket.on('monitoring_started', function(data) {
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
        });
        
        socket.on('monitoring_stopped', function(data) {
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        });
        
        function getMemoryUtil(memData, bufferType) {
            if(!memData || !memData[bufferType]) return 0;
            const bufferStats = memData[bufferType];
            if(bufferStats.memory_stats && bufferStats.memory_stats.utilization !== undefined) {
                return Math.min(1.0, bufferStats.memory_stats.utilization);
            }
            const operations = bufferStats.operations || 0;
            return Math.min(1.0, operations / 100);
        }
        
        function startMonitoring() {
            socket.emit('start_monitoring');
        }
        
        function stopMonitoring() {
            socket.emit('stop_monitoring');
        }
    </script>
</body>
</html>
            """)
        
        @self.app.route('/educational')
        def educational():
            """Educational tool page"""
            return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>ReRAM Simulator - Educational Tool</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: white; }
        .header { text-align: center; margin-bottom: 30px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .panel { background: #2d2d2d; padding: 20px; border-radius: 8px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎓 CNN to CrossBar Mapping - Educational Tool</h1>
        <p>Interactive learning platform for understanding neural network hardware mapping</p>
        <a href="/" style="color: #4CAF50;">← Back to Dashboard</a>
    </div>
    
    <div class="container">
        <div class="panel">
            <h3>📚 Tutorial Steps</h3>
            <ol>
                <li><strong>CNN Architecture Analysis</strong> - Understanding layer structure</li>
                <li><strong>Weight Reshaping</strong> - Converting 4D tensors to 2D matrices</li>
                <li><strong>Crossbar Allocation</strong> - Mapping weights to hardware</li>
                <li><strong>Data Flow Visualization</strong> - Following execution paths</li>
                <li><strong>Live Execution Monitoring</strong> - Real-time performance observation</li>
            </ol>
        </div>
        
        <div class="panel">
            <h3>🧠 CNN Layer Mapping</h3>
            <p>This educational tool helps you understand how Convolutional Neural Network layers 
            are mapped to ReRAM crossbar arrays for efficient hardware acceleration.</p>
            
            <p><strong>Key Concepts:</strong></p>
            <ul>
                <li>4D weight tensors are reshaped into 2D matrices for crossbar storage</li>
                <li>Each crossbar can store a portion of the neural network weights</li>
                <li>Analog computation performs matrix-vector multiplication efficiently</li>
                <li>ADCs and DACs handle analog-to-digital conversions</li>
            </ul>
        </div>
        
        <div class="panel">
            <h3>⚡ Real-time Learning</h3>
            <p>Return to the <a href="/" style="color: #4CAF50;">main dashboard</a> to see 
            live visualization of how your CNN model executes on the ReRAM hardware.</p>
        </div>
    </div>
</body>
</html>
            """)
    
    def setup_socketio_events(self):
        """Setup SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            emit('connected', {'message': 'Connected to ReRAM Simulator'})
            
        @self.socketio.on('start_monitoring')
        def handle_start_monitoring():
            self.start_monitoring()
            emit('monitoring_started', {'status': 'monitoring_started'})
            
        @self.socketio.on('stop_monitoring')
        def handle_stop_monitoring():
            self.stop_monitoring()
            emit('monitoring_stopped', {'status': 'monitoring_stopped'})
    
    def connect_execution_engine(self, execution_engine: ExecutionEngine):
        """Connect to an execution engine for monitoring"""
        self.execution_engine = execution_engine
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        if self.stats_thread is None or not self.stats_thread.is_alive():
            self.stats_thread = threading.Thread(target=self._monitoring_loop)
            self.stats_thread.daemon = True
            self.stats_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring = False
        
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                if self.execution_engine:
                    stats = self.get_real_time_stats()
                    self.socketio.emit('stats_update', stats)
                time.sleep(1)  # Update every second
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1)
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get current statistics from the simulator"""
        if not self.execution_engine:
            return {}
            
        try:
            # Get chip statistics
            chip_stats = self.execution_engine.chip.get_total_statistics()
            
            # Create mock crossbar data (32 crossbars)
            crossbar_data = []
            for i in range(32):
                crossbar_data.append({
                    'id': f'XB-{i}',
                    'utilization': np.random.random() * 0.8,  # Mock utilization
                    'operations': int(np.random.random() * 1000),
                    'energy': np.random.random() * 0.01
                })
            
            # Create mock memory data
            memory_data = {
                'global_buffer': {
                    'operations': chip_stats.get('memory', {}).get('global_buffer', {}).get('operations', 0),
                    'memory_stats': {'utilization': np.random.random() * 0.3}
                },
                'shared_buffers': {
                    'operations': 0,
                    'memory_stats': {'utilization': np.random.random() * 0.2}
                },
                'local_buffers': {
                    'operations': chip_stats.get('memory', {}).get('local_buffers', {}).get('operations', 0),
                    'memory_stats': {'utilization': np.random.random() * 0.5}
                }
            }
            
            # Create execution progress data
            execution_data = {
                'running': True,
                'current_layer': 0,
                'total_layers': 7,  # LeNet has 7 layers
                'progress': np.random.random(),
                'execution_cycles': chip_stats.get('performance', {}).get('total_cycles', 0)
            }
            
            return {
                'chip': chip_stats,
                'crossbars': crossbar_data,
                'memory': memory_data,
                'execution': execution_data,
                'peripherals': {
                    'adc_utilization': np.random.random() * 0.6,
                    'dac_utilization': np.random.random() * 0.6
                }
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    def run(self, debug=False, host='0.0.0.0'):
        """Run the web server"""
        print(f"Starting web visualization server on http://localhost:{self.port}")
        self.socketio.run(self.app, host=host, port=self.port, debug=debug, allow_unsafe_werkzeug=True)


def create_web_visualization_server(port: int = 8080) -> WebVisualizationServer:
    """Create and return a web visualization server instance"""
    if not FLASK_AVAILABLE:
        raise ImportError("Flask dependencies not available. Install with: pip install flask flask-socketio")
    
    return WebVisualizationServer(port=port)