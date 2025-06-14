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
        
        # Execution tracking
        self.current_layer = 0
        self.total_layers = 0
        self.execution_running = False
        self.layer_progress = {}
        
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
                const totalOps = data.peripherals?.total_crossbar_operations || data.chip.performance?.total_operations || 0;
                document.getElementById('total-operations').innerText = totalOps.toLocaleString();
                document.getElementById('execution-cycles').innerText = 
                    (data.execution?.execution_cycles || 0).toLocaleString();
                
                const energy = (data.chip.energy?.total_energy || 0) * 1000;
                document.getElementById('energy-consumption').innerText = energy.toFixed(2) + ' mJ';
            }
            
            // Update ADC/DAC information
            if(data.peripherals) {
                const adcUtil = (data.peripherals.adc_utilization || 0) * 100;
                const dacUtil = (data.peripherals.dac_utilization || 0) * 100;
                document.getElementById('adc-util').innerText = adcUtil.toFixed(1) + '%';
                document.getElementById('dac-util').innerText = dacUtil.toFixed(1) + '%';
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
        
        # Initialize tracking based on DNN configuration
        if hasattr(execution_engine, 'dnn_manager') and execution_engine.dnn_manager:
            self.total_layers = len(execution_engine.dnn_manager.dnn_config.layers)
        
    def update_execution_progress(self, current_layer: int, layer_progress: float = 0.0):
        """Update the current execution progress"""
        self.current_layer = current_layer
        self.execution_running = True
        self.layer_progress[current_layer] = layer_progress
        
    def set_execution_complete(self):
        """Mark execution as complete"""
        self.execution_running = False
        self.current_layer = self.total_layers
        
    def execute_with_web_monitoring(self, input_data, **kwargs):
        """Execute inference with web monitoring integration"""
        if not self.execution_engine:
            return {"success": False, "error": "No execution engine connected"}
        
        # Start execution tracking
        self.execution_running = True
        self.current_layer = 0
        
        # Patch the execution engine to report progress
        original_execute = self.execution_engine.execute_inference
        
        def monitored_execute(input_data, **kwargs):
            # Store reference to web server in execution engine for progress updates
            self.execution_engine._web_server = self
            
            # Call original execution
            result = original_execute(input_data, **kwargs)
            
            # Mark as complete
            self.set_execution_complete()
            return result
        
        # Execute with monitoring
        return monitored_execute(input_data, **kwargs)
        
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
    
    def _convert_to_json_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # For enum types and other objects, convert to string
            return str(obj)
        elif hasattr(obj, 'name'):
            # For enum types with name attribute
            return obj.name
        else:
            return obj

    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get current statistics from the simulator"""
        if not self.execution_engine:
            return {}
            
        try:
            # Get real chip statistics
            chip_stats_raw = self.execution_engine.chip.get_total_statistics()
            chip_stats = self._convert_to_json_serializable(chip_stats_raw)
            
            # Get actual crossbar data from the chip
            crossbar_data = []
            crossbar_index = 0
            total_crossbar_ops = 0
            total_adc_conversions = 0
            total_dac_conversions = 0
            
            for supertile_idx, supertile in enumerate(self.execution_engine.chip.supertiles):
                for tile_idx, tile in enumerate(supertile.tiles):
                    for xb_idx, crossbar in enumerate(tile.crossbars):
                        crossbar_stats = crossbar.get_statistics()
                        operations = int(crossbar_stats.get('total_operations', 0))
                        total_crossbar_ops += operations
                        
                        # Calculate utilization based on operations
                        max_ops = 1000  # Normalize against reasonable maximum
                        utilization = min(1.0, operations / max_ops) if max_ops > 0 else 0.0
                        
                        crossbar_data.append({
                            'id': f'ST{supertile_idx}_T{tile_idx}_XB{xb_idx}',
                            'utilization': float(utilization),
                            'operations': operations,
                            'energy': float(crossbar_stats.get('total_energy', 0))
                        })
                        crossbar_index += 1
                        
                        if crossbar_index >= 32:  # Limit to 32 for display
                            break
                    if crossbar_index >= 32:
                        break
                if crossbar_index >= 32:
                    break
            
            # Pad with zeros if we have fewer than 32 crossbars
            while len(crossbar_data) < 32:
                crossbar_data.append({
                    'id': f'XB-{len(crossbar_data)}',
                    'utilization': 0.0,
                    'operations': 0,
                    'energy': 0.0
                })
            
            # Get real ADC/DAC conversions from peripherals
            for supertile in self.execution_engine.chip.supertiles:
                for tile in supertile.tiles:
                    if hasattr(tile, 'peripheral_manager'):
                        pm = tile.peripheral_manager
                        total_adc_conversions += sum(adc.conversion_count for adc in pm.output_adcs)
                        total_dac_conversions += sum(dac.conversion_count for dac in pm.input_dacs)
            
            # Get real memory statistics
            if hasattr(self.execution_engine, 'system') and hasattr(self.execution_engine.system, 'buffer_manager'):
                memory_stats_raw = self.execution_engine.system.buffer_manager.get_all_statistics()
                memory_stats = self._convert_to_json_serializable(memory_stats_raw)
            else:
                memory_stats = {}
            
            memory_data = {
                'global_buffer': {
                    'operations': int(memory_stats.get('global_buffer', {}).get('operations', 0)),
                    'memory_stats': {'utilization': memory_stats.get('global_buffer', {}).get('memory_stats', {}).get('utilization', 0)}
                },
                'shared_buffers': {
                    'operations': int(memory_stats.get('shared_buffers', {}).get('operations', 0)),
                    'memory_stats': {'utilization': memory_stats.get('shared_buffers', {}).get('memory_stats', {}).get('utilization', 0)}
                },
                'local_buffers': {
                    'operations': int(memory_stats.get('local_buffers', {}).get('operations', 0)),
                    'memory_stats': {'utilization': memory_stats.get('local_buffers', {}).get('memory_stats', {}).get('utilization', 0)}
                }
            }
            
            # Get current execution progress from the execution engine
            current_layer = self.current_layer
            total_layers = self.total_layers if self.total_layers > 0 else 7
            
            # Try to get more accurate progress from execution engine's layer log
            if hasattr(self.execution_engine, 'layer_execution_log') and self.execution_engine.layer_execution_log:
                completed_layers = len(self.execution_engine.layer_execution_log)
                current_layer = min(completed_layers, total_layers - 1)
                self.current_layer = current_layer
            
            # Calculate overall progress
            if total_layers > 0:
                layer_progress = (current_layer / total_layers) if current_layer < total_layers else 1.0
            else:
                layer_progress = 0.0
            
            # If execution is complete, set progress to 100%
            if not self.execution_running and current_layer >= total_layers - 1:
                layer_progress = 1.0
            
            execution_data = {
                'running': self.execution_running,
                'current_layer': int(current_layer),
                'total_layers': int(total_layers),
                'progress': float(layer_progress),
                'execution_cycles': int(chip_stats.get('performance', {}).get('total_cycles', 0))
            }
            
            # Calculate ADC/DAC utilization
            max_conversions = 100000  # Reasonable maximum for normalization
            adc_utilization = min(1.0, total_adc_conversions / max_conversions) if max_conversions > 0 else 0.0
            dac_utilization = min(1.0, total_dac_conversions / max_conversions) if max_conversions > 0 else 0.0
            
            return {
                'chip': chip_stats,
                'crossbars': crossbar_data,
                'memory': memory_data,
                'execution': execution_data,
                'peripherals': {
                    'adc_utilization': float(adc_utilization),
                    'dac_utilization': float(dac_utilization),
                    'adc_conversions': int(total_adc_conversions),
                    'dac_conversions': int(total_dac_conversions),
                    'total_crossbar_operations': int(total_crossbar_ops)
                }
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            import traceback
            traceback.print_exc()
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