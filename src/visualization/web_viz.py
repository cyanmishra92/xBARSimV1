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
        
        @self.app.route('/architecture')
        def architecture():
            """Detailed architecture visualization page"""
            return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>ReRAM Architecture - Detailed View</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0a0e1a; color: white; overflow-x: hidden; }
        
        .arch-header { text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .arch-header h1 { font-size: 32px; margin-bottom: 10px; }
        .arch-nav { display: flex; justify-content: center; gap: 20px; margin-top: 15px; }
        .arch-nav a { color: white; text-decoration: none; padding: 8px 16px; background: rgba(255,255,255,0.2); border-radius: 20px; transition: all 0.3s; }
        .arch-nav a:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); }
        
        .arch-container { display: grid; grid-template-columns: 300px 1fr 300px; height: calc(100vh - 120px); gap: 15px; padding: 15px; }
        
        .arch-sidebar { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; overflow-y: auto; backdrop-filter: blur(10px); }
        .arch-main { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; backdrop-filter: blur(10px); }
        
        .component-tree { list-style: none; }
        .component-item { padding: 8px; margin: 4px 0; border-radius: 6px; cursor: pointer; transition: all 0.3s; border-left: 3px solid transparent; }
        .component-item:hover { background: rgba(255,255,255,0.1); border-left-color: #00d4aa; }
        .component-item.active { background: rgba(0,212,170,0.2); border-left-color: #00d4aa; }
        .component-item.chip { color: #ff6b6b; font-weight: bold; }
        .component-item.supertile { color: #4ecdc4; margin-left: 15px; }
        .component-item.tile { color: #45b7d1; margin-left: 30px; }
        .component-item.crossbar { color: #96ceb4; margin-left: 45px; }
        .component-item.memory { color: #ffd93d; margin-left: 45px; }
        .component-item.peripheral { color: #a29bfe; margin-left: 45px; }
        
        .detail-panel { background: rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; margin-bottom: 15px; }
        .detail-title { color: #00d4aa; font-size: 16px; font-weight: bold; margin-bottom: 10px; }
        .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 14px; }
        .detail-item { padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .detail-label { color: #b0b8c1; }
        .detail-value { color: white; font-weight: 500; }
        
        .arch-3d-container { height: 400px; border: 2px solid rgba(255,255,255,0.1); border-radius: 8px; margin-bottom: 20px; position: relative; }
        .arch-controls { display: flex; justify-content: center; gap: 10px; margin-bottom: 15px; }
        .arch-btn { padding: 8px 16px; background: rgba(0,212,170,0.2); border: 1px solid #00d4aa; color: #00d4aa; border-radius: 6px; cursor: pointer; transition: all 0.3s; }
        .arch-btn:hover { background: rgba(0,212,170,0.3); transform: translateY(-1px); }
        .arch-btn.active { background: #00d4aa; color: #0a0e1a; }
        
        .dataflow-diagram { background: rgba(255,255,255,0.05); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .dataflow-svg { width: 100%; height: 300px; }
        
        .interconnect-diagram { background: rgba(255,255,255,0.05); border-radius: 8px; padding: 20px; }
        .interconnect-svg { width: 100%; height: 250px; }
        
        .legend { display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px; }
        .legend-item { display: flex; align-items: center; gap: 8px; }
        .legend-color { width: 16px; height: 16px; border-radius: 3px; }
        
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 15px; }
        .metric-card { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #00d4aa; }
        .metric-label { font-size: 12px; color: #b0b8c1; margin-top: 5px; }
        
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .status-active { background: #00d4aa; }
        .status-idle { background: #666; }
        .status-busy { background: #ff6b6b; }
    </style>
</head>
<body>
    <div class="arch-header">
        <h1>🏗️ ReRAM Crossbar Architecture - Deep Dive</h1>
        <p>Comprehensive microarchitectural visualization of hierarchical ReRAM system</p>
        <div class="arch-nav">
            <a href="/">🏠 Dashboard</a>
            <a href="/educational">🎓 Educational</a>
            <a href="/architecture">🏗️ Architecture</a>
        </div>
    </div>
    
    <div class="arch-container">
        <!-- Left Sidebar: Component Tree -->
        <div class="arch-sidebar">
            <h3 style="color: #00d4aa; margin-bottom: 20px;">🔍 Architecture Explorer</h3>
            
            <div class="detail-panel">
                <div class="detail-title">📊 Real-Time Status</div>
                <div id="connection-status">
                    <span class="status-indicator status-idle"></span>Disconnected
                </div>
                <div style="margin-top: 10px;">
                    <button id="arch-start-btn" onclick="startArchitectureMonitoring()" style="background: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-right: 5px;">Start Monitor</button>
                    <button id="arch-stop-btn" onclick="stopArchitectureMonitoring()" style="background: #f44336; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;" disabled>Stop Monitor</button>
                </div>
            </div>
            
            <ul class="component-tree" id="component-tree">
                <li class="component-item chip active" data-component="chip">
                    🔲 ReRAM Chip
                    <ul>
                        <li class="component-item supertile" data-component="supertile">
                            🏢 SuperTile Array (2x)
                            <ul>
                                <li class="component-item tile" data-component="tile">🏠 Processing Tiles (4x each)</li>
                                <li class="component-item memory" data-component="shared_memory">💾 Shared eDRAM Buffer</li>
                            </ul>
                        </li>
                        <li class="component-item memory" data-component="global_memory">🗄️ Global DRAM Buffer</li>
                        <li class="component-item peripheral" data-component="microcontroller">🖥️ Microcontroller Unit</li>
                        <li class="component-item peripheral" data-component="interconnect">🔗 Interconnect Network</li>
                    </ul>
                </li>
            </ul>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="total-crossbars">32</div>
                    <div class="metric-label">Total Crossbars</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="total-cells">524K</div>
                    <div class="metric-label">ReRAM Cells</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="active-tiles">0</div>
                    <div class="metric-label">Active Tiles</div>
                </div>
            </div>
        </div>
        
        <!-- Main Content: Detailed Visualization -->
        <div class="arch-main">
            <div class="arch-controls">
                <button class="arch-btn active" data-view="hierarchy">🏗️ Hierarchy</button>
                <button class="arch-btn" data-view="dataflow">🌊 Data Flow</button>
                <button class="arch-btn" data-view="interconnect">🔗 Interconnect</button>
                <button class="arch-btn" data-view="memory">💾 Memory</button>
                <button class="arch-btn" data-view="3d">🎮 3D View</button>
            </div>
            
            <!-- 3D Architecture Visualization -->
            <div class="arch-3d-container" id="arch-3d-container">
                <div id="arch-3d-canvas"></div>
            </div>
            
            <!-- Data Flow Diagram -->
            <div class="dataflow-diagram" id="dataflow-view" style="display: none;">
                <h3 style="color: #00d4aa; margin-bottom: 15px;">🌊 Neural Network Data Flow</h3>
                <svg class="dataflow-svg" id="dataflow-svg"></svg>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ff6b6b;"></div>
                        <span>Input Data Path</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #4ecdc4;"></div>
                        <span>Weight Data Path</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ffd93d;"></div>
                        <span>Output Data Path</span>
                    </div>
                </div>
            </div>
            
            <!-- Interconnect Network -->
            <div class="interconnect-diagram" id="interconnect-view" style="display: none;">
                <h3 style="color: #00d4aa; margin-bottom: 15px;">🔗 Interconnect Network Topology</h3>
                <svg class="interconnect-svg" id="interconnect-svg"></svg>
            </div>
        </div>
        
        <!-- Right Sidebar: Component Details -->
        <div class="arch-sidebar">
            <h3 style="color: #00d4aa; margin-bottom: 20px;">📋 Component Details</h3>
            
            <div class="detail-panel" id="selected-component">
                <div class="detail-title">🔲 ReRAM Chip Overview</div>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="detail-label">Architecture:</div>
                        <div class="detail-value">Hierarchical</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">SuperTiles:</div>
                        <div class="detail-value">2</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Total Tiles:</div>
                        <div class="detail-value">8</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Total Crossbars:</div>
                        <div class="detail-value">32</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Crossbar Size:</div>
                        <div class="detail-value">128×128</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Total Cells:</div>
                        <div class="detail-value">524,288</div>
                    </div>
                </div>
            </div>
            
            <div class="detail-panel" id="performance-details">
                <div class="detail-title">⚡ Performance Metrics</div>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="detail-label">Operations:</div>
                        <div class="detail-value" id="perf-operations">0</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Energy:</div>
                        <div class="detail-value" id="perf-energy">0 mJ</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Utilization:</div>
                        <div class="detail-value" id="perf-utilization">0%</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Throughput:</div>
                        <div class="detail-value" id="perf-throughput">0 ops/s</div>
                    </div>
                </div>
            </div>
            
            <div class="detail-panel" id="memory-hierarchy">
                <div class="detail-title">💾 Memory Hierarchy</div>
                <div style="margin-bottom: 10px;">
                    <div style="color: #b0b8c1; font-size: 12px;">Global DRAM Buffer</div>
                    <div style="background: #333; height: 15px; border-radius: 8px; overflow: hidden; margin: 5px 0;">
                        <div id="global-memory-bar" style="background: linear-gradient(90deg, #ff6b6b, #ff8787); height: 100%; width: 0%; transition: width 0.3s;"></div>
                    </div>
                    <div style="font-size: 12px; color: #666;" id="global-memory-text">4096 KB | 0% utilized</div>
                </div>
                
                <div style="margin-bottom: 10px;">
                    <div style="color: #b0b8c1; font-size: 12px;">Shared eDRAM Buffers</div>
                    <div style="background: #333; height: 15px; border-radius: 8px; overflow: hidden; margin: 5px 0;">
                        <div id="shared-memory-bar" style="background: linear-gradient(90deg, #4ecdc4, #44a08d); height: 100%; width: 0%; transition: width 0.3s;"></div>
                    </div>
                    <div style="font-size: 12px; color: #666;" id="shared-memory-text">512 KB × 2 | 0% utilized</div>
                </div>
                
                <div>
                    <div style="color: #b0b8c1; font-size: 12px;">Local SRAM Buffers</div>
                    <div style="background: #333; height: 15px; border-radius: 8px; overflow: hidden; margin: 5px 0;">
                        <div id="local-memory-bar" style="background: linear-gradient(90deg, #ffd93d, #f39c12); height: 100%; width: 0%; transition: width 0.3s;"></div>
                    </div>
                    <div style="font-size: 12px; color: #666;" id="local-memory-text">64 KB × 8 | 0% utilized</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let socket = io();
        let selectedComponent = 'chip';
        let currentView = 'hierarchy';
        let scene, camera, renderer, chipGroup;
        
        // Initialize 3D visualization
        function init3DView() {
            const container = document.getElementById('arch-3d-canvas');
            
            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0e1a);
            
            // Camera
            camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(15, 12, 15);
            camera.lookAt(0, 0, 0);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            container.appendChild(renderer.domElement);
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(20, 20, 10);
            directionalLight.castShadow = true;
            scene.add(directionalLight);
            
            // Create detailed chip architecture
            createDetailedChip();
            
            // Animation loop
            animate();
        }
        
        function createDetailedChip() {
            chipGroup = new THREE.Group();
            
            // Create base chip substrate
            const chipGeometry = new THREE.BoxGeometry(20, 0.5, 15);
            const chipMaterial = new THREE.MeshLambertMaterial({ color: 0x2d3436, transparent: true, opacity: 0.8 });
            const chipBase = new THREE.Mesh(chipGeometry, chipMaterial);
            chipBase.position.y = -0.25;
            chipGroup.add(chipBase);
            
            // Create SuperTiles
            for (let st = 0; st < 2; st++) {
                const stGroup = new THREE.Group();
                stGroup.position.x = (st - 0.5) * 10;
                
                // SuperTile substrate
                const stGeometry = new THREE.BoxGeometry(8, 0.3, 12);
                const stMaterial = new THREE.MeshLambertMaterial({ color: 0x636e72, transparent: true, opacity: 0.9 });
                const stMesh = new THREE.Mesh(stGeometry, stMaterial);
                stMesh.position.y = 0.15;
                stGroup.add(stMesh);
                
                // Shared eDRAM buffer for SuperTile
                const edramGeometry = new THREE.BoxGeometry(7, 0.8, 1.5);
                const edramMaterial = new THREE.MeshLambertMaterial({ color: 0x4ecdc4, transparent: true, opacity: 0.7 });
                const edramMesh = new THREE.Mesh(edramGeometry, edramMaterial);
                edramMesh.position.set(0, 0.6, -4.5);
                stGroup.add(edramMesh);
                
                // Create Tiles within SuperTile
                for (let t = 0; t < 4; t++) {
                    const tileGroup = new THREE.Group();
                    const tileX = (t % 2) * 4 - 2;
                    const tileZ = Math.floor(t / 2) * 3 - 1.5;
                    tileGroup.position.set(tileX, 0.5, tileZ);
                    
                    // Tile substrate
                    const tileGeometry = new THREE.BoxGeometry(3.5, 0.2, 2.5);
                    const tileMaterial = new THREE.MeshLambertMaterial({ color: 0x74b9ff, transparent: true, opacity: 0.8 });
                    const tileMesh = new THREE.Mesh(tileGeometry, tileMaterial);
                    tileGroup.add(tileMesh);
                    
                    // Local SRAM buffer for Tile
                    const sramGeometry = new THREE.BoxGeometry(3, 0.4, 0.4);
                    const sramMaterial = new THREE.MeshLambertMaterial({ color: 0xffd93d, transparent: true, opacity: 0.8 });
                    const sramMesh = new THREE.Mesh(sramGeometry, sramMaterial);
                    sramMesh.position.set(0, 0.3, -1);
                    tileGroup.add(sramMesh);
                    
                    // Create Crossbars within Tile
                    for (let cb = 0; cb < 4; cb++) {
                        const crossbarGeometry = new THREE.BoxGeometry(0.7, 0.15, 0.7);
                        const crossbarMaterial = new THREE.MeshLambertMaterial({ color: 0x96ceb4, transparent: true, opacity: 0.9 });
                        const crossbarMesh = new THREE.Mesh(crossbarGeometry, crossbarMaterial);
                        
                        const cbX = (cb % 2) * 1.5 - 0.75;
                        const cbZ = Math.floor(cb / 2) * 1.2 - 0.6;
                        crossbarMesh.position.set(cbX, 0.25, cbZ);
                        
                        // Add peripheral components around crossbar
                        // ADCs
                        const adcGeometry = new THREE.CylinderGeometry(0.05, 0.05, 0.3);
                        const adcMaterial = new THREE.MeshLambertMaterial({ color: 0xa29bfe });
                        for (let i = 0; i < 4; i++) {
                            const adc = new THREE.Mesh(adcGeometry, adcMaterial);
                            const angle = (i * Math.PI) / 2;
                            adc.position.set(
                                cbX + Math.cos(angle) * 0.5,
                                0.4,
                                cbZ + Math.sin(angle) * 0.5
                            );
                            tileGroup.add(adc);
                        }
                        
                        crossbarMesh.userData = {
                            type: 'crossbar',
                            supertile: st,
                            tile: t,
                            crossbar: cb,
                            id: `ST${st}_T${t}_XB${cb}`
                        };
                        
                        tileGroup.add(crossbarMesh);
                    }
                    
                    stGroup.add(tileGroup);
                }
                
                chipGroup.add(stGroup);
            }
            
            // Global DRAM buffer
            const dramGeometry = new THREE.BoxGeometry(18, 1, 2);
            const dramMaterial = new THREE.MeshLambertMaterial({ color: 0xff6b6b, transparent: true, opacity: 0.7 });
            const dramMesh = new THREE.Mesh(dramGeometry, dramMaterial);
            dramMesh.position.set(0, 0.8, -6.5);
            chipGroup.add(dramMesh);
            
            // Microcontroller
            const mcuGeometry = new THREE.BoxGeometry(3, 1.2, 2);
            const mcuMaterial = new THREE.MeshLambertMaterial({ color: 0x6c5ce7, transparent: true, opacity: 0.8 });
            const mcuMesh = new THREE.Mesh(mcuGeometry, mcuMaterial);
            mcuMesh.position.set(8, 0.9, -5);
            chipGroup.add(mcuMesh);
            
            // Interconnect visualization (simplified)
            const interconnectMaterial = new THREE.LineBasicMaterial({ color: 0x00d4aa, opacity: 0.6, transparent: true });
            const interconnectGeometry = new THREE.BufferGeometry();
            const interconnectPoints = [];
            
            // Add interconnect lines between major components
            for (let i = -8; i <= 8; i += 4) {
                interconnectPoints.push(i, 0.1, -6); // Horizontal lines
                interconnectPoints.push(i, 0.1, 6);
            }
            for (let j = -6; j <= 6; j += 4) {
                interconnectPoints.push(-8, 0.1, j); // Vertical lines
                interconnectPoints.push(8, 0.1, j);
            }
            
            interconnectGeometry.setAttribute('position', new THREE.Float32BufferAttribute(interconnectPoints, 3));
            const interconnectLines = new THREE.LineSegments(interconnectGeometry, interconnectMaterial);
            chipGroup.add(interconnectLines);
            
            scene.add(chipGroup);
        }
        
        function animate() {
            requestAnimationFrame(animate);
            
            // Slowly rotate the chip for better viewing
            if (chipGroup) {
                chipGroup.rotation.y += 0.003;
            }
            
            renderer.render(scene, camera);
        }
        
        // WebSocket connection and data handling
        socket.on('connect', function() {
            document.getElementById('connection-status').innerHTML = 
                '<span class="status-indicator status-active"></span>Connected';
        });
        
        socket.on('stats_update', function(data) {
            updateArchitectureMetrics(data);
        });
        
        function updateArchitectureMetrics(data) {
            console.log('Architecture metrics update:', data);
            
            // Update performance metrics
            if (data.peripherals) {
                const totalOps = data.peripherals.total_crossbar_operations || 0;
                document.getElementById('perf-operations').textContent = totalOps.toLocaleString();
                
                // Update throughput calculation
                const throughput = data.peripherals.throughput || 0;
                document.getElementById('perf-throughput').textContent = throughput.toFixed(1) + ' ops/s';
                
                // Update utilization percentage
                const utilization = totalOps > 0 ? Math.min(100, (totalOps / 1000) * 100) : 0;
                document.getElementById('perf-utilization').textContent = utilization.toFixed(1) + '%';
            }
            
            if (data.chip && data.chip.energy) {
                const energy = (data.chip.energy.total_energy || 0) * 1000;
                document.getElementById('perf-energy').textContent = energy.toFixed(2) + ' mJ';
            }
            
            // Update total metrics in left sidebar
            if (data.peripherals) {
                const activeTiles = data.crossbars ? data.crossbars.filter(cb => cb.utilization > 0.1).length / 4 : 0;
                document.getElementById('active-tiles').textContent = Math.floor(activeTiles);
            }
            
            // Update memory utilization bars
            if (data.memory) {
                const globalUtil = getMemoryUtil(data.memory, 'global_buffer') * 100;
                const sharedUtil = getMemoryUtil(data.memory, 'shared_buffers') * 100;
                const localUtil = getMemoryUtil(data.memory, 'local_buffers') * 100;
                
                document.getElementById('global-memory-bar').style.width = globalUtil + '%';
                document.getElementById('shared-memory-bar').style.width = sharedUtil + '%';
                document.getElementById('local-memory-bar').style.width = localUtil + '%';
                
                document.getElementById('global-memory-text').textContent = 
                    `4096 KB | ${globalUtil.toFixed(1)}% utilized`;
                document.getElementById('shared-memory-text').textContent = 
                    `512 KB × 2 | ${sharedUtil.toFixed(1)}% utilized`;
                document.getElementById('local-memory-text').textContent = 
                    `64 KB × 8 | ${localUtil.toFixed(1)}% utilized`;
            }
            
            // Update 3D visualization with real data
            update3DVisualization(data);
            
            // Update data flow diagram if visible
            if (currentView === 'dataflow') {
                updateDataFlowVisualization(data);
            }
            
            // Update interconnect diagram if visible
            if (currentView === 'interconnect') {
                updateInterconnectVisualization(data);
            }
        }
        
        function getMemoryUtil(memData, bufferType) {
            if (!memData || !memData[bufferType]) return 0;
            const bufferStats = memData[bufferType];
            if (bufferStats.memory_stats && bufferStats.memory_stats.utilization !== undefined) {
                return Math.min(1.0, bufferStats.memory_stats.utilization);
            }
            const operations = bufferStats.operations || 0;
            return Math.min(1.0, operations / 100);
        }
        
        // Handle component selection
        function selectComponent(componentType, componentData = {}) {
            selectedComponent = componentType;
            
            // Update component tree highlighting
            document.querySelectorAll('.component-item').forEach(item => {
                item.classList.remove('active');
            });
            
            const selectedItem = document.querySelector(`[data-component="${componentType}"]`);
            if (selectedItem) {
                selectedItem.classList.add('active');
            }
            
            // Update detail panel
            updateComponentDetails(componentType, componentData);
        }
        
        function updateComponentDetails(componentType, data = {}) {
            const detailPanel = document.getElementById('selected-component');
            let detailHTML = '';
            
            switch (componentType) {
                case 'chip':
                    detailHTML = `
                        <div class="detail-title">🔲 ReRAM Chip Overview</div>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <div class="detail-label">Architecture:</div>
                                <div class="detail-value">Hierarchical</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">SuperTiles:</div>
                                <div class="detail-value">2</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Total Tiles:</div>
                                <div class="detail-value">8</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Total Crossbars:</div>
                                <div class="detail-value">32</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Crossbar Size:</div>
                                <div class="detail-value">128×128</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Total Cells:</div>
                                <div class="detail-value">524,288</div>
                            </div>
                        </div>
                    `;
                    break;
                    
                case 'supertile':
                    detailHTML = `
                        <div class="detail-title">🏢 SuperTile Details</div>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <div class="detail-label">Tiles per SuperTile:</div>
                                <div class="detail-value">4</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Crossbars per SuperTile:</div>
                                <div class="detail-value">16</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Shared eDRAM Buffer:</div>
                                <div class="detail-value">512 KB</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">eDRAM Banks:</div>
                                <div class="detail-value">4</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">eDRAM Latency:</div>
                                <div class="detail-value">3-4 cycles</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Interconnect Type:</div>
                                <div class="detail-value">Mesh Network</div>
                            </div>
                        </div>
                    `;
                    break;
                    
                case 'tile':
                    detailHTML = `
                        <div class="detail-title">🏠 Processing Tile Details</div>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <div class="detail-label">Crossbars per Tile:</div>
                                <div class="detail-value">4</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Local SRAM Buffer:</div>
                                <div class="detail-value">64 KB</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">SRAM Banks:</div>
                                <div class="detail-value">2</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">SRAM Latency:</div>
                                <div class="detail-value">1 cycle</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">ADC Units:</div>
                                <div class="detail-value">128-256</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">DAC Units:</div>
                                <div class="detail-value">512</div>
                            </div>
                        </div>
                    `;
                    break;
                    
                case 'global_memory':
                    detailHTML = `
                        <div class="detail-title">🗄️ Global DRAM Buffer</div>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <div class="detail-label">Memory Type:</div>
                                <div class="detail-value">DRAM</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Total Capacity:</div>
                                <div class="detail-value">4096 KB (4 MB)</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Banks:</div>
                                <div class="detail-value">8</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Read Latency:</div>
                                <div class="detail-value">10 cycles</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Write Latency:</div>
                                <div class="detail-value">12 cycles</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Data Width:</div>
                                <div class="detail-value">256 bits</div>
                            </div>
                        </div>
                    `;
                    break;
                    
                case 'shared_memory':
                    detailHTML = `
                        <div class="detail-title">💾 Shared eDRAM Buffer</div>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <div class="detail-label">Memory Type:</div>
                                <div class="detail-value">eDRAM (embedded)</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Capacity per SuperTile:</div>
                                <div class="detail-value">512 KB</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Banks:</div>
                                <div class="detail-value">4</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Read Latency:</div>
                                <div class="detail-value">3 cycles</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Write Latency:</div>
                                <div class="detail-value">4 cycles</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Refresh Rate:</div>
                                <div class="detail-value">64ms</div>
                            </div>
                        </div>
                    `;
                    break;
                    
                case 'microcontroller':
                    detailHTML = `
                        <div class="detail-title">🖥️ Microcontroller Unit</div>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <div class="detail-label">Clock Frequency:</div>
                                <div class="detail-value">1000 MHz</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Pipeline Stages:</div>
                                <div class="detail-value">5</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Instruction Buffer:</div>
                                <div class="detail-value">32 entries</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Control Registers:</div>
                                <div class="detail-value">16</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">ALU Units:</div>
                                <div class="detail-value">2</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Memory Controllers:</div>
                                <div class="detail-value">3</div>
                            </div>
                        </div>
                    `;
                    break;
                    
                case 'interconnect':
                    detailHTML = `
                        <div class="detail-title">🔗 Interconnect Network</div>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <div class="detail-label">Topology:</div>
                                <div class="detail-value">2D Mesh</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Data Width:</div>
                                <div class="detail-value">256 bits</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Clock Frequency:</div>
                                <div class="detail-value">1000 MHz</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Router Latency:</div>
                                <div class="detail-value">2 cycles</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Link Latency:</div>
                                <div class="detail-value">1 cycle</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Flow Control:</div>
                                <div class="detail-value">Credit-based</div>
                            </div>
                        </div>
                    `;
                    break;
                    
                default:
                    detailHTML = `
                        <div class="detail-title">🔲 Component Details</div>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <div class="detail-label">Component:</div>
                                <div class="detail-value">${componentType}</div>
                            </div>
                        </div>
                    `;
            }
            
            detailPanel.innerHTML = detailHTML;
        }
        
        // Update 3D visualization with real-time data
        function update3DVisualization(data) {
            if (!chipGroup || !data.crossbars) return;
            
            // Update crossbar materials based on utilization
            chipGroup.traverse(child => {
                if (child.userData && child.userData.type === 'crossbar') {
                    const cbData = data.crossbars.find(cb => cb.id === child.userData.id);
                    if (cbData && child.material) {
                        const utilization = cbData.utilization || 0;
                        
                        // Color based on utilization: blue (idle) -> green -> yellow -> red (busy)
                        if (utilization > 0.7) {
                            child.material.color.setHex(0xff6b6b); // Red - very busy
                        } else if (utilization > 0.4) {
                            child.material.color.setHex(0xffd93d); // Yellow - moderately busy
                        } else if (utilization > 0.1) {
                            child.material.color.setHex(0x4ecdc4); // Green - active
                        } else {
                            child.material.color.setHex(0x96ceb4); // Blue-green - idle
                        }
                        
                        // Pulse effect for active crossbars
                        if (utilization > 0.1) {
                            const pulse = (Math.sin(Date.now() * 0.005) + 1) * 0.2 + 0.8;
                            child.material.opacity = pulse;
                        } else {
                            child.material.opacity = 0.9;
                        }
                    }
                }
            });
        }
        
        // Update data flow visualization with real-time data
        function updateDataFlowVisualization(data) {
            if (!data.execution) return;
            
            const svg = d3.select('#dataflow-svg');
            
            // Add execution progress indicators
            const progressData = [
                { layer: 0, progress: data.execution.current_layer > 0 ? 1.0 : data.execution.progress },
                { layer: 1, progress: data.execution.current_layer > 1 ? 1.0 : (data.execution.current_layer === 1 ? data.execution.progress : 0) },
                { layer: 2, progress: data.execution.current_layer > 2 ? 1.0 : (data.execution.current_layer === 2 ? data.execution.progress : 0) }
            ];
            
            // Update layer node opacity based on progress
            svg.selectAll('.layer-node')
                .each(function(d, i) {
                    const progress = progressData[i] ? progressData[i].progress : 0;
                    d3.select(this).select('rect')
                        .attr('opacity', 0.4 + progress * 0.6)
                        .attr('stroke', progress > 0.1 ? '#00d4aa' : '#666')
                        .attr('stroke-width', progress > 0.1 ? 3 : 1);
                });
        }
        
        // Update interconnect visualization with real-time data
        function updateInterconnectVisualization(data) {
            if (!data.memory && !data.peripherals) return;
            
            const svg = d3.select('#interconnect-svg');
            
            // Update node utilization based on memory and peripheral activity
            svg.selectAll('.interconnect-node circle')
                .attr('stroke-width', function(d) {
                    if (d.type === 'memory' && data.memory) {
                        const util = getMemoryUtil(data.memory, 'global_buffer');
                        return 2 + util * 4;
                    } else if (d.type === 'controller' && data.peripherals) {
                        const util = (data.peripherals.total_crossbar_operations || 0) / 1000;
                        return 2 + Math.min(util, 1) * 4;
                    }
                    return 2;
                })
                .attr('stroke', function(d) {
                    const activity = Math.random() * 0.5 + 0.5; // Simulate activity
                    if (activity > 0.7) return '#ff6b6b';
                    if (activity > 0.4) return '#ffd93d';
                    return '#4ecdc4';
                });
        }
        
        // Monitoring control functions
        function startArchitectureMonitoring() {
            console.log('Starting architecture monitoring...');
            socket.emit('start_monitoring');
            document.getElementById('arch-start-btn').disabled = true;
            document.getElementById('arch-stop-btn').disabled = false;
        }
        
        function stopArchitectureMonitoring() {
            console.log('Stopping architecture monitoring...');
            socket.emit('stop_monitoring');
            document.getElementById('arch-start-btn').disabled = false;
            document.getElementById('arch-stop-btn').disabled = true;
        }
        
        // Enhanced WebSocket event handling
        socket.on('monitoring_started', function(data) {
            console.log('Monitoring started:', data);
            document.getElementById('arch-start-btn').disabled = true;
            document.getElementById('arch-stop-btn').disabled = false;
        });
        
        socket.on('monitoring_stopped', function(data) {
            console.log('Monitoring stopped:', data);
            document.getElementById('arch-start-btn').disabled = false;
            document.getElementById('arch-stop-btn').disabled = true;
        });
        
        // Initialize everything
        document.addEventListener('DOMContentLoaded', function() {
            init3DView();
            
            // Handle view switching
            document.querySelectorAll('.arch-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const view = this.dataset.view;
                    switchView(view);
                });
            });
            
            // Handle component tree clicks
            document.querySelectorAll('.component-item').forEach(item => {
                item.addEventListener('click', function() {
                    const componentType = this.dataset.component;
                    selectComponent(componentType);
                });
            });
            
            // Auto-start monitoring when connected
            socket.on('connect', function() {
                setTimeout(startArchitectureMonitoring, 1000);
            });
        });
        
        function switchView(view) {
            currentView = view;
            
            // Update button states
            document.querySelectorAll('.arch-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.view === view);
            });
            
            // Show/hide appropriate views
            document.getElementById('arch-3d-container').style.display = 
                view === 'hierarchy' || view === '3d' ? 'block' : 'none';
            document.getElementById('dataflow-view').style.display = 
                view === 'dataflow' ? 'block' : 'none';
            document.getElementById('interconnect-view').style.display = 
                view === 'interconnect' ? 'block' : 'none';
                
            // Initialize specific view content
            if (view === 'dataflow') {
                initializeDataFlowDiagram();
            } else if (view === 'interconnect') {
                initializeInterconnectDiagram();
            } else if (view === 'memory') {
                initializeMemoryView();
            }
        }
        
        function initializeDataFlowDiagram() {
            const svg = d3.select('#dataflow-svg');
            svg.selectAll('*').remove();
            
            const width = parseInt(svg.style('width'));
            const height = parseInt(svg.style('height'));
            
            // Create data flow diagram showing CNN layers mapped to hardware
            const layers = [
                { name: 'Input\n(16×16×1)', x: 50, y: 150, type: 'input' },
                { name: 'Conv2D\n(3×3×8)', x: 200, y: 100, type: 'conv' },
                { name: 'Pooling\n(2×2)', x: 200, y: 200, type: 'pool' },
                { name: 'Dense\n(10)', x: 350, y: 150, type: 'dense' },
                { name: 'Output\n(10)', x: 500, y: 150, type: 'output' }
            ];
            
            const hardware = [
                { name: 'Global\nDRAM', x: 150, y: 50, type: 'memory' },
                { name: 'SuperTile 0', x: 300, y: 50, type: 'supertile' },
                { name: 'SuperTile 1', x: 450, y: 50, type: 'supertile' },
                { name: 'Crossbar\nArrays', x: 300, y: 250, type: 'crossbar' },
                { name: 'Shared\neDRAM', x: 450, y: 250, type: 'memory' }
            ];
            
            // Draw connections
            const connections = [
                { source: layers[0], target: layers[1] },
                { source: layers[1], target: layers[2] },
                { source: layers[2], target: layers[3] },
                { source: layers[3], target: layers[4] },
                { source: hardware[0], target: hardware[1] },
                { source: hardware[1], target: hardware[3] },
                { source: hardware[2], target: hardware[4] }
            ];
            
            // Draw connection lines
            svg.selectAll('.connection')
                .data(connections)
                .enter()
                .append('line')
                .attr('class', 'connection')
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y)
                .attr('stroke', '#00d4aa')
                .attr('stroke-width', 2)
                .attr('opacity', 0.6);
            
            // Draw layer nodes
            const layerNodes = svg.selectAll('.layer-node')
                .data(layers)
                .enter()
                .append('g')
                .attr('class', 'layer-node')
                .attr('transform', d => `translate(${d.x},${d.y})`);
            
            layerNodes.append('rect')
                .attr('x', -40)
                .attr('y', -25)
                .attr('width', 80)
                .attr('height', 50)
                .attr('rx', 8)
                .attr('fill', d => {
                    switch(d.type) {
                        case 'input': return '#ff6b6b';
                        case 'conv': return '#4ecdc4';
                        case 'pool': return '#45b7d1';
                        case 'dense': return '#96ceb4';
                        case 'output': return '#ffd93d';
                        default: return '#666';
                    }
                })
                .attr('opacity', 0.8);
            
            layerNodes.append('text')
                .attr('text-anchor', 'middle')
                .attr('dy', '0.35em')
                .attr('fill', 'white')
                .attr('font-size', '12px')
                .attr('font-weight', 'bold')
                .selectAll('tspan')
                .data(d => d.name.split('\n'))
                .enter()
                .append('tspan')
                .attr('x', 0)
                .attr('dy', (d, i) => i === 0 ? '-0.2em' : '1.2em')
                .text(d => d);
            
            // Draw hardware nodes
            const hardwareNodes = svg.selectAll('.hardware-node')
                .data(hardware)
                .enter()
                .append('g')
                .attr('class', 'hardware-node')
                .attr('transform', d => `translate(${d.x},${d.y})`);
            
            hardwareNodes.append('rect')
                .attr('x', -45)
                .attr('y', -20)
                .attr('width', 90)
                .attr('height', 40)
                .attr('rx', 6)
                .attr('fill', d => {
                    switch(d.type) {
                        case 'memory': return '#a29bfe';
                        case 'supertile': return '#fd79a8';
                        case 'crossbar': return '#fdcb6e';
                        default: return '#666';
                    }
                })
                .attr('opacity', 0.7)
                .attr('stroke', '#fff')
                .attr('stroke-width', 2);
            
            hardwareNodes.append('text')
                .attr('text-anchor', 'middle')
                .attr('dy', '0.35em')
                .attr('fill', 'white')
                .attr('font-size', '11px')
                .attr('font-weight', 'bold')
                .selectAll('tspan')
                .data(d => d.name.split('\n'))
                .enter()
                .append('tspan')
                .attr('x', 0)
                .attr('dy', (d, i) => i === 0 ? '-0.3em' : '1.1em')
                .text(d => d);
        }
        
        function initializeInterconnectDiagram() {
            const svg = d3.select('#interconnect-svg');
            svg.selectAll('*').remove();
            
            const width = parseInt(svg.style('width'));
            const height = parseInt(svg.style('height'));
            
            // Create mesh interconnect topology
            const nodes = [];
            const links = [];
            
            // Create 2x2 mesh of SuperTiles
            for (let i = 0; i < 2; i++) {
                for (let j = 0; j < 2; j++) {
                    nodes.push({
                        id: `ST_${i}_${j}`,
                        x: 150 + j * 200,
                        y: 80 + i * 120,
                        type: 'supertile',
                        utilization: Math.random() * 0.8 + 0.2
                    });
                }
            }
            
            // Add memory controllers
            nodes.push({
                id: 'MEM_CTRL',
                x: width / 2,
                y: 30,
                type: 'memory',
                utilization: 0.6
            });
            
            // Add microcontroller
            nodes.push({
                id: 'MCU',
                x: width - 80,
                y: height / 2,
                type: 'controller',
                utilization: 0.4
            });
            
            // Create mesh connections
            const stNodes = nodes.filter(n => n.type === 'supertile');
            for (let i = 0; i < stNodes.length; i++) {
                for (let j = i + 1; j < stNodes.length; j++) {
                    const distance = Math.sqrt(
                        Math.pow(stNodes[i].x - stNodes[j].x, 2) + 
                        Math.pow(stNodes[i].y - stNodes[j].y, 2)
                    );
                    if (distance < 250) { // Connect nearby nodes
                        links.push({
                            source: stNodes[i],
                            target: stNodes[j],
                            utilization: Math.random() * 0.7
                        });
                    }
                }
            }
            
            // Connect to memory and MCU
            stNodes.forEach(st => {
                links.push({
                    source: st,
                    target: nodes.find(n => n.id === 'MEM_CTRL'),
                    utilization: Math.random() * 0.5
                });
                links.push({
                    source: st,
                    target: nodes.find(n => n.id === 'MCU'),
                    utilization: Math.random() * 0.3
                });
            });
            
            // Draw links
            svg.selectAll('.interconnect-link')
                .data(links)
                .enter()
                .append('line')
                .attr('class', 'interconnect-link')
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y)
                .attr('stroke', d => {
                    const intensity = d.utilization;
                    if (intensity > 0.6) return '#ff6b6b';
                    if (intensity > 0.3) return '#ffd93d';
                    return '#4ecdc4';
                })
                .attr('stroke-width', d => 2 + d.utilization * 4)
                .attr('opacity', 0.7);
            
            // Draw nodes
            const nodeGroups = svg.selectAll('.interconnect-node')
                .data(nodes)
                .enter()
                .append('g')
                .attr('class', 'interconnect-node')
                .attr('transform', d => `translate(${d.x},${d.y})`);
            
            nodeGroups.append('circle')
                .attr('r', d => d.type === 'supertile' ? 25 : 20)
                .attr('fill', d => {
                    switch(d.type) {
                        case 'supertile': return '#74b9ff';
                        case 'memory': return '#a29bfe';
                        case 'controller': return '#fd79a8';
                        default: return '#666';
                    }
                })
                .attr('opacity', 0.8)
                .attr('stroke', '#fff')
                .attr('stroke-width', 2);
            
            // Add utilization indicators
            nodeGroups.append('circle')
                .attr('r', d => (d.type === 'supertile' ? 25 : 20) * 0.7)
                .attr('fill', 'none')
                .attr('stroke', '#00d4aa')
                .attr('stroke-width', 3)
                .attr('stroke-dasharray', d => {
                    const circumference = 2 * Math.PI * (d.type === 'supertile' ? 25 : 20) * 0.7;
                    const filled = circumference * d.utilization;
                    return `${filled} ${circumference - filled}`;
                })
                .attr('stroke-dashoffset', d => {
                    const circumference = 2 * Math.PI * (d.type === 'supertile' ? 25 : 20) * 0.7;
                    return circumference * 0.25; // Start from top
                });
            
            nodeGroups.append('text')
                .attr('text-anchor', 'middle')
                .attr('dy', '0.35em')
                .attr('fill', 'white')
                .attr('font-size', '10px')
                .attr('font-weight', 'bold')
                .text(d => d.id.replace('_', '\n'));
        }
        
        function initializeMemoryView() {
            // Show memory view in the 3D container
            document.getElementById('arch-3d-container').style.display = 'block';
            
            // Update 3D scene to highlight memory components
            if (chipGroup) {
                // Reset all materials
                chipGroup.traverse(child => {
                    if (child.material) {
                        child.material.opacity = 0.3;
                        child.material.transparent = true;
                    }
                });
                
                // Highlight memory components
                chipGroup.traverse(child => {
                    if (child.userData && 
                        (child.userData.type === 'memory' || 
                         child.material.color.getHex() === 0xff6b6b || // DRAM
                         child.material.color.getHex() === 0x4ecdc4 || // eDRAM
                         child.material.color.getHex() === 0xffd93d)) { // SRAM
                        child.material.opacity = 1.0;
                    }
                });
            }
        }
        
        // Handle window resize
        window.addEventListener('resize', function() {
            if (camera && renderer) {
                const container = document.getElementById('arch-3d-canvas');
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }
        });
    </script>
</body>
</html>
            """)
        
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
        <div style="margin: 10px 0;">
            <a href="/architecture" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; padding: 10px 20px; border-radius: 25px; margin: 0 10px; font-weight: bold; transition: all 0.3s;">
                🏗️ Detailed Architecture View
            </a>
            <a href="/educational" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; text-decoration: none; padding: 10px 20px; border-radius: 25px; margin: 0 10px; font-weight: bold; transition: all 0.3s;">
                🎓 Educational Tool
            </a>
        </div>
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
        <div style="margin: 15px 0;">
            <a href="/" style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; text-decoration: none; padding: 10px 20px; border-radius: 25px; margin: 0 10px; font-weight: bold; transition: all 0.3s;">
                🏠 Dashboard
            </a>
            <a href="/architecture" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; padding: 10px 20px; border-radius: 25px; margin: 0 10px; font-weight: bold; transition: all 0.3s;">
                🏗️ Architecture View
            </a>
        </div>
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
            <p>Monitor live execution progress and see how CNNs map to ReRAM hardware:</p>
            <div id="live-status" style="margin: 15px 0; padding: 15px; background: #333; border-radius: 8px;">
                <div>Status: <span id="exec-status" style="color: #ffd93d;">Waiting for simulation...</span></div>
                <div>Current Layer: <span id="current-layer">N/A</span></div>
                <div>Crossbar Operations: <span id="crossbar-ops">0</span></div>
                <div>Active Crossbars: <span id="active-crossbars">0</span></div>
            </div>
            <p>Return to the <a href="/" style="color: #4CAF50;">main dashboard</a> for detailed visualization.</p>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <script>
        let socket = io();
        
        socket.on('connect', function() {
            document.getElementById('exec-status').textContent = 'Connected to simulator';
            document.getElementById('exec-status').style.color = '#4CAF50';
            
            // Auto-start monitoring for educational purposes
            setTimeout(() => {
                socket.emit('start_monitoring');
            }, 1000);
        });
        
        socket.on('stats_update', function(data) {
            // Update educational dashboard with live data
            if (data.execution) {
                if (data.execution.running) {
                    document.getElementById('exec-status').textContent = 'Simulation running...';
                    document.getElementById('exec-status').style.color = '#4CAF50';
                    document.getElementById('current-layer').textContent = 
                        `Layer ${data.execution.current_layer + 1} of ${data.execution.total_layers}`;
                } else {
                    document.getElementById('exec-status').textContent = 'Simulation complete';
                    document.getElementById('exec-status').style.color = '#ffd93d';
                }
            }
            
            if (data.peripherals) {
                document.getElementById('crossbar-ops').textContent = 
                    (data.peripherals.total_crossbar_operations || 0).toLocaleString();
                document.getElementById('active-crossbars').textContent = 
                    data.peripherals.active_crossbars || 0;
            }
        });
        
        socket.on('disconnect', function() {
            document.getElementById('exec-status').textContent = 'Disconnected';
            document.getElementById('exec-status').style.color = '#f44336';
        });
    </script>
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
            
            # Calculate throughput (operations per second)
            current_time = time.time()
            if not hasattr(self, '_last_ops_time'):
                self._last_ops_time = current_time
                self._last_ops_count = total_crossbar_ops
                throughput = 0.0
            else:
                time_diff = current_time - self._last_ops_time
                ops_diff = total_crossbar_ops - self._last_ops_count
                if time_diff > 0:
                    throughput = ops_diff / time_diff
                else:
                    throughput = 0.0
                
                # Update for next calculation
                self._last_ops_time = current_time
                self._last_ops_count = total_crossbar_ops
            
            # Add network topology data for interconnect visualization
            network_data = {
                'supertiles': len(self.execution_engine.chip.supertiles),
                'tiles_per_supertile': len(self.execution_engine.chip.supertiles[0].tiles) if self.execution_engine.chip.supertiles else 0,
                'crossbars_per_tile': len(self.execution_engine.chip.supertiles[0].tiles[0].crossbars) if self.execution_engine.chip.supertiles and self.execution_engine.chip.supertiles[0].tiles else 0,
                'total_nodes': len(self.execution_engine.chip.supertiles) + 2  # SuperTiles + Memory Controller + MCU
            }
            
            return {
                'chip': chip_stats,
                'crossbars': crossbar_data,
                'memory': memory_data,
                'execution': execution_data,
                'network': network_data,
                'peripherals': {
                    'adc_utilization': float(adc_utilization),
                    'dac_utilization': float(dac_utilization),
                    'adc_conversions': int(total_adc_conversions),
                    'dac_conversions': int(total_dac_conversions),
                    'total_crossbar_operations': int(total_crossbar_ops),
                    'throughput': float(throughput),
                    'active_crossbars': len([cb for cb in crossbar_data if cb['utilization'] > 0.1])
                },
                'timestamp': current_time
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