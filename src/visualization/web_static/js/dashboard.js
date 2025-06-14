/**
 * ReRAM Crossbar Simulator - Web Dashboard JavaScript
 * Real-time monitoring and visualization
 */

class ReRAMDashboard {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.isMonitoring = false;
        this.crossbarGrid = null;
        this.chip3DVisualization = null;
        this.lastUpdateTime = 0;
        this.performanceHistory = {
            operations: [],
            energy: [],
            timestamps: []
        };
        
        // UI Elements
        this.elements = {
            connectionStatus: document.getElementById('connection-status'),
            startBtn: document.getElementById('start-monitoring'),
            stopBtn: document.getElementById('stop-monitoring'),
            crossbarHeatmap: document.getElementById('crossbar-heatmap'),
            chip3DView: document.getElementById('chip-3d-view'),
            tooltip: document.getElementById('tooltip')
        };
        
        this.setupEventListeners();
    }
    
    init() {
        console.log('Initializing ReRAM Dashboard...');
        this.connectWebSocket();
        this.initializeCrossbarGrid();
        this.initialize3DVisualization();
    }
    
    setupEventListeners() {
        // Button event listeners
        this.elements.startBtn.addEventListener('click', () => this.startMonitoring());
        this.elements.stopBtn.addEventListener('click', () => this.stopMonitoring());
        
        // Window resize handling
        window.addEventListener('resize', () => {
            if (this.chip3DVisualization) {
                this.chip3DVisualization.handleResize();
            }
        });
    }
    
    connectWebSocket() {
        try {
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('Connected to ReRAM Simulator');
                this.updateConnectionStatus(true);
            });
            
            this.socket.on('disconnect', () => {
                console.log('Disconnected from ReRAM Simulator');
                this.updateConnectionStatus(false);
            });
            
            this.socket.on('stats_update', (data) => {
                this.handleStatsUpdate(data);
            });
            
            this.socket.on('monitoring_started', (data) => {
                console.log('Monitoring started:', data);
                this.isMonitoring = true;
                this.updateMonitoringButtons();
            });
            
            this.socket.on('monitoring_stopped', (data) => {
                console.log('Monitoring stopped:', data);
                this.isMonitoring = false;
                this.updateMonitoringButtons();
            });
            
            this.socket.on('connected', (data) => {
                console.log('Server message:', data);
            });
            
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    updateConnectionStatus(connected) {
        this.isConnected = connected;
        const statusEl = this.elements.connectionStatus;
        const indicator = statusEl.querySelector('.status-indicator');
        
        if (connected) {
            statusEl.innerHTML = '<span class="status-indicator status-connected"></span>Connected';
        } else {
            statusEl.innerHTML = '<span class="status-indicator status-disconnected"></span>Disconnected';
        }
        
        this.updateMonitoringButtons();
    }
    
    updateMonitoringButtons() {
        this.elements.startBtn.disabled = !this.isConnected || this.isMonitoring;
        this.elements.stopBtn.disabled = !this.isConnected || !this.isMonitoring;
    }
    
    startMonitoring() {
        if (this.socket && this.isConnected) {
            this.socket.emit('start_monitoring');
        }
    }
    
    stopMonitoring() {
        if (this.socket && this.isConnected) {
            this.socket.emit('stop_monitoring');
        }
    }
    
    handleStatsUpdate(data) {
        console.log('Stats update received:', data);
        
        try {
            // Update crossbar activity
            if (data.crossbars) {
                this.updateCrossbarHeatmap(data.crossbars);
            }
            
            // Update memory utilization
            if (data.memory) {
                this.updateMemoryBars(data.memory);
            }
            
            // Update peripheral activity
            if (data.peripherals) {
                this.updatePeripheralActivity(data.peripherals);
            }
            
            // Update performance metrics
            if (data.chip) {
                this.updatePerformanceMetrics(data.chip);
            }
            
            // Update execution progress
            if (data.execution) {
                this.updateExecutionProgress(data.execution);
            }
            
            // Update 3D visualization
            if (this.chip3DVisualization && data.crossbars) {
                this.chip3DVisualization.updateActivity(data.crossbars);
            }
            
            this.lastUpdateTime = Date.now();
            
        } catch (error) {
            console.error('Error handling stats update:', error);
        }
    }
    
    initializeCrossbarGrid() {
        const heatmapEl = this.elements.crossbarHeatmap;
        heatmapEl.innerHTML = ''; // Clear existing content
        
        // Create a 4x8 grid representing 32 crossbars (2 SuperTiles × 4 Tiles × 4 Crossbars)
        for (let i = 0; i < 32; i++) {
            const cell = document.createElement('div');
            cell.className = 'crossbar-cell';
            cell.id = `crossbar-${i}`;
            cell.dataset.crossbarId = i;
            
            // Add hover effects
            cell.addEventListener('mouseenter', (e) => this.showCrossbarTooltip(e, i));
            cell.addEventListener('mouseleave', () => this.hideTooltip());
            
            heatmapEl.appendChild(cell);
        }
    }
    
    updateCrossbarHeatmap(crossbarData) {
        crossbarData.forEach((crossbar, index) => {
            const cell = document.getElementById(`crossbar-${index}`);
            if (!cell) return;
            
            // Remove existing activity classes
            cell.classList.remove('active', 'medium', 'low');
            
            // Add activity class based on utilization
            const utilization = crossbar.utilization || 0;
            if (utilization > 0.7) {
                cell.classList.add('active');
            } else if (utilization > 0.3) {
                cell.classList.add('medium');
            } else if (utilization > 0.1) {
                cell.classList.add('low');
            }
            
            // Store data for tooltip
            cell.dataset.operations = crossbar.operations || 0;
            cell.dataset.utilization = (utilization * 100).toFixed(1);
            cell.dataset.energy = crossbar.energy || 0;
            cell.dataset.crossbarId = crossbar.id || `XB-${index}`;
        });
    }
    
    updateMemoryBars(memoryData) {
        // Update global buffer
        const globalUtil = this.getMemoryUtilization(memoryData, 'global_buffer') * 100;
        document.getElementById('global-buffer-fill').style.width = `${globalUtil}%`;
        
        // Update shared buffers
        const sharedUtil = this.getMemoryUtilization(memoryData, 'shared_buffers') * 100;
        document.getElementById('shared-buffer-fill').style.width = `${sharedUtil}%`;
        
        // Update local buffers
        const localUtil = this.getMemoryUtilization(memoryData, 'local_buffers') * 100;
        document.getElementById('local-buffer-fill').style.width = `${localUtil}%`;
    }
    
    getMemoryUtilization(memoryData, bufferType) {
        if (!memoryData || !memoryData[bufferType]) return 0;
        
        const bufferStats = memoryData[bufferType];
        if (bufferStats.memory_stats && bufferStats.memory_stats.utilization !== undefined) {
            return Math.min(1.0, bufferStats.memory_stats.utilization);
        }
        
        // Fallback: calculate based on operations
        const operations = bufferStats.operations || 0;
        return Math.min(1.0, operations / 100); // Normalize to max 100 operations
    }
    
    updatePeripheralActivity(peripheralData) {
        // Update ADC activity
        const adcUtil = (peripheralData.adc_utilization || 0) * 100;
        document.getElementById('adc-progress').style.width = `${adcUtil}%`;
        document.getElementById('adc-percentage').textContent = `${adcUtil.toFixed(1)}%`;
        
        // Update DAC activity
        const dacUtil = (peripheralData.dac_utilization || 0) * 100;
        document.getElementById('dac-progress').style.width = `${dacUtil}%`;
        document.getElementById('dac-percentage').textContent = `${dacUtil.toFixed(1)}%`;
    }
    
    updatePerformanceMetrics(chipData) {
        // Total operations
        const totalOps = chipData.performance?.total_operations || 0;
        document.getElementById('total-operations').textContent = totalOps.toLocaleString();
        
        // Energy consumption
        const totalEnergy = chipData.energy?.total_energy || 0;
        const energyMJ = (totalEnergy * 1000).toFixed(2); // Convert to mJ
        document.getElementById('energy-consumption').textContent = `${energyMJ} mJ`;
        
        // Execution cycles (get from execution data if available)
        const execCycles = chipData.execution_cycles || 0;
        document.getElementById('execution-cycles').textContent = execCycles.toLocaleString();
        
        // Calculate throughput
        const currentTime = Date.now();
        if (this.lastUpdateTime > 0) {
            const timeDelta = (currentTime - this.lastUpdateTime) / 1000; // seconds
            const opsDelta = totalOps - (this.performanceHistory.operations.slice(-1)[0] || 0);
            const throughput = timeDelta > 0 ? (opsDelta / timeDelta).toFixed(1) : 0;
            document.getElementById('throughput').textContent = `${throughput} ops/s`;
        }
        
        // Store history for throughput calculation
        this.performanceHistory.operations.push(totalOps);
        this.performanceHistory.energy.push(totalEnergy);
        this.performanceHistory.timestamps.push(currentTime);
        
        // Keep only last 10 data points
        if (this.performanceHistory.operations.length > 10) {
            this.performanceHistory.operations.shift();
            this.performanceHistory.energy.shift();
            this.performanceHistory.timestamps.shift();
        }
    }
    
    updateExecutionProgress(executionData) {
        const progressEl = document.getElementById('execution-progress');
        
        if (executionData.running) {
            const currentLayer = executionData.current_layer || 0;
            const totalLayers = executionData.total_layers || 1;
            const progress = (executionData.progress || 0) * 100;
            
            progressEl.innerHTML = `
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Layer ${currentLayer + 1} of ${totalLayers}</span>
                        <span>${progress.toFixed(1)}%</span>
                    </div>
                    <div class="memory-bar">
                        <div class="memory-bar-fill" style="width: ${progress}%"></div>
                    </div>
                </div>
                <div style="text-align: center; color: #b0b8c1; font-size: 12px;">
                    Cycles: ${executionData.execution_cycles?.toLocaleString() || 0}
                </div>
            `;
        } else {
            progressEl.innerHTML = `
                <div class="loading">Waiting for execution data</div>
            `;
        }
    }
    
    initialize3DVisualization() {
        try {
            this.chip3DVisualization = new Chip3DVisualization('chip-3d-view');
            this.chip3DVisualization.init();
        } catch (error) {
            console.error('Failed to initialize 3D visualization:', error);
            document.getElementById('chip-3d-view').innerHTML = 
                '<div style="text-align: center; color: #ff6b6b; padding: 20px;">3D visualization not available</div>';
        }
    }
    
    showCrossbarTooltip(event, crossbarIndex) {
        const cell = event.target;
        const tooltip = this.elements.tooltip;
        
        const crossbarId = cell.dataset.crossbarId || `XB-${crossbarIndex}`;
        const operations = cell.dataset.operations || 0;
        const utilization = cell.dataset.utilization || 0;
        const energy = parseFloat(cell.dataset.energy || 0).toFixed(2);
        
        tooltip.innerHTML = `
            <strong>${crossbarId}</strong><br>
            Operations: ${operations}<br>
            Utilization: ${utilization}%<br>
            Energy: ${energy} J
        `;
        
        tooltip.style.left = event.pageX + 10 + 'px';
        tooltip.style.top = event.pageY - 10 + 'px';
        tooltip.style.opacity = 1;
    }
    
    hideTooltip() {
        this.elements.tooltip.style.opacity = 0;
    }
}

/**
 * 3D Chip Visualization using Three.js
 */
class Chip3DVisualization {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.crossbars = [];
        this.animationId = null;
        
        // Configuration
        this.config = {
            supertiles: 2,
            tilesPerSupertile: 4,
            crossbarsPerTile: 4
        };
    }
    
    init() {
        if (!window.THREE) {
            console.error('Three.js not loaded');
            return;
        }
        
        this.initializeScene();
        this.createChipGeometry();
        this.startAnimation();
        this.handleResize();
        
        console.log('3D Chip visualization initialized');
    }
    
    initializeScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0e1a);
        
        // Camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
        this.camera.position.set(10, 8, 10);
        this.camera.lookAt(0, 0, 0);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setClearColor(0x0a0e1a, 0);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Clear container and add renderer
        this.container.innerHTML = '';
        this.container.appendChild(this.renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Add some colored accent lights
        const light1 = new THREE.PointLight(0x00d4aa, 0.5, 20);
        light1.position.set(-5, 5, 5);
        this.scene.add(light1);
        
        const light2 = new THREE.PointLight(0x6c5ce7, 0.5, 20);
        light2.position.set(5, 5, -5);
        this.scene.add(light2);
    }
    
    createChipGeometry() {
        // Create SuperTiles
        for (let st = 0; st < this.config.supertiles; st++) {
            const stGroup = new THREE.Group();
            stGroup.position.x = st * 6 - 3;
            
            // SuperTile base
            const stGeometry = new THREE.BoxGeometry(5, 0.5, 5);
            const stMaterial = new THREE.MeshLambertMaterial({ 
                color: 0x2d3436,
                transparent: true,
                opacity: 0.8
            });
            const stMesh = new THREE.Mesh(stGeometry, stMaterial);
            stMesh.position.y = -0.25;
            stGroup.add(stMesh);
            
            // Create Tiles within SuperTile
            for (let t = 0; t < this.config.tilesPerSupertile; t++) {
                const tileGroup = new THREE.Group();
                const tileX = (t % 2) * 2.2 - 1.1;
                const tileZ = Math.floor(t / 2) * 2.2 - 1.1;
                tileGroup.position.set(tileX, 0.5, tileZ);
                
                // Tile base
                const tileGeometry = new THREE.BoxGeometry(2, 0.3, 2);
                const tileMaterial = new THREE.MeshLambertMaterial({ 
                    color: 0x636e72,
                    transparent: true,
                    opacity: 0.9
                });
                const tileMesh = new THREE.Mesh(tileGeometry, tileMaterial);
                tileGroup.add(tileMesh);
                
                // Create Crossbars within Tile
                for (let cb = 0; cb < this.config.crossbarsPerTile; cb++) {
                    const crossbarGeometry = new THREE.BoxGeometry(0.8, 0.2, 0.8);
                    const crossbarMaterial = new THREE.MeshLambertMaterial({ 
                        color: 0x74b9ff,
                        transparent: true,
                        opacity: 0.7
                    });
                    const crossbarMesh = new THREE.Mesh(crossbarGeometry, crossbarMaterial);
                    
                    const cbX = (cb % 2) * 0.9 - 0.45;
                    const cbZ = Math.floor(cb / 2) * 0.9 - 0.45;
                    crossbarMesh.position.set(cbX, 0.3, cbZ);
                    
                    // Store reference for activity updates
                    const crossbarIndex = st * this.config.tilesPerSupertile * this.config.crossbarsPerTile + 
                                         t * this.config.crossbarsPerTile + cb;
                    crossbarMesh.userData = { 
                        index: crossbarIndex,
                        defaultColor: 0x74b9ff,
                        material: crossbarMaterial
                    };
                    
                    this.crossbars.push(crossbarMesh);
                    tileGroup.add(crossbarMesh);
                }
                
                stGroup.add(tileGroup);
            }
            
            this.scene.add(stGroup);
        }
        
        // Add coordinate system helper
        const axesHelper = new THREE.AxesHelper(2);
        axesHelper.position.set(-8, 0, -8);
        this.scene.add(axesHelper);
    }
    
    updateActivity(crossbarData) {
        crossbarData.forEach((data, index) => {
            if (index < this.crossbars.length) {
                const crossbar = this.crossbars[index];
                const utilization = data.utilization || 0;
                
                // Update color based on activity level
                let color;
                if (utilization > 0.7) {
                    color = 0xff6b6b; // Red for high activity
                } else if (utilization > 0.3) {
                    color = 0xffd93d; // Yellow for medium activity
                } else if (utilization > 0.1) {
                    color = 0x00d4aa; // Green for low activity
                } else {
                    color = crossbar.userData.defaultColor; // Default blue
                }
                
                crossbar.material.color.setHex(color);
                
                // Add pulsing effect for active crossbars
                if (utilization > 0.1) {
                    const scale = 1 + Math.sin(Date.now() * 0.01) * 0.1 * utilization;
                    crossbar.scale.set(scale, scale, scale);
                } else {
                    crossbar.scale.set(1, 1, 1);
                }
            }
        });
    }
    
    startAnimation() {
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            
            // Rotate the entire scene slowly
            this.scene.rotation.y += 0.005;
            
            this.renderer.render(this.scene, this.camera);
        };
        
        animate();
    }
    
    handleResize() {
        if (!this.camera || !this.renderer) return;
        
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }
    
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.container && this.renderer) {
            this.container.removeChild(this.renderer.domElement);
        }
    }
}

// Export for global access
window.ReRAMDashboard = ReRAMDashboard;
window.Chip3DVisualization = Chip3DVisualization;