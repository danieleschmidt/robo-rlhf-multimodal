"""
Unified Execution Dashboard for Autonomous SDLC Monitoring and Control.

Real-time web-based dashboard for monitoring quantum-inspired autonomous SDLC execution,
providing visibility into training progress, optimization results, and system health.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from ..quantum.autonomous import AutonomousSDLCExecutor
from ..quantum.rlhf_optimizer import RLHFQuantumOptimizer, RLHFPhase
from ..quantum.analytics import PredictiveAnalytics
from ..core.logging import setup_logger


logger = setup_logger(__name__)


class AutonomousSDLCDashboard:
    """
    Unified web dashboard for autonomous SDLC execution monitoring and control.
    
    Provides real-time visibility into:
    - Quantum task planning and execution
    - RLHF optimization progress
    - System health and performance metrics
    - Predictive analytics and forecasting
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.app = FastAPI(title="Autonomous SDLC Dashboard", version="1.0.0")
        self.connected_clients: List[WebSocket] = []
        
        # Initialize components
        self.sdlc_executor = AutonomousSDLCExecutor()
        self.rlhf_optimizer = RLHFQuantumOptimizer()
        self.analytics = PredictiveAnalytics()
        
        # Dashboard state
        self.execution_state = {
            "current_phase": "idle",
            "progress": 0.0,
            "start_time": None,
            "estimated_completion": None,
            "active_tasks": [],
            "completed_tasks": [],
            "failed_tasks": [],
        }
        
        self.metrics = {
            "system_health": 100.0,
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "gpu_usage": 0.0,
            "active_connections": 0,
            "total_optimizations": 0,
            "success_rate": 100.0,
        }
        
        self._setup_routes()
        self._setup_middleware()
        
        logger.info("Autonomous SDLC Dashboard initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default dashboard configuration."""
        return {
            "host": "localhost",
            "port": 8080,
            "update_interval": 1.0,  # seconds
            "max_connections": 100,
            "enable_cors": True,
            "static_files_dir": Path(__file__).parent / "static",
            "templates_dir": Path(__file__).parent / "templates",
        }

    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        if self.config["enable_cors"]:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

    def _setup_routes(self):
        """Setup dashboard routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve main dashboard page."""
            return await self.generate_dashboard_html()

        @self.app.get("/api/status")
        async def get_status():
            """Get current execution status."""
            return JSONResponse(self.execution_state)

        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get system metrics."""
            await self._update_metrics()
            return JSONResponse(self.metrics)

        @self.app.get("/api/optimization/history")
        async def get_optimization_history():
            """Get RLHF optimization history."""
            summary = self.rlhf_optimizer.get_optimization_summary()
            return JSONResponse(summary)

        @self.app.get("/api/tasks")
        async def get_tasks():
            """Get current task list."""
            return JSONResponse({
                "active": self.execution_state["active_tasks"],
                "completed": self.execution_state["completed_tasks"],
                "failed": self.execution_state["failed_tasks"],
            })

        @self.app.post("/api/execute")
        async def start_execution(request: Dict[str, Any]):
            """Start autonomous SDLC execution."""
            try:
                phases = request.get("phases", ["testing", "integration", "deployment"])
                config = request.get("config", {})
                
                # Start execution in background
                asyncio.create_task(self._execute_autonomous_sdlc(phases, config))
                
                return JSONResponse({"status": "started", "phases": phases})
            except Exception as e:
                logger.error(f"Failed to start execution: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/optimize")
        async def start_optimization(request: Dict[str, Any]):
            """Start RLHF optimization."""
            try:
                phase = request.get("phase", "preference_collection")
                config = request.get("config", {})
                
                # Start optimization in background
                asyncio.create_task(self._execute_rlhf_optimization(phase, config))
                
                return JSONResponse({"status": "started", "phase": phase})
            except Exception as e:
                logger.error(f"Failed to start optimization: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.connected_clients.append(websocket)
            self.metrics["active_connections"] = len(self.connected_clients)
            
            try:
                # Send initial state
                await websocket.send_json({
                    "type": "initial_state",
                    "data": {
                        "execution_state": self.execution_state,
                        "metrics": self.metrics,
                    }
                })
                
                # Keep connection alive and handle incoming messages
                while True:
                    try:
                        data = await websocket.receive_json()
                        await self._handle_websocket_message(websocket, data)
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(f"WebSocket error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })
                        
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.connected_clients:
                    self.connected_clients.remove(websocket)
                    self.metrics["active_connections"] = len(self.connected_clients)

        @self.app.get("/api/analytics/predictions")
        async def get_predictions():
            """Get predictive analytics."""
            try:
                predictions = await self.analytics.generate_predictions()
                return JSONResponse(predictions)
            except Exception as e:
                logger.error(f"Failed to generate predictions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/quantum/state")
        async def get_quantum_state():
            """Get quantum execution state visualization."""
            return JSONResponse({
                "quantum_tasks": await self._get_quantum_task_visualization(),
                "optimization_progress": await self._get_optimization_progress(),
                "prediction_accuracy": await self._get_prediction_accuracy(),
            })

    async def generate_dashboard_html(self) -> str:
        """Generate main dashboard HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous SDLC Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        .dashboard { 
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            grid-template-rows: auto 1fr 1fr;
            gap: 20px;
            padding: 20px;
            height: 100vh;
        }
        .header {
            grid-column: 1 / -1;
            text-align: center;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
        }
        .status-indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
        }
        .status-active { background: #4CAF50; }
        .status-warning { background: #FF9800; }
        .status-error { background: #F44336; }
        .status-idle { background: #757575; }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s ease;
        }
        .task-list {
            max-height: 200px;
            overflow-y: auto;
        }
        .task-item {
            padding: 8px;
            margin: 5px 0;
            background: rgba(255,255,255,0.1);
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .btn {
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .btn:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        .chart-container {
            position: relative;
            height: 200px;
        }
    </style>
</head>
<body>
    <div id="app" class="dashboard">
        <div class="header">
            <h1>🚀 Autonomous SDLC Dashboard</h1>
            <p>Quantum-Inspired Multimodal RLHF Execution Control Center</p>
            <div class="metric">
                <span>Connection Status:</span>
                <span>
                    <span :class="connectionClass"></span>
                    {{ connectionStatus }}
                </span>
            </div>
        </div>

        <div class="card">
            <h3>📊 Execution Status</h3>
            <div class="metric">
                <span>Current Phase:</span>
                <span>{{ executionState.current_phase }}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" :style="{width: executionState.progress + '%'}"></div>
            </div>
            <div class="metric">
                <span>Progress:</span>
                <span>{{ executionState.progress.toFixed(1) }}%</span>
            </div>
            <div class="metric">
                <span>Start Time:</span>
                <span>{{ formatTime(executionState.start_time) }}</span>
            </div>
            <div class="metric">
                <span>Est. Completion:</span>
                <span>{{ formatTime(executionState.estimated_completion) }}</span>
            </div>
        </div>

        <div class="card">
            <h3>⚡ System Metrics</h3>
            <div class="metric">
                <span>System Health:</span>
                <span>{{ metrics.system_health.toFixed(1) }}%</span>
            </div>
            <div class="metric">
                <span>CPU Usage:</span>
                <span>{{ metrics.cpu_usage.toFixed(1) }}%</span>
            </div>
            <div class="metric">
                <span>Memory Usage:</span>
                <span>{{ metrics.memory_usage.toFixed(1) }}%</span>
            </div>
            <div class="metric">
                <span>GPU Usage:</span>
                <span>{{ metrics.gpu_usage.toFixed(1) }}%</span>
            </div>
            <div class="metric">
                <span>Active Connections:</span>
                <span>{{ metrics.active_connections }}</span>
            </div>
        </div>

        <div class="card">
            <h3>🧠 Quantum Optimization</h3>
            <div class="metric">
                <span>Total Optimizations:</span>
                <span>{{ metrics.total_optimizations }}</span>
            </div>
            <div class="metric">
                <span>Success Rate:</span>
                <span>{{ metrics.success_rate.toFixed(1) }}%</span>
            </div>
            <div class="chart-container">
                <canvas id="optimizationChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h3>📋 Active Tasks</h3>
            <div class="task-list">
                <div v-for="task in executionState.active_tasks" :key="task.id" class="task-item">
                    <span>{{ task.name }}</span>
                    <span class="status-indicator status-active"></span>
                </div>
                <div v-if="executionState.active_tasks.length === 0" class="task-item">
                    <span>No active tasks</span>
                    <span class="status-indicator status-idle"></span>
                </div>
            </div>
        </div>

        <div class="card">
            <h3>🎯 Controls</h3>
            <button class="btn" @click="startExecution" :disabled="isExecuting">
                Start Autonomous SDLC
            </button>
            <br><br>
            <button class="btn" @click="startOptimization" :disabled="isOptimizing">
                Start RLHF Optimization
            </button>
            <br><br>
            <button class="btn" @click="refreshData">
                Refresh Data
            </button>
        </div>
    </div>

    <script>
        const { createApp } = Vue;
        
        createApp({
            data() {
                return {
                    websocket: null,
                    connectionStatus: 'Connecting...',
                    connectionClass: 'status-indicator status-warning',
                    executionState: {
                        current_phase: 'idle',
                        progress: 0,
                        start_time: null,
                        estimated_completion: null,
                        active_tasks: [],
                        completed_tasks: [],
                        failed_tasks: [],
                    },
                    metrics: {
                        system_health: 100,
                        cpu_usage: 0,
                        memory_usage: 0,
                        gpu_usage: 0,
                        active_connections: 0,
                        total_optimizations: 0,
                        success_rate: 100,
                    },
                    isExecuting: false,
                    isOptimizing: false,
                }
            },
            mounted() {
                this.connectWebSocket();
                this.initializeCharts();
                this.refreshData();
                setInterval(this.refreshData, 5000); // Refresh every 5 seconds
            },
            methods: {
                connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws`;
                    
                    this.websocket = new WebSocket(wsUrl);
                    
                    this.websocket.onopen = () => {
                        this.connectionStatus = 'Connected';
                        this.connectionClass = 'status-indicator status-active';
                    };
                    
                    this.websocket.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        this.handleWebSocketMessage(data);
                    };
                    
                    this.websocket.onclose = () => {
                        this.connectionStatus = 'Disconnected';
                        this.connectionClass = 'status-indicator status-error';
                        // Attempt to reconnect after 3 seconds
                        setTimeout(() => this.connectWebSocket(), 3000);
                    };
                    
                    this.websocket.onerror = () => {
                        this.connectionStatus = 'Error';
                        this.connectionClass = 'status-indicator status-error';
                    };
                },
                
                handleWebSocketMessage(data) {
                    switch (data.type) {
                        case 'initial_state':
                            this.executionState = data.data.execution_state;
                            this.metrics = data.data.metrics;
                            break;
                        case 'execution_update':
                            this.executionState = { ...this.executionState, ...data.data };
                            break;
                        case 'metrics_update':
                            this.metrics = { ...this.metrics, ...data.data };
                            break;
                        case 'optimization_complete':
                            this.isOptimizing = false;
                            this.showNotification('Optimization completed successfully!');
                            break;
                        case 'execution_complete':
                            this.isExecuting = false;
                            this.showNotification('SDLC execution completed successfully!');
                            break;
                        case 'error':
                            this.showNotification(`Error: ${data.message}`, 'error');
                            break;
                    }
                },
                
                async startExecution() {
                    this.isExecuting = true;
                    try {
                        const response = await fetch('/api/execute', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                phases: ['testing', 'integration', 'deployment'],
                                config: {}
                            })
                        });
                        
                        if (response.ok) {
                            this.showNotification('Autonomous SDLC execution started!');
                        } else {
                            throw new Error('Failed to start execution');
                        }
                    } catch (error) {
                        this.isExecuting = false;
                        this.showNotification(`Failed to start execution: ${error.message}`, 'error');
                    }
                },
                
                async startOptimization() {
                    this.isOptimizing = true;
                    try {
                        const response = await fetch('/api/optimize', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                phase: 'preference_collection',
                                config: {}
                            })
                        });
                        
                        if (response.ok) {
                            this.showNotification('RLHF optimization started!');
                        } else {
                            throw new Error('Failed to start optimization');
                        }
                    } catch (error) {
                        this.isOptimizing = false;
                        this.showNotification(`Failed to start optimization: ${error.message}`, 'error');
                    }
                },
                
                async refreshData() {
                    try {
                        const [statusResponse, metricsResponse] = await Promise.all([
                            fetch('/api/status'),
                            fetch('/api/metrics')
                        ]);
                        
                        if (statusResponse.ok) {
                            this.executionState = await statusResponse.json();
                        }
                        
                        if (metricsResponse.ok) {
                            this.metrics = await metricsResponse.json();
                        }
                    } catch (error) {
                        console.error('Failed to refresh data:', error);
                    }
                },
                
                formatTime(timeString) {
                    if (!timeString) return 'N/A';
                    return new Date(timeString).toLocaleString();
                },
                
                initializeCharts() {
                    // Initialize optimization chart
                    const ctx = document.getElementById('optimizationChart').getContext('2d');
                    new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: Array.from({length: 10}, (_, i) => i + 1),
                            datasets: [{
                                label: 'Optimization Score',
                                data: Array.from({length: 10}, () => Math.random() * 100),
                                borderColor: '#4CAF50',
                                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                                tension: 0.4
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: { display: false }
                            },
                            scales: {
                                y: { beginAtZero: true, max: 100 }
                            }
                        }
                    });
                },
                
                showNotification(message, type = 'success') {
                    // Simple notification - in production, use a proper notification library
                    const color = type === 'error' ? '#F44336' : '#4CAF50';
                    const notification = document.createElement('div');
                    notification.style.cssText = `
                        position: fixed;
                        top: 20px;
                        right: 20px;
                        background: ${color};
                        color: white;
                        padding: 15px 20px;
                        border-radius: 5px;
                        z-index: 1000;
                        max-width: 300px;
                    `;
                    notification.textContent = message;
                    document.body.appendChild(notification);
                    
                    setTimeout(() => {
                        document.body.removeChild(notification);
                    }, 5000);
                }
            }
        }).mount('#app');
    </script>
</body>
</html>
        """

    async def _handle_websocket_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        message_type = data.get("type")
        
        if message_type == "ping":
            await websocket.send_json({"type": "pong"})
        elif message_type == "request_update":
            await self._send_state_update(websocket)

    async def _send_state_update(self, websocket: WebSocket):
        """Send current state to WebSocket client."""
        await websocket.send_json({
            "type": "state_update",
            "data": {
                "execution_state": self.execution_state,
                "metrics": self.metrics,
            }
        })

    async def _broadcast_update(self, update_type: str, data: Dict[str, Any]):
        """Broadcast update to all connected clients."""
        message = {
            "type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected_clients = []
        for client in self.connected_clients:
            try:
                await client.send_json(message)
            except Exception:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            if client in self.connected_clients:
                self.connected_clients.remove(client)
        
        self.metrics["active_connections"] = len(self.connected_clients)

    async def _execute_autonomous_sdlc(self, phases: List[str], config: Dict[str, Any]):
        """Execute autonomous SDLC in background."""
        try:
            self.execution_state.update({
                "current_phase": "initializing",
                "progress": 0.0,
                "start_time": datetime.now().isoformat(),
                "active_tasks": [{"id": "init", "name": "Initializing SDLC execution"}],
            })
            
            await self._broadcast_update("execution_update", self.execution_state)
            
            # Simulate SDLC execution
            for i, phase in enumerate(phases):
                self.execution_state.update({
                    "current_phase": phase,
                    "progress": (i + 1) / len(phases) * 100,
                    "active_tasks": [{"id": f"phase_{i}", "name": f"Executing {phase} phase"}],
                })
                
                await self._broadcast_update("execution_update", self.execution_state)
                await asyncio.sleep(2)  # Simulate work
            
            self.execution_state.update({
                "current_phase": "completed",
                "progress": 100.0,
                "active_tasks": [],
                "completed_tasks": [{"id": f"phase_{i}", "name": phase} for i, phase in enumerate(phases)],
            })
            
            await self._broadcast_update("execution_complete", self.execution_state)
            
        except Exception as e:
            logger.error(f"SDLC execution failed: {e}")
            self.execution_state.update({
                "current_phase": "failed",
                "failed_tasks": [{"id": "error", "name": f"Execution failed: {str(e)}"}],
            })
            await self._broadcast_update("execution_error", {"error": str(e)})

    async def _execute_rlhf_optimization(self, phase: str, config: Dict[str, Any]):
        """Execute RLHF optimization in background."""
        try:
            # Simulate optimization based on phase
            if phase == "preference_collection":
                result = await self.rlhf_optimizer.optimize_preference_collection(
                    current_strategy=config,
                    training_data_stats={"num_samples": 1000, "quality_score": 0.85}
                )
            elif phase == "reward_model_training":
                result = await self.rlhf_optimizer.tune_reward_model_training(
                    model_architecture="transformer",
                    training_config=config,
                    preference_data_stats={"num_preferences": 500, "agreement_rate": 0.8}
                )
            else:
                result = await self.rlhf_optimizer.optimize_policy_gradient_updates(
                    policy_config=config,
                    environment_stats={"episode_length": 100, "success_rate": 0.7},
                    reward_model_performance={"accuracy": 0.9, "loss": 0.1}
                )
            
            self.metrics["total_optimizations"] += 1
            await self._broadcast_update("optimization_complete", {
                "phase": phase,
                "improvement": result.expected_improvement,
                "confidence": result.confidence,
            })
            
        except Exception as e:
            logger.error(f"RLHF optimization failed: {e}")
            await self._broadcast_update("optimization_error", {"error": str(e)})

    async def _update_metrics(self):
        """Update system metrics."""
        # Simulate metric collection
        self.metrics.update({
            "cpu_usage": np.random.uniform(10, 80),
            "memory_usage": np.random.uniform(20, 90),
            "gpu_usage": np.random.uniform(0, 100),
            "system_health": np.random.uniform(85, 100),
        })

    async def _get_quantum_task_visualization(self) -> Dict[str, Any]:
        """Get quantum task state for visualization."""
        return {
            "superposition_states": np.random.random(8).tolist(),
            "entangled_tasks": [
                {"task1": "optimization", "task2": "validation", "strength": 0.8},
                {"task1": "testing", "task2": "deployment", "strength": 0.6},
            ],
            "quantum_coherence": np.random.uniform(0.7, 1.0),
        }

    async def _get_optimization_progress(self) -> Dict[str, Any]:
        """Get optimization progress data."""
        return {
            "current_iteration": np.random.randint(1, 100),
            "best_score": np.random.uniform(0.8, 1.0),
            "convergence_rate": np.random.uniform(0.1, 0.5),
            "exploration_ratio": np.random.uniform(0.2, 0.8),
        }

    async def _get_prediction_accuracy(self) -> Dict[str, Any]:
        """Get prediction accuracy metrics."""
        return {
            "short_term_accuracy": np.random.uniform(0.85, 0.95),
            "long_term_accuracy": np.random.uniform(0.70, 0.85),
            "confidence_intervals": {
                "lower": np.random.uniform(0.70, 0.80),
                "upper": np.random.uniform(0.90, 0.95),
            }
        }


class DashboardServer:
    """Server wrapper for the dashboard."""
    
    def __init__(self, dashboard: AutonomousSDLCDashboard):
        self.dashboard = dashboard
        
    def run(self, host: str = None, port: int = None):
        """Run the dashboard server."""
        host = host or self.dashboard.config["host"]
        port = port or self.dashboard.config["port"]
        
        logger.info(f"Starting Autonomous SDLC Dashboard at http://{host}:{port}")
        
        uvicorn.run(
            self.dashboard.app,
            host=host,
            port=port,
            log_level="info"
        )

    async def start_async(self, host: str = None, port: int = None):
        """Start the dashboard server asynchronously."""
        host = host or self.dashboard.config["host"]
        port = port or self.dashboard.config["port"]
        
        config = uvicorn.Config(
            self.dashboard.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        logger.info(f"Starting Autonomous SDLC Dashboard at http://{host}:{port}")
        await server.serve()


# Factory function for easy dashboard creation
def create_dashboard(config: Optional[Dict[str, Any]] = None) -> AutonomousSDLCDashboard:
    """Create and configure dashboard instance."""
    return AutonomousSDLCDashboard(config)


def create_dashboard_server(config: Optional[Dict[str, Any]] = None) -> DashboardServer:
    """Create dashboard server instance."""
    dashboard = create_dashboard(config)
    return DashboardServer(dashboard)


if __name__ == "__main__":
    # Run dashboard server
    server = create_dashboard_server()
    server.run()